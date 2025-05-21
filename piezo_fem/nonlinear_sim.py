"""Helper class to organize the usage of the piezo nonlinear simulations"""
# Python standard libraries
import os

# Third party libraries
import numpy as np
import numpy.typing as npt

# Local libraries
import piezo_fem as pfem


def create_6x6_axisymmetric(c_tilde):
    """Converts the 4x4 axisymmetric stiffness matrix given the 6x6 matrix in
    Voigt-Notation.

    Parameters:
        c_tilde: 6x6 Matrix in Voigt-Notation
    """
    return np.array([
        [c_tilde[0, 0], c_tilde[0, 2], 0, c_tilde[0, 1]],
        [c_tilde[0, 2], c_tilde[2, 2], 0, c_tilde[0, 2]],
        [0, 0, c_tilde[3, 3], 0],
        [c_tilde[0, 1], c_tilde[0, 2], 0, c_tilde[0, 0]]
    ])


class PiezoNonlinearStationary:
    """Wrapper class to handle stationary nonlinear simulations using pfem."""

    # Attributes
    sim_name: str
    sim_directory: str
    mesh: pfem.Mesh
    material_manager: pfem.MaterialManager

    def __init__(
        self,
        sim_name,
        base_folder,
        mesh_file
    ):
        self.sim_name = sim_name
        self.sim_directory = os.path.join(base_folder, sim_name)

        if not os.path.isdir(self.sim_directory):
            os.makedirs(self.sim_directory)

        if os.path.exists(mesh_file):
            self.mesh = pfem.Mesh(
                    mesh_file,
                    True
            )
        else:
            self.mesh = pfem.Mesh(
                mesh_file,
                False
            )
            self.mesh.generate_rectangular_mesh(
                mesh_size=0.0001
            )

    def set_material(
        self,
        material_name: str,
        material_data: pfem.MaterialData
    ):
        # Create new material manager
        _, elements = self.mesh.get_mesh_nodes_and_elements()
        self.material_manager = pfem.MaterialManager(len(elements))

        # Apply material for every element
        self.material_manager.add_material(
            material_name,
            material_data,
            "All",
            np.arange(self.material_manager.number_of_elements)
        )
        self.material_manager.initialize_materials()

    def run_simulation(
        self,
        excitations: npt.NDArray,
        is_linear: bool,
        is_disc: bool,
        nonlinear_matrix_factor: float,
        nonlinear_matrix: npt.NDArray
    ):
        # Check if materials are set
        if self.material_manager is None:
            raise ValueError(
                "Cannot run nonlinear simulation. Please set a"
                " material before running the simulation."
            )

        # Get mesh data
        nodes, elements = self.mesh.get_mesh_nodes_and_elements()
        number_of_nodes = len(nodes)

        # Setup simulation
        sim = pfem.NonlinearPiezoSimStationary(
            pfem.MeshData(
                nodes,
                elements
            ),
            self.material_manager
        )

        # Setup nonlinear elasticity matrix
        nonlinear_ela_matrix = nonlinear_matrix_factor * \
            create_6x6_axisymmetric(
                nonlinear_matrix
            )
        if is_linear:
            print("Linear simulation")
        else:
            print("Nonlinear matrix:\n", nonlinear_ela_matrix)

        sim.assemble(nonlinear_ela_matrix)

        # Get boundary condition node indices
        node_indices = self.mesh.get_nodes_by_physical_groups(
            ["Electrode", "Ground", "Symaxis"]
        )

        self.u_combined = {}
        u = None
        previous_u = None
        for excitation in excitations:
            # Setup dirichlet boundary conditions
            dirichlet_nodes = []
            dirichlet_values = []

            print("Excitation:", excitation)
            for index in node_indices["Electrode"]:
                dirichlet_nodes.append(index+2*number_of_nodes)
                dirichlet_values.append(excitation)

            for index in node_indices["Ground"]:
                dirichlet_nodes.append(index+2*number_of_nodes)
                dirichlet_values.append(0)

            # This sets the u_z component of the (0,0) node to 0 in order to
            # make the solution unique
            dirichlet_nodes.append(1)
            dirichlet_values.append(0)

            if not is_disc:
                # If its not a disc this must also be set for the u_r component
                dirichlet_nodes.append(0)
                dirichlet_values.append(0)
            else:
                for index in node_indices["Symaxis"]:
                    dirichlet_nodes.append(2*index)  # Since only u_r
                    dirichlet_values.append(0)

            sim.dirichlet_nodes = np.array(dirichlet_nodes)
            sim.dirichlet_values = np.array(dirichlet_values)

            if is_linear:
                u = sim.solve_linear()
            else:
                u = sim.solve_newton(
                    tolerance=1e-12,
                    max_iter=300,
                    u_start=previous_u
                )

            previous_u = u.copy()

            np.save(
                os.path.join(
                    self.sim_directory,
                    f"u_stat_comb_{excitation}V.npy"
                ),
                u
            )

            self.u_combined[excitation] = u
