"""Helper class to organize the usage of the piezo nonlinear simulations"""
# Python standard libraries
import os
from typing import Union, Dict, List

# Third party libraries
import numpy as np
import numpy.typing as npt

# Local libraries
from piezo_fem import Mesh, MaterialManager, FieldType, MaterialData, \
    MeshData, SimulationData
from .simulation import NonlinearPiezoSimStationary, NonlinearPiezoSimTime


class PiezoNonlinear:
    """Wrapper class to handle stationary and time dependent nonlinear
    simulations.

    Attributes:
        sim_name: Name of the simulation and the simulation folder.
        sim_directory: Path to the directory in which the result files
            are saved.
        mesh: Mesh object containing the gmsh mesh information.
        material_manger MaterialManager object.

    Parameters:
        sim_name: Name of the simulation and the simulation folder which is
            created.
        base_directory: Path to a directory in which the simulation directory
            is created.
        mesh_file: Path to a mesh file which shall be used in this simulation.
    """

    # Simulation settings
    sim_name: str
    sim_directory: str
    mesh: Mesh
    material_manager: MaterialManager
    solver: Union[NonlinearPiezoSimTime, NonlinearPiezoSimStationary]
    nonlinear_material_matrix: npt.NDArray
    boundary_conditions: List[Dict]

    def __init__(
        self,
        sim_name: str,
        base_directory: str,
        mesh: Mesh
    ):
        self.sim_name = sim_name
        self.sim_directory = os.path.join(base_directory, sim_name)

        if not os.path.isdir(self.sim_directory):
            os.makedirs(self.sim_directory)

        self.boundary_conditions = []
        self.dirichlet_nodes = []
        self.dirichlet_values = []
        self.nonlinear_material_matrix = np.zeros((4, 4))
        self.solver = None
        self.mesh = mesh
        nodes, elements = mesh.get_mesh_nodes_and_elements()
        self.mesh_data = MeshData(nodes, elements)
        self.material_manager = MaterialManager(len(elements))

    def add_material(
        self,
        material_name: str,
        material_data: MaterialData,
        physical_group_name: str
    ):
        """Adds a material to the simulation

        Parameters:
            material_name: Name of the material.
            material_data: Material data.
            physical_group_name: Name of the physical group for which the
                material shall be set. If this is None or "" the material will
                be set for the whole model.
        """
        if physical_group_name is None or physical_group_name == "":
            element_indices = np.arange(len(self.mesh_data.elements))
        else:
            element_indices = self.mesh.get_elements_by_physical_groups(
                [physical_group_name]
            )[physical_group_name]

        self.material_manager.add_material(
            material_name,
            material_data,
            physical_group_name,
            element_indices
        )

    def add_dirichlet_bc(
        self,
        field_type: FieldType,
        physical_group_name: str,
        values: npt.NDArray
    ):
        """Adds a dirichlet boundary condition (dirichlet bc) to the
        simulation. The given values are for each time step. Therefore each
        value is applied to every node from the given physical_group.

        Parameters:
            field_type: The type of field for which the bc shall be set.
            physical_group_name: Name of the physical group for which the
                boundary condition shall be set.
            values: List of values per time step. Value of the bc for each time
                step. The value for one time step is applied to every element
                equally. Length: number_of_time_step.
        """
        # Save boundary condition for serialization
        self.boundary_conditions.append({
            "field_type": field_type.name,
            "physical_group": physical_group_name,
            "values": values.tolist()
        })

        # Apply the given values for all nodes from the given physical_group
        node_indices = self.mesh.get_nodes_by_physical_groups(
            [physical_group_name]
        )[physical_group_name]

        number_of_nodes = len(self.mesh_data.nodes)

        for node_index in node_indices:
            real_index = 0

            # Depending on the variable type and the simulation types
            # the corresponding field variable may be found at different
            # positions of the solution vector
            if field_type is FieldType.PHI:
                real_index = 2*number_of_nodes+node_index
            elif field_type is FieldType.U_R:
                real_index = 2*node_index
            elif field_type is FieldType.U_Z:
                real_index = 2*node_index+1
            else:
                raise ValueError(f"Unknown variable type {field_type}")

            self.dirichlet_nodes.append(real_index)
            self.dirichlet_values.append(values)

    def setup_stationary_simulation(self):
        """Set the simulation type to a nonlinaer stationary simulation.
        """
        self.solver = NonlinearPiezoSimStationary(
            self.mesh_data,
            self.material_manager
        )

    def setup_time_dependent_simulation(
        self,
        delta_t: float,
        number_of_time_steps: int,
        gamma: float,
        beta: float
    ):
        """Sets the simulation type to a nonlinear time dependent simulation.

        Parameters:
            delta_t: Difference between each time step.
            number_of_time_steps: Total number of time steps which are
                simulated.
            gamma: Time integration parameter (best value = 0.5).
            beta: Time integration parameter (best value = 0.25).
            """
        self.solver = NonlinearPiezoSimTime(
            self.mesh_data,
            self.material_manager,
            SimulationData(
                delta_t,
                number_of_time_steps,
                gamma,
                beta
            )
        )

    def set_nonlinear_material_6x6(self, mmatrix: npt.NDArray):
        """Sets the nonlinear matrix matrix based on a given 6x6 C matrix.

        Parameters:
            mmatrix: Material matrix 6x6.
        """
        self.nonlinear_material_matrix = np.array([
            [mmatrix[0, 0], mmatrix[0, 2], 0, mmatrix[0, 1]],
            [mmatrix[0, 2], mmatrix[2, 2], 0, mmatrix[0, 2]],
            [0, 0, mmatrix[3, 3], 0],
            [mmatrix[0, 1], mmatrix[0, 2], 0, mmatrix[0, 0]]
        ])

    def simulate(
        self,
        **kwargs
    ):
        """Runs the simulation using the previously set materials, boundary
        conditions and simulation type. The simulation results are stored in
        the solver object.

        Paramters:
            The kwargs parameter can have the following attributes:
            - "tolerance" (float): Can be set for either simulation.
                Sets the tolerance
                for the Newton-Raphson scheme (norm of the reisual must be
                smaller in order to stop the algorithm). Typical values can be
                in the order of 1e-11.
            - "max_iter" (int): Can be set for either simulation.
                Sets the maximum number of iterations which are done until
                the algorithm stops. If the tolerance is met it is stopped
                earlier.
            - "electrode_elements" (npt.NDArray): Can only be set for the time
                dependent simulation. If the charge shall be calculated it is
                necessary to set this parameter. Set it to the indices of the
                elements on which the charge shall be calculated.
            - "u_start" (npt.NDArray): Can only be set for the stationary
                simulation. Gives a initial guess for u to achieve a faster
                convergence.
            - "alpha" (float): Can be set for either simulation. Factor which
                is multiplied with the delta_u of each iteration.
                Typically a value between 0 and 1. Implements a sort of
                damping in each iteration which can result in a better
                convergence.
            - "load_factor" (float): Can only be set for the stationary
                simulation. Is multiplied with the load vector. Therefore the
                given can be adjusted to achieve a faster convergence.
        """
        # Check if materials are set
        if self.material_manager is None:
            raise ValueError(
                "Cannot run nonlinear simulation. Please set a"
                " material before running the simulation."
            )

        # Check if solvers are set
        if self.solver is None:
            raise ValueError(
                "Cannot run nonlinear simulation. Please set a"
                " simulation type before running the simulation"
            )

        self.material_manager.initialize_materials()
        self.solver.dirichlet_nodes = np.array(self.dirichlet_nodes)
        self.solver.dirichlet_values = np.array(self.dirichlet_values)

        # Run the simulation
        self.solver.assemble(self.nonlinear_material_matrix)

        if isinstance(self.solver, NonlinearPiezoSimTime):
            if kwargs["electrode_elements"] is not None:
                self.charge_calculated = True
            self.solver.solve_time_implicit(
                **kwargs
            )
        elif isinstance(self.solver, NonlinearPiezoSimStationary):
            self.solver.solve_newton(
                **kwargs
            )
        else:
            raise ValueError(
                "Cannot run simulation for simulation type"
                f" {type(self.solver)}"
            )

    def save_simulation_results(self, file_prefix: str = ""):
        """Save the simulation results to the simulation folder. If a prefix
        is given this is prepend to the name of the output file.

        Parameters:
            file_prefix: String which is prepend to the output file names.
        """
        if not os.path.exists(self.sim_directory):
            os.makedirs(self.sim_directory)

        if file_prefix != "":
            file_prefix += "_"

        if isinstance(self.solver, NonlinearPiezoSimTime):
            np.save(
                os.path.join(
                    self.sim_directory,
                    f"{file_prefix}u.npy"
                ),
                self.solver.u
            )
        elif isinstance(self.solver, NonlinearPiezoSimStationary):
            np.save(
                os.path.join(
                    self.sim_directory,
                    f"{file_prefix}u.npy"
                ),
                self.solver.u
            )
            if self.charge_calculated:
                np.save(
                    os.path.join(
                        self.sim_directory,
                        f"{file_prefix}q.npy"
                    ),
                    self.solver.q
                )
