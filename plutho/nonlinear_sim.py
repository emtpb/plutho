"""Helper class to organize the usage of the piezo nonlinear simulations"""
# Python standard libraries
import os
from typing import Union, Dict, List
import configparser
import json

# Third party libraries
import numpy as np
import numpy.typing as npt

# Local libraries
from plutho import Mesh, MaterialManager, FieldType, MaterialData, \
    MeshData, SimulationData
from .simulation import NonlinearPiezoSimStationary, NonlinearPiezoSimTime, \
    NonlinearType


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
    solver: Union[None, NonlinearPiezoSimTime, NonlinearPiezoSimStationary]
    nonlinear_material_matrix: npt.NDArray
    boundary_conditions: List[Dict]

    def __init__(
        self,
        working_directory: str,
        sim_name: str,
        mesh: Mesh
    ):
        self.sim_name = sim_name
        self.sim_directory = os.path.join(working_directory, sim_name)

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
        self.charge_calculated = False

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

    def clear_dirichlet_bcs(self):
        """Resets the dirichlet boundary conditions."""
        self.boundary_conditions = []
        self.dirichlet_nodes = []
        self.dirichlet_values = []

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

    def set_nonlinearity_type(self, ntype: NonlinearType, **kwargs):
        """Sets the type of nonlinearity for the simulation.

        Parameters:
            ntype: NonlinearType object.
            kwargs: Can contain different parameters based on the nonlinear
                type:
                - If ntype=NonlinearType.Rayleigh:
                    "zeta" (float): Scalar nonlinearity parameter.
                - If ntype=NonlinearType.Custom:
                    "nonlinear_matrix" (npt.NDArray): 6x6 custom nonlinear
                        material matrix.
        """
        # TODO (almost) Duplicate code with nonlinear/piezo_time.py
        if ntype is NonlinearType.Rayleigh:
            if "zeta" not in kwargs:
                raise ValueError(
                    "Missing 'zeta' parameter for nonlinear type: Rayleigh"
                )
        elif ntype is NonlinearType.Custom:
            if "nonlinear_matrix" not in kwargs:
                raise ValueError(
                    "Nonlinearity matrix is missing as parameter for curstom "
                    "nonlinearity"
                )
        else:
            raise NotImplementedError(
                f"Nonlinearity type {NonlinearType.value} is not implemented"
            )

        self.nonlinear_type = ntype
        self.nonlinear_params = kwargs


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
            - "electrode_normals" (npt.NDArray): List of normal vectors
                corresponding to each electrode element (same index).
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
                given load can be adjusted to achieve a faster convergence.
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

        # Assemble
        self.solver.assemble(
            self.nonlinear_type,
            nonlinear_matrix=self.nonlinear_material_matrix
        )

        # Run simulation
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

        np.save(
            os.path.join(
                self.sim_directory,
                f"{file_prefix}u.npy"
            ),
            self.solver.u
        )

        if isinstance(self.solver, NonlinearPiezoSimTime):
            if self.charge_calculated:
                np.save(
                    os.path.join(
                        self.sim_directory,
                        f"{file_prefix}q.npy"
                    ),
                    self.solver.q
                )

    def save_simulation_settings(self, prefix: str = ""):
        if prefix != "":
            prefix += "_"

        if not os.path.exists(self.sim_directory):
            os.makedirs(self.sim_directory)

        settings = configparser.ConfigParser()

        if isinstance(self.solver, NonlinearPiezoSimStationary):
            simulation_type = "NonlinearStationary"
        elif isinstance(self.solver, NonlinearPiezoSimTime):
            simulation_type = "NonlinearTime"
        else:
            raise ValueError(
                f"Cannot save simulation type {type(self.solver)}"
            )

        # General simulation settings
        general_settings = {
            "name": self.sim_name,
            "mesh_file": self.mesh.mesh_file_path,
            "simulation_type": simulation_type
        }
        settings["general"] = general_settings

        # Material setings and data
        material_settings = {}
        for material in self.material_manager.materials:
            material_settings[material.material_name] = material.to_dict()
        material_settings["nonlinear_piezo_matrix"] = \
            self.nonlinear_material_matrix.tolist()

        # Simulation data
        if isinstance(self.solver, NonlinearPiezoSimTime):
            settings["simulation"] = self.solver.simulation_data.__dict__

        # Boundary conditions
        boundary_conditions = {}
        for index, bc in enumerate(self.boundary_conditions):
            boundary_conditions[index] = bc

        # Save simulation data to config file
        with open(
            os.path.join(
                self.sim_directory,
                f"{prefix}{self.sim_name}.cfg"
            ),
            "w",
            encoding="UTF-8"
        ) as fd:
            settings.write(fd)

        # Save materials to json
        with open(
            os.path.join(
                self.sim_directory,
                f"{prefix}materials.json"
            ),
            "w",
            encoding="UTF-8"
        ) as fd:
            json.dump(material_settings, fd, indent=2)

        # Save boundary conditions to json
        with open(
            os.path.join(
                self.sim_directory,
                f"{prefix}boundary_conditions.json"
            ),
            "w",
            encoding="UTF-8"
        ) as fd:
            json.dump(boundary_conditions, fd, indent=2)

    @staticmethod
    def load_simulation_settings(
        simulation_folder: str,
        prefix: str = ""
    ):
        if prefix != "":
            prefix += "_"

        simulation_name = os.path.basename(simulation_folder)

        # Check if folder has all necessary files
        necessary_files = [
            os.path.join(simulation_folder, f"{prefix}{simulation_name}.cfg"),
            os.path.join(simulation_folder, f"{prefix}materials.json"),
            os.path.join(
                simulation_folder,
                f"{prefix}boundary_conditions.json"
            )
        ]
        for file_path in necessary_files:
            if not os.path.isfile(file_path):
                raise IOError(f"{file_path} does not exist.")

        settings = configparser.ConfigParser()
        settings.read(necessary_files[0])

        # Read general simulation settings
        mesh_file = settings["general"]["mesh_file"]
        simulation_type = settings["general"]["simulation_type"]
        simulation = PiezoNonlinear(
            simulation_folder,
            "",
            Mesh(mesh_file)
        )
        # Workaround since simulation_folder and simulation_name are
        # combined in the constructor, see empty string above for the
        # constructor
        simulation.sim_name = simulation_name

        if simulation_type == "NonlinearStationary":
            simulation.setup_stationary_simulation()
        elif simulation_type == "NonlinearTime":
            simulation_data = SimulationData(
                float(settings["simulation"]["delta_t"]),
                int(settings["simulation"]["number_of_time_steps"]),
                float(settings["simulation"]["gamma"]),
                float(settings["simulation"]["beta"]),
            )
            simulation.setup_time_dependent_simulation(
                simulation_data.delta_t,
                simulation_data.number_of_time_steps,
                simulation_data.gamma,
                simulation_data.beta
            )
        else:
            raise IOError(
                f"Cannot deserialize {simulation_type} simulation type"
            )

        # Read materials
        with open(necessary_files[1], "r", encoding="UTF-8") as fd:
            material_settings = json.load(fd)
            for material_name in material_settings.keys():
                if material_name == "nonlinear_piezo_matrix":
                    simulation.nonlinear_material_matrix = np.array(
                        material_settings[material_name]
                    )
                else:
                    simulation.add_material(
                        material_name,
                        MaterialData(
                            **material_settings[material_name]["material_data"]
                        ),
                        material_settings[material_name]["physical_group_name"]
                    )
            simulation.material_manager.initialize_materials()

        # Read boundary conditions
        with open(necessary_files[2], "r", encoding="UTF-8") as fd:
            bc_settings = json.load(fd)
            for index in bc_settings.keys():
                field_type = getattr(
                    FieldType,
                    bc_settings[index]["field_type"]
                )
                physical_group = bc_settings[index]["physical_group"]
                values = np.array(bc_settings[index]["values"])

                simulation.add_dirichlet_bc(
                    field_type,
                    physical_group,
                    values
                )

        return simulation

    def load_simulation_results(self):
        if isinstance(self.solver, NonlinearPiezoSimStationary):
            u_file = os.path.join(
                self.sim_directory,
                "u.npy"
            )
            self.solver.u = np.load(u_file)
        elif isinstance(self.solver, NonlinearPiezoSimTime):
            u_file = os.path.join(
                self.sim_directory,
                "u.npy"
            )
            q_file = os.path.join(
                self.sim_directory,
                "q.npy"
            )

            self.solver.u = np.load(u_file)
            if os.path.isfile(q_file):
                self.solver.q = np.load(q_file)
        else:
            raise ValueError(
                "Cannot load simulation settings of simulation type",
                type(self.solver)
            )
