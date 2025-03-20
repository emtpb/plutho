"""Basic module to handle single simulations."""
# Python standard libraries
import os
from enum import Enum
from typing import Union, List, Dict
import configparser
import json

# Third party libraries
import numpy as np
import numpy.typing as npt

# Local libraries
from .simulation.base import MeshData, SimulationData
from .simulation import ThermoSimTime, PiezoSimTime, ThermoPiezoSimTime, \
    PiezoSimFreq
from .materials import MaterialData, MaterialManager
from .mesh import Mesh


class SimulationException(Exception):
    """Custom exception to simplify errors."""


class FieldType(Enum):
    """Possible field types which are calculated using differnet simulations.
    """
    U_R = 0
    U_Z = 1
    PHI = 2
    THETA = 3


class SingleSimulation:
    """Base class to handle single simulations. Multiple different simulation
    types can be chosen.

    Arguments:
        simulation_name: Name of the current simulation.
        simulation_directory: Directory where the results files are saved.
            Combination of working_directory given to constructor and the
            simulation name.
        material_manager: MaterialManager object used to handle different
            materials as well as temperature dependent material parameters.
        mesh: Contains the mesh.
        mesh_data: Contains the nodes and element arrays of the mesh.
        dirichlet_nodes: Array of node indices at which a dirichlet boundary
            shall be set.
        dirichlet_values: Array of time dependent values which are set
            at the corresponding dirichlet_node: [node_index, time_index].
        solver: Contains the simulation solver class.
        charge_calculated: Is True if a charge is calculated in the simulation.
    """
    # Basic settings
    simulation_name: str
    simulation_directory: str

    # Materials
    material_manager: MaterialManager

    # Mesh
    mesh: Mesh
    mesh_data: MeshData

    # Dirichlet bc
    boundary_conditions: List[Dict]  # Used for serialization
    dirichlet_nodes: List[int]
    dirichlet_values: List[npt.NDArray]

    # Simulation
    solver: Union[ThermoSimTime, PiezoSimFreq, PiezoSimTime, ThermoPiezoSimTime]
    charge_calculated: bool
    mech_loss_calculated: bool

    def __init__(
        self,
        working_directory: str,
        simulation_name: str,
        mesh: Mesh
    ):
        simulation_directory = os.path.join(working_directory, simulation_name)

        self.simulation_directory = simulation_directory
        self.simulation_name = simulation_name
        self.mesh = mesh

        nodes, elements = mesh.get_mesh_nodes_and_elements()
        self.mesh_data = MeshData(nodes, elements)
        self.material_manager = MaterialManager(len(elements))
        self.charge_calculated = False
        self.mech_loss_calculated = False
        self.boundary_conditions = []

        # Initialize dirichlet bc arrays
        self.dirichlet_nodes = []
        self.dirichlet_values = []

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
        physical_group: str,
        values: npt.NDArray
    ):
        """Adds a dirichlet boundary condition (dirichlet bc) to the simulation.

        Parameters:
            field_type: The type of field for which the bc shall be set.
            physical_group: Name of the physical group for which the boundary
                condition shall be set.
            values: List of values per time step. Value of the bc for each time
                step. The value for one time step is applied to every element
                equally.
        """
        if isinstance(self.solver, ThermoSimTime):
            if (field_type is FieldType.U_R or
                    field_type is FieldType.U_Z or
                    field_type is FieldType.PHI):
                raise ValueError(
                    f"Unknown variable type {field_type} given for"
                    f"simulation type {type(self.solver)}")
        elif (isinstance(self.solver, PiezoSimFreq) or
                isinstance(self.solver, PiezoSimTime)):
            if field_type is FieldType.THETA:
                raise ValueError(
                    f"Unknown variable type {field_type} given for"
                    f"simulation type {type(self.solver)}")

        self.boundary_conditions.append({
            "field_type": field_type.name,
            "physical_group": physical_group,
            "values": values.tolist()
        })

        # TODO Rename for simpler understanding that the values are applied
        # to all given nodes in the physical group equally
        node_indices = self.mesh.get_nodes_by_physical_groups(
            [physical_group]
        )[physical_group]

        number_of_nodes = len(self.mesh_data.nodes)
        for node_index in node_indices:
            real_index = 0
            # Depending on the variable type and the simulation types
            # the corresponding field variable may be found at different
            # positions of the solution vector
            if field_type is FieldType.PHI:
                real_index = 2*number_of_nodes+node_index
            elif field_type is FieldType.THETA:
                if isinstance(self.solver, ThermoSimTime):
                    real_index = node_index
                else:
                    real_index = 3*number_of_nodes+node_index
            elif field_type is FieldType.U_R:
                real_index = 2*node_index
            elif field_type is FieldType.U_Z:
                real_index = 2*node_index+1
            else:
                raise ValueError(f"Unknown variable type {field_type}")

            self.dirichlet_nodes.append(real_index)
            self.dirichlet_values.append(values)

    def setup_thermo_time_domain(
        self,
        delta_t: float,
        number_of_time_steps: int,
        gamma: float
    ):
        """Sets the simulation to a thermal simulation in the time domain.

        Parameters:
            delta_t: Difference between the time steps.
            number_of_time_steps: Total number of time steps.
            gamma: Time integration parameter (Set to 0.5 for unconditionally
                stable simulation).
        """
        # TODO Change parameter to simulation data?
        self.solver = ThermoSimTime(
            self.mesh_data,
            self.material_manager,
            SimulationData(
                delta_t,
                number_of_time_steps,
                gamma,
                0.0
            )
        )

    def setup_piezo_time_domain(
        self,
        simulation_data: SimulationData
    ):
        """Sets a electro-mechanical (piezo) simulation in the time domain.

        Parameters:
            simulation_data: Simulation data object.
        """
        self.solver = PiezoSimTime(
            self.mesh_data,
            self.material_manager,
            simulation_data
        )

    def setup_piezo_freq_domain(self, frequencies: npt.NDArray):
        """Sets a electro-mechanical (piezo) simulation in the frequency
        domain

        Parameters:
            frequencies: Frequencies which shall be simulated.
        """
        self.solver = PiezoSimFreq(
            self.mesh_data,
            self.material_manager,
            frequencies
        )

    def setup_thermal_piezo_time_domain(
        self,
        simulation_data: SimulationData
    ):
        """Sets a thermo-piezoelectrical simulation.

        Parameters:
            simulation_data: SimulationData object.
        """
        self.solver = ThermoPiezoSimTime(
            self.mesh_data,
            self.material_manager,
            simulation_data
        )

    def simulate(self, **kwargs):
        """Runs a simulation. Note that a simulation must be setup
        beforehand.

        Parameters:
            **kwargs:
                'starting_temperature': Sets a starting temperature field.
                'initialize_materials': The materials are initialized with
                    the given starting_temperature field.
                'theta_start': Sets the starting field for the pure thermal
                    simulation.
                'time_step': Sets a starting time index for the pure thermal
                    simulation.
                'electrode_elements': List of element indices for which the
                    charge is calculated.
                'calculate_mech_loss': Set to true of the mech loss shall
                    be calculated.
        """

        # Check if solver is set
        if self.solver == None:
            raise SimulationException(
                "Please setup a solver first using any "
                "setup function"
            )

        starting_temperature = None
        if "starting_temperature" in kwargs:
            start_temp = kwargs["starting_temperature"]
            if isinstance(start_temp, int) or isinstance(start_temp, float):
                starting_temperature = start_temp*np.ones(
                    len(self.mesh_data.elements)
                )
            else:
                starting_temperature = kwargs["starting_temperature"]
        if ("initialize_materials" not in kwargs
                or kwargs["initialize_materials"]):
            self.solver.material_manager.initialize_materials(
                starting_temperature
            )
        self.solver.dirichlet_nodes = np.array(self.dirichlet_nodes)
        self.solver.dirichlet_values = np.array(self.dirichlet_values)
        self.solver.assemble()
        if isinstance(self.solver, ThermoSimTime):
            theta_start = None
            # TODO Rename theta start and starting temperature
            if "theta_start" in kwargs:
                theta_start = kwargs["theta_start"]
            if "time_step" in kwargs:
                return self.solver.solve_until_material_parameters_change(
                    theta_start,
                    kwargs["time_step"]
                )
            else:
                self.solver.solve_time(
                    theta_start
                )
        elif isinstance(self.solver, PiezoSimFreq):
            electrode_elements = None
            if "electrode_elements" in kwargs:
                electrode_elements = kwargs["electrode_elements"]
                self.charge_calculated = True
            calculate_mech_loss = False
            if "calculate_mech_loss" in kwargs:
                calculate_mech_loss = True
                self.mech_loss_calculated = True
            self.solver.solve_frequency(
                electrode_elements,
                calculate_mech_loss
            )
        elif isinstance(self.solver, PiezoSimTime):
            electrode_elements = None
            if "electrode_elements" in kwargs:
                electrode_elements = kwargs["electrode_elements"]
                self.charge_calculated = True

            self.solver.solve_time(
                electrode_elements
            )
        elif isinstance(self.solver, ThermoPiezoSimTime):
            electrode_elements = None
            if "electrode_elements" in kwargs:
                electrode_elements = kwargs["electrode_elements"]
                self.charge_calculated = True
            theta_start = None
            if "theta_start" in kwargs:
                theta_start = kwargs["theta_start"]

            self.solver.solve_time(
                electrode_elements,
                theta_start
            )
        else:
            return

    def save_simulation_results(self, file_prefix: str = ""):
        """Save the simulation results to the simulation folder. If a prefix
        is given this is prepend to the name of the output file.

        Parameters:
            file_prefix: String which is prepend to the output file names.
        """
        if not os.path.exists(self.simulation_directory):
            os.makedirs(self.simulation_directory)

        if file_prefix != "":
            file_prefix += "_"

        if isinstance(self.solver, ThermoSimTime):
            np.save(
                os.path.join(
                    self.simulation_directory,
                    f"{file_prefix}theta.npy"
                ),
                self.solver.theta
            )
        elif isinstance(self.solver, PiezoSimFreq):
            np.save(
                os.path.join(
                    self.simulation_directory,
                    f"{file_prefix}u.npy"
                ),
                self.solver.u
            )
            if self.mech_loss_calculated:
                np.save(
                    os.path.join(
                        self.simulation_directory,
                        f"{file_prefix}mech_loss.npy"
                    ),
                    self.solver.mech_loss
                )
            if self.charge_calculated:
                np.save(
                    os.path.join(
                        self.simulation_directory,
                        f"{file_prefix}q.npy"
                    ),
                    self.solver.q
                )
        elif isinstance(self.solver, PiezoSimTime):
            np.save(
                os.path.join(
                    self.simulation_directory,
                    f"{file_prefix}u.npy"
                ),
                self.solver.u
            )
            if self.charge_calculated:
                np.save(
                    os.path.join(
                        self.simulation_directory,
                        f"{file_prefix}q.npy"
                    ),
                    self.solver.q
                )
        elif isinstance(self.solver, ThermoPiezoSimTime):
            np.save(
                os.path.join(
                    self.simulation_directory,
                    f"{file_prefix}u.npy"
                ),
                self.solver.u
            )
            # Mech loss is always calculated in PiezoSimTherm
            np.save(
                os.path.join(
                    self.simulation_directory,
                    f"{file_prefix}mech_loss.npy"
                ),
                self.solver.mech_loss
            )
            if self.charge_calculated:
                np.save(
                    os.path.join(
                        self.simulation_directory,
                        f"{file_prefix}q.npy"
                    ),
                    self.solver.q
                )

    def save_simulation_settings(self, prefix=""):
        """Save the simulation settings to the simulation folder. If a prefix
        is given this is prepend to the name of the output file.

        Parameters:
            file_prefix: String which is prepend to the output file names.
        """
        if prefix != "":
            prefix += "_"

        if not os.path.exists(self.simulation_directory):
            os.makedirs(self.simulation_directory)

        settings = configparser.ConfigParser()

        if isinstance(self.solver, ThermoSimTime):
            simulation_type = "ThermoSimTime"
        elif isinstance(self.solver, PiezoSimTime):
            simulation_type = "PiezoTime"
        elif isinstance(self.solver, PiezoSimFreq):
            simulation_type = "PiezoFreq"
        elif isinstance(self.solver, ThermoPiezoSimTime):
            simulation_type = "ThermoPiezo"
        else:
            raise ValueError(
                f"Cannot save simulation type {type(self.solver)}"
            )

        # General simulation settings
        general_settings = {
            "name": self.simulation_name,
            "mesh_file": self.mesh.mesh_file_path,
            "simulation_type": simulation_type
        }
        settings["general"] = general_settings

        # Material settings and data
        material_settings = {}
        for material in self.material_manager.materials:
            material_settings[material.material_name] = material.to_dict()

        if isinstance(self.solver, PiezoSimFreq):
            # Save frequencies
            np.savetxt(
                os.path.join(
                    self.simulation_directory,
                    f"{prefix}frequencies.txt"
                ),
                self.solver.frequencies
            )
        else:
            # Simulation data
            settings["simulation"] = self.solver.simulation_data.__dict__

        # Boundary conditions
        boundary_conditions = {}
        for index, bc in enumerate(self.boundary_conditions):
            boundary_conditions[index] = bc

        # Save simulation data to config file
        with open(
            os.path.join(
                self.simulation_directory,
                f"{prefix}{self.simulation_name}.cfg"
            ),
            "w",
            encoding="UTF-8"
        ) as fd:
            settings.write(fd)

        # Save materials to json
        with open(
            os.path.join(
                self.simulation_directory,
                f"{prefix}materials.json"
            ),
            "w",
            encoding="UTF-8"
        ) as fd:
            json.dump(material_settings, fd, indent=2)

        # Save boundary conditions to json
        with open(
            os.path.join(
                self.simulation_directory,
                f"{prefix}boundary_conditions.json"
            ),
            "w",
            encoding="UTF-8"
        ) as fd:
            json.dump(boundary_conditions, fd, indent=2)

    @staticmethod
    def load_simulation_settings(simulation_folder: str):
        """Loads the simulation settings from the given folder and return
        a SingleSimulation object.

        Parameters:
            simulation_folder: Path to a folder containing simulation files.
        """
        # TODO Right now in a coupled sim the simulation settings
        # are overwriding itself
        # --> Add additional possible label to prevent this


        simulation_name = os.path.basename(simulation_folder)

        # Check if folder has all necessary files
        necessary_files = [
            os.path.join(simulation_folder, f"{simulation_name}.cfg"),
            os.path.join(simulation_folder, "materials.json"),
            os.path.join(simulation_folder, "boundary_conditions.json")
        ]
        for file_path in necessary_files:
            if not os.path.isfile(file_path):
                raise IOError(f"{file_path} odes not exist.")

        settings = configparser.ConfigParser()
        settings.read(necessary_files[0])

        # Read general simulation settings
        mesh_file = settings["general"]["mesh_file"]
        simulation_type = settings["general"]["simulation_type"]
        simulation = SingleSimulation(
            simulation_folder,
            "",
            Mesh(mesh_file, True)
        )
        # Workaround since the simulation_folder and simulation_name are
        # combined in the constructor, see empty string above for the
        # constructor
        simulation.simulation_name = simulation_name

        # Read simulation data
        if simulation_type == "PiezoFreq":
            # Check if file exists
            frequencies_file = os.path.join(
                simulation_folder,
                "frequencies.txt"
            )
            if os.path.isfile(frequencies_file):
                frequencies = np.loadtxt(frequencies_file)
                simulation.setup_piezo_freq_domain(frequencies)
            else:
                raise IOError(
                    "Frequencies file not found for frequency type sim"
                )
        else:
            # Load simulation data
            simulation_data = SimulationData(
                float(settings["simulation"]["delta_t"]),
                int(settings["simulation"]["number_of_time_steps"]),
                float(settings["simulation"]["gamma"]),
                float(settings["simulation"]["beta"]),
            )
            if simulation_type == "ThermoSimTime":
                simulation.setup_thermo_time_domain(
                    simulation_data.delta_t,
                    simulation_data.number_of_time_steps,
                    simulation_data.gamma
                )
            elif simulation_type == "PiezoTime":
                simulation.setup_piezo_time_domain(simulation_data)
            elif simulation_type == "ThermoPiezo":
                simulation.setup_thermal_piezo_time_domain(simulation_data)
            else:
                raise IOError(
                    f"Cannot deserialize {simulation_type} simulation type"
                )

        # Read materials
        with open(necessary_files[1], "r", encoding="UTF-8") as fd:
            material_settings = json.load(fd)
            for material_name in material_settings.keys():
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
        """Load the simulation results which are in the current simulation
        folder to the current SingleSimulation object."""
        if isinstance(self.solver, ThermoSimTime):
            theta_file = os.path.join(
                self.simulation_directory,
                "theta.npy"
            )
            if os.path.isfile(theta_file):
                self.solver.theta = np.load(theta_file)
            else:
                raise IOError("Couldn't find theta file.")
        elif isinstance(self.solver, PiezoSimTime):
            u_file = os.path.join(
                self.simulation_directory,
                "u.npy"
            )
            q_file = os.path.join(
                self.simulation_directory,
                "q.npy"
            )
            if os.path.isfile(u_file):
                self.solver.u = np.load(u_file)
            else:
                raise IOError("Couldn't find u file.")
            if os.path.isfile(q_file):
                self.solver.q = np.load(q_file)
        elif isinstance(self.solver, PiezoSimFreq):
            u_file = os.path.join(
                self.simulation_directory,
                "u.npy"
            )
            q_file = os.path.join(
                self.simulation_directory,
                "q.npy"
            )
            mech_loss_file = os.path.join(
                self.simulation_directory,
                "mech_loss.npy"
            )
            print(u_file)
            if os.path.isfile(u_file):
                self.solver.u = np.load(u_file)
            else:
                raise IOError("Couldn't find u file.")
            if os.path.isfile(q_file):
                self.solver.q = np.load(q_file)
            if os.path.isfile(mech_loss_file):
                self.solver.mech_loss = np.load(mech_loss_file)
        elif isinstance(self.solver, ThermoPiezoSimTime):
            u_file = os.path.join(
                self.simulation_directory,
                "u.npy"
            )
            q_file = os.path.join(
                self.simulation_directory,
                "q.npy"
            )
            mech_loss_file = os.path.join(
                self.simulation_directory,
                "mech_loss.npy"
            )
            if os.path.isfile(u_file):
                self.solver.u = np.load(u_file)
            else:
                raise IOError("Couldn't find u file.")
            if os.path.isfile(q_file):
                self.solver.q = np.load(q_file)
            if os.path.isfile(mech_loss_file):
                self.solver.mech_loss = np.load(mech_loss_file)
        else:
            raise ValueError(
                "Cannot load simulation settings of simulation type",
                type(self.solver)
            )
