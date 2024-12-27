
# Python standard libraries
import os
from enum import Enum
from typing import Union, List

# Third party libraries
import numpy as np
import numpy.typing as npt

# Local libraries
from .simulation.base import MeshData, SimulationData
from .simulation import HeatConductionSim, PiezoSim, PiezoSimTherm, \
    PiezoFreqSim
from .materials import MaterialData, MaterialManager
from .mesh import Mesh


class VariableType(Enum):
    U_R = 0
    U_Z = 1
    PHI = 2
    THETA = 3


class SingleSimulation:

    # Basic settings
    simulation_name: str
    simulation_directory: str

    # Materials
    material_manager: MaterialManager

    # Mesh
    mesh: Mesh
    mesh_data: MeshData

    # Dirichlet bc
    dirichlet_nodes: List[int]
    dirichlet_values: List[npt.NDArray]

    # Simulation
    solver: Union[HeatConductionSim, PiezoFreqSim, PiezoSim, PiezoSimTherm]

    def __init__(
            self,
            working_directory: str,
            simulation_name: str,
            mesh: Mesh):
        # Check if working directory exists and if not create it
        simulation_directory = os.path.join(working_directory, simulation_name)
        if not os.path.exists(simulation_directory):
            os.makedirs(simulation_directory)

        self.simulation_directory = simulation_directory
        self.simulation_name = simulation_name
        self.mesh = mesh

        nodes, elements = mesh.get_mesh_nodes_and_elements()
        self.mesh_data = MeshData(nodes, elements)
        self.material_manager = MaterialManager(len(elements))

        # Initialize dirichlet bc arrays
        self.dirichlet_nodes = []
        self.dirichlet_values = []

    def add_material(
            self,
            material_name: str,
            material_data: MaterialData,
            physical_group_name: str):
        if physical_group_name is None:
            element_indices = np.arange(len(self.mesh_data.elements))
        else:
            element_indices = self.mesh.get_elements_by_physical_groups(
                [physical_group_name]
            )[physical_group_name]

        self.material_manager.add_material(
            material_name,
            material_data,
            element_indices
        )

    def add_dirichlet_bc(
            self,
            variable_type: VariableType,
            physical_group: str,
            values: npt.NDArray):
        if isinstance(self.solver, HeatConductionSim):
            if (variable_type is VariableType.U_R or
                    variable_type is VariableType.U_Z or
                    variable_type is VariableType.PHI):
                raise ValueError(
                    f"Unknown variable type {variable_type} given for"
                    f"simulation type {type(self.solver)}")
        elif (isinstance(self.solver, PiezoFreqSim) or
                isinstance(self.solver, PiezoSim)):
            if variable_type is VariableType.THETA:
                raise ValueError(
                    f"Unknown variable type {variable_type} given for"
                    f"simulation type {type(self.solver)}")

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
            if variable_type is VariableType.PHI:
                real_index = 2*number_of_nodes+node_index
            elif variable_type is VariableType.THETA:
                if isinstance(self.solver, HeatConductionSim):
                    real_index = node_index
                else:
                    real_index = 3*number_of_nodes+node_index
            elif variable_type is VariableType.U_R:
                real_index = 2*node_index
            elif variable_type is VariableType.U_Z:
                real_index = 2*node_index+1
            else:
                raise ValueError(f"Unknown variable type {variable_type}")
            self.dirichlet_nodes.append(real_index)
            self.dirichlet_values.append(values)

    def setup_heat_conduction_time_domain(
            self,
            delta_t: float,
            number_of_time_steps: int,
            gamma: float):
        self.solver = HeatConductionSim(
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
            simulation_data: SimulationData):
        self.solver = PiezoSim(
            self.mesh_data,
            self.material_manager,
            simulation_data
        )

    def setup_piezo_freq_domain(self):
        self.solver = PiezoFreqSim(
            self.mesh_data,
            self.material_manager
        )

    def setup_thermal_piezo_time_domain(
            self,
            simulation_data: SimulationData):
        self.solver = PiezoSimTherm(
            self.mesh_data,
            self.material_manager,
            simulation_data
        )

    def simulate(self, **kwargs):
        self.material_manager.initialize_materials()

        self.solver.dirichlet_nodes = np.array(self.dirichlet_nodes)
        self.solver.dirichlet_values = np.array(self.dirichlet_values)
        self.solver.assemble()
        if isinstance(self.solver, HeatConductionSim):
            theta_start = None
            if "theta_start" in kwargs:
                theta_start = kwargs["theta_start"]
            self.solver.solve_time(
                theta_start
            )
        elif isinstance(self.solver, PiezoFreqSim):
            if "frequencies" not in kwargs:
                raise ValueError("'frequencies' must be given as argument.")
            electrode_elements = None
            if "electrode_elements" in kwargs:
                electrode_elements = kwargs["electrode_elements"]

            self.solver.solve_frequency(
                kwargs["frequencies"],
                electrode_elements
            )
        elif isinstance(self.solver, PiezoSim):
            electrode_elements = None
            if "electrode_elements" in kwargs:
                electrode_elements = kwargs["electrode_elements"]

            self.solver.solve_time(
                electrode_elements
            )
        elif isinstance(self.solver, PiezoSimTherm):
            electrode_elements = None
            if "electrode_elements" in kwargs:
                electrode_elements = kwargs["electrode_elements"]
            theta_start = None
            if "theta_start" in kwargs:
                theta_start = kwargs["theta_start"]

            self.solver.solve_time(
                electrode_elements,
                theta_start
            )
        else:
            return
