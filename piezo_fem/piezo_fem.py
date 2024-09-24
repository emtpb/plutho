"""Module for setting up the simulations."""

# Python standard libraries
from typing import Union
import os
import numpy as np
import numpy.typing as npt

# Local libraries
from .simulation.base import MeshData, MaterialData, SimulationData, \
    SimulationType, ModelType, get_dirichlet_boundary_conditions
from .simulation.fem_piezo_temp_time import PiezoSimTherm
from .simulation.fem_piezo_time import PiezoSim
from .gmsh_handler import GmshHandler


class SimulationException(Exception):
    """Custom exception to simplify errors."""


class Simulation:
    """Wrapper class to make the handlign of the different simulation
    classes easier."""

    # Basic parameters
    workspace_directory: str
    gmsh_handler: GmshHandler
    simulation_type: SimulationType
    model_type: ModelType

    # Simulation parameters
    solver: Union[PiezoSim, PiezoSimTherm]
    mesh_data: MeshData
    material_data: MaterialData
    simulation_data: SimulationData
    excitation: npt.NDArray

    def __init__(self, workspace_directory, material_data):
        self.workspace_directory = workspace_directory
        self.material_data = material_data

        if not os.path.exists(self.workspace_directory):
            os.makedirs(self.workspace_directory)

    def create_disc_mesh(self, radius, height, mesh_size):
        if not self.gmsh_handler:
            self.gmsh_handler = GmshHandler(
                os.path.join(self.workspace_directory, "disc.msh")
            )

        self.gmsh_handler.generate_rectangular_mesh(
            radius, height, mesh_size, 0
        )

        nodes, elements = self.gmsh_handler.get_mesh_nodes_and_elements()
        self.mesh_data = MeshData(nodes, elements)

    def create_ring_mesh(self, inner_radius, outer_radius, height, mesh_size):
        if not self.gmsh_handler:
            self.gmsh_handler = GmshHandler(
                os.path.join(self.workspace_directory, "ring.msh")
            )

        self.gmsh_handler.generate_rectangular_mesh(
            outer_radius, height, mesh_size, inner_radius
        )

        nodes, elements = self.gmsh_handler.get_mesh_nodes_and_elements()
        self.mesh_data = MeshData(nodes, elements)

    def set_triangle_pulse_excitation(self, amplitude, number_of_time_steps):
        excitation = np.zeros(number_of_time_steps)
        excitation[1:10] = (
            amplitude
            * np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
        )

        self.excitation = excitation

    def set_sinusoidal_excitation(self, frequency, number_of_time_steps, delta_t):
        time_values = np.arange(number_of_time_steps)*delta_t
        self.excitation = np.sin(2*np.pi*time_values*frequency)

    def set_simulation(
            self,
            delta_t,
            number_of_time_steps,
            gamma,
            beta, 
            simulation_type,
            model_type):
        # Check if everything is set up.
        if not self.mesh_data:
            raise SimulationException("Please set mesh data first.")
        if not self.material_data:
            raise SimulationException("Please set material data first.")
        if not self.excitation:
            raise SimulationException("Please set an excitation first.")

        self.model_type = model_type
        self.simulation_data = SimulationData(
            delta_t,
            number_of_time_steps,
            gamma,
            beta
        )

        if simulation_type is SimulationType.PIEZOELECTRIC:
            self.solver = PiezoSim(
                self.mesh_data,
                self.material_data,
                self.simulation_data
            )
        elif simulation_type is SimulationType.THERMOPIEZOELECTRIC:
            self.solver = PiezoSimTherm(
                self.mesh_data,
                self.material_data,
                self.simulation_data
            )
        else:
            raise SimulationException(
                f"Simulation type {simulation_type} is not implemented.")

        self.simulation_type = simulation_type
        dirichlet_nodes, dirichlet_values = get_dirichlet_boundary_conditions(
            self.gmsh_handler,
            self.excitation,
            self.simulation_data.number_of_time_steps,
            self.model_type is ModelType.DISC
        )
        self.solver.dirichlet_nodes = dirichlet_nodes
        self.solver.dirichlet_values = dirichlet_values

    def solve(self):
        if not self.solver:
            raise SimulationException("Please setup the simulation first.")
        # Get excited triangles
        pg_elements = self.gmsh_handler.get_elements_by_physical_groups(
            ["Electrode"])
        electrode_triangles = pg_elements["Electrode"]

        self.solver.assemble()
        self.solver.solve_time(electrode_triangles)
        # TODO Values as attributes? Return values or make wrappers to solver?

    def save_simulation_settings(self):
        # TODO Save simulation settings to a config file

    # TODO Make a function to read settings from a config file

    def save_simulation_results(self):
        u_file_path = os.path.join(self.workspace_directory, "u")
        q_file_path = os.path.join(self.workspace_directory, "q")

        np.save(u_file_path, self.solver.u)
        np.save(q_file_path, self.solver.q)

        if self.simulation_type is SimulationType.THERMOPIEZOELECTRIC:
            mech_loss_file_path = os.path.join(
                self.workspace_directory, "mech_loss")
            np.save(mech_loss_file_path, self.solver.mech_loss)
            # TODO Add energy stored?

    def create_post_processing_views(self):
        if self.simulation_data is SimulationType.PIEZOELECTRIC:
            self.gmsh_handler.create_u_default_post_processing_view(
                self.solver.u,
                self.simulation_data.number_of_time_steps,
                self.simulation_data.delta_t,
                False
            )
        elif self.simulation_type is SimulationType.THERMOPIEZOELECTRIC:
            self.gmsh_handler.create_u_default_post_processing_view(
                self.solver.u,
                self.simulation_data.number_of_time_steps,
                self.simulation_data.delta_t,
                True
            )
            self.gmsh_handler.create_element_post_processing_view(
                self.solver.mech_loss,
                self.simulation_data.number_of_time_steps,
                self.simulation_data.delta_t,
                1,
                "Mechanical loss"
            )

        else:
            raise SimulationException(
                f"Simulation type {self.simulation_type} is not implemented.")
