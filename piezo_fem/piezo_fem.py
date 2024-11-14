"""Module for setting up the simulations."""

# Python standard libraries
from __future__ import annotations
from typing import Union, Any
import configparser
import os
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

# Local libraries
from .simulation.base import MeshData, MaterialData, SimulationData, \
    SimulationType, ModelType, create_dirichlet_bc_nodes, \
    ExcitationInfo, ExcitationType
from .simulation.fem_piezo_temp_time import PiezoSimTherm
from .simulation.fem_piezo_time import PiezoSim
from .gmsh_handler import GmshHandler
from .materials import pic255, MaterialManager
from .postprocessing import calculate_impedance

class SimulationException(Exception):
    """Custom exception to simplify errors."""


class PiezoSimulation:
    """Wrapper class to make the handling of the different simulation
    classes easier. In order to use this it is necessary to call certain
    functions to setup the simulation. Those include:
    1. Setting up a mesh -> Disc or ring
    2. Setting simulation settings
    3. Setting excitation -> Triangle pulse or sinusoidal
    4. Settings boundary conditions
    5. Solving
    (6. Saving results)
    Examples can be found in the exmaples folder.

    Parameters:
        workspace_directory: Directory where all the files are stored.
        masterial_data: Contains information about the used material.
        simulation_name: Name of the simulation.

    Attributes:
        workspace_directory: Directory where all the files are stored.
        gmsh_handler: Handler to manage the gmsh meshes and files.
        simulation_type: Simulation type.
        simulation_name: Name of the simulation. Used for names of the
            different files.
        solver: Either PiezoSim or PiezoSimTherm depending on the
            simulation_type.
        mesh_data: Contains information about the used mesh.
        material_handler: Contains information about the used material.
        simulation_data: Contains information about the simulation itself.
        excitation: Excitation function.
        excitation_info: Information about the edxcitation used to store in the
            config file.
    """

    # Basic parameters
    workspace_directory: str
    gmsh_handler: GmshHandler
    simulation_type: SimulationType
    simulation_name: str

    # Simulation parameters
    solver: Union[PiezoSim, PiezoSimTherm]
    mesh_data: Union[Any, MeshData]
    material_data: MaterialData
    simulation_data: SimulationData
    excitation: npt.NDArray
    excitation_info: ExcitationInfo
    material_manager: MaterialManager

    def __init__(self,
                 workspace_directory: str,
                 simulation_name: str):
        self.workspace_directory = workspace_directory
        self.simulation_name = simulation_name

        if not os.path.exists(self.workspace_directory):
            os.makedirs(self.workspace_directory)

    def set_material_data(
            self,
            material_name: str,
            material_data: MaterialData,
            starting_temperature: Union[float, npt.NDArray] = 0):
        """Sets the material data for the model.

        Parameters:
            material_name: Name of the material, used in .cfg file.
            material_data: MaterialData object.
            starting_temperature: Only needed if the material data is
                temperature dependent. If its a float the value will be set
                constant for the whole model. If its a np.ndarray it must have
                the size of number of elements.
        """
        if not hasattr(self, "mesh_data"):
            raise SimulationException("Please set mesh data first.")

        self.material_manager = MaterialManager(
            material_name,
            material_data,
            len(self.mesh_data.elements),
            starting_temperature
        )

    def create_disc_mesh(self,
                         radius: float,
                         height: float,
                         mesh_size: float):
        """Creates a disc mesh for the simulation.

        Parameters:
            radius: Radius of the disc.
            height: Height of the disc.
            mesh_size: Maximum distance between mesh elements.
        """
        if not hasattr(self, "gmsh_handler"):
            self.gmsh_handler = GmshHandler(
                os.path.join(
                    self.workspace_directory,
                    f"{self.simulation_name}_disc.msh")
            )

        self.gmsh_handler.generate_rectangular_mesh(
            radius, height, mesh_size, 0
        )

        self.gmsh_handler.model_type = ModelType.DISC

        nodes, elements = self.gmsh_handler.get_mesh_nodes_and_elements()
        self.mesh_data = MeshData(nodes, elements)

    def create_ring_mesh(self,
                         inner_radius: float,
                         outer_radius: float,
                         height: float,
                         mesh_size: float):
        """Creates a ring mesh for the simulation.

        Parameters:
            inner_radius: Inner radius of the ring.
            outer_radius: Outer radius of the ring.
            height: Height of the ring.
            mesh_size: Maximum distance between mesh elements.
        """
        if not hasattr(self, "gmsh_handler"):
            self.gmsh_handler = GmshHandler(
                os.path.join(
                    self.workspace_directory,
                    f"{self.simulation_name}_ring.msh"
                )
            )

        self.gmsh_handler.generate_rectangular_mesh(
            outer_radius, height, mesh_size, inner_radius
        )

        self.gmsh_handler.model_type = ModelType.RING

        nodes, elements = self.gmsh_handler.get_mesh_nodes_and_elements()
        self.mesh_data = MeshData(nodes, elements)

    def set_triangle_pulse_excitation(self,
                                      amplitude: float):
        """Settings a single triangle pulse as excitation. The length of the
        triangle pulse is dependent on the delta_t. It takes up 9 time steps.

        Parameters:
            amplitude: Sets the amplitude of the triangle pulse."""
        if not hasattr(self, "simulation_data"):
            raise SimulationException("Please set simulation data first.")

        self.excitation_info = ExcitationInfo(
            amplitude,
            0.0,
            ExcitationType.TRIANGULAR_PULSE
        )
        excitation = np.zeros(self.simulation_data.number_of_time_steps)
        excitation[1:10] = (
            amplitude
            * np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
        )

        self.excitation = excitation

    def set_sinusoidal_excitation(self,
                                  amplitude: float,
                                  frequency: float):
        """Sets a sin function as excitation.

        Parameters:
            amplitude: Amplitude of the sin function.
            frequency: Frequency of the sin function in Hz."""
        if not hasattr(self, "simulation_data"):
            raise SimulationException("Please set simulation data first.")

        self.excitation_info = ExcitationInfo(
            amplitude,
            frequency,
            ExcitationType.SINUSOIDAL
        )
        time_values = (
            np.arange(self.simulation_data.number_of_time_steps)
            * self.simulation_data.delta_t
        )
        self.excitation = amplitude*np.sin(2*np.pi*time_values*frequency)

    def set_simulation(
            self,
            delta_t: float,
            number_of_time_steps: int,
            gamma: float,
            beta: float,
            simulation_type: SimulationType):
        """Sets the simulation settings.

        Parameters:
            delta_t: Difference between the time steps.
            number_of_time_steps: Total time step count.
            gamma: Time integration parameter.
            beta: Time integration parameter.
            simulation_type: Type of the simulation.
        """
        # Check if everything is set up.
        if not hasattr(self, "mesh_data"):
            raise SimulationException("Please set mesh data first.")
        if not hasattr(self, "material_manager"):
            raise SimulationException("Please set material data first.")

        self.simulation_data = SimulationData(
            delta_t,
            number_of_time_steps,
            gamma,
            beta
        )

        if simulation_type is SimulationType.PIEZOELECTRIC:
            self.solver = PiezoSim(
                self.mesh_data,
                self.material_manager,
                self.simulation_data
            )
        elif simulation_type is SimulationType.THERMOPIEZOELECTRIC:
            self.solver = PiezoSimTherm(
                self.mesh_data,
                self.material_manager,
                self.simulation_data
            )
        else:
            raise SimulationException(
                f"Simulation type {simulation_type} is not implemented.")
        self.simulation_type = simulation_type

    def set_boundary_conditions(self):
        """Sets boundary conditions for the simulation."""
        if not hasattr(self, "gmsh_handler"):
            raise SimulationException("Please setup mesh first.")
        if not hasattr(self, "excitation_info"):
            raise SimulationException("Please setup excitation first.")
        if not hasattr(self, "simulation_data"):
            raise SimulationException("Please setup simulation data first.")
        dirichlet_nodes, dirichlet_values = create_dirichlet_bc_nodes(
            self.gmsh_handler,
            self.excitation,
            self.simulation_data.number_of_time_steps,
            self.gmsh_handler.model_type is ModelType.DISC
        )
        self.solver.dirichlet_nodes = dirichlet_nodes
        self.solver.dirichlet_values = dirichlet_values

    def simulate(self):
        """Executes the simulation. Results can be accessed using
        self.solver."""
        if not hasattr(self, "gmsh_handler"):
            raise SimulationException("Please setup mesh first.")
        if not hasattr(self, "solver"):
            raise SimulationException("Please setup the simulation first.")
        # Get excited triangles
        pg_elements = self.gmsh_handler.get_elements_by_physical_groups(
            ["Electrode"])
        electrode_triangles = pg_elements["Electrode"]

        self.solver.assemble()
        self.solver.solve_time(
            electrode_triangles,
            self.gmsh_handler.model_type is ModelType.DISC
        )

    def save_simulation_settings(self,
                                 description: str = ""):
        """Saves the current simulation setup in a config file in the
        working directory.

        Parameters:
            description: Description which can be added to the config file.
        """
        settings = configparser.ConfigParser()
        general_settings = {
                "name": self.simulation_name,
                "material_name": self.material_manager.name
        }
        if description != "":
            general_settings["description"] = description
        general_settings["simulation_type"] = self.simulation_type.value
        settings["general"] = general_settings
        settings["simulation"] = self.simulation_data.__dict__
        settings["excitation"] = self.excitation_info.asdict()
        settings["model"] = self.gmsh_handler.as_dict()

        with open(
                os.path.join(
                    self.workspace_directory,
                    f"{self.simulation_name}.cfg"),
                "w",
                encoding="UTF-8") as fd:
            settings.write(fd)

    def load_simulation_results(self):
        """Loads already existing simulation results from the folder into
        the solver object.
        """
        if not os.path.exists(self.workspace_directory):
            raise IOError("No simulation results found.")

        displacement_file = os.path.join(
            self.workspace_directory,
            f"{self.simulation_name}_u.npy")
        charge_file = os.path.join(
            self.workspace_directory,
            f"{self.simulation_name}_q.npy")

        self.solver.u = np.load(displacement_file)
        self.solver.q = np.load(charge_file)

        if isinstance(self.solver, PiezoSimTherm):
            mech_loss_file = os.path.join(
                self.workspace_directory,
                f"{self.simulation_name}_mech_loss.npy"
            )
            self.solver.mech_loss = np.load(mech_loss_file)

    @staticmethod
    def load_simulation_settings(
            config_file_path: str,
            new_working_diretory: str = "") -> PiezoSimulation:
        """Creates a simulation object from the given config.

        Parameters:
            config_file_path: Pathto the config file.
            new_working_directory: Overrides the working directory of the
                given file with a new one. The simulation files are then saved
                in the new working_direcotry. If set to "" the working
                directory of the file is used.

        Returns:
            Simulation object created from the config file."""
        if not os.path.exists(config_file_path):
            raise IOError(f"Couldn_t find config file {config_file_path}.")

        config = configparser.ConfigParser()
        config.read(config_file_path)

        simulation_name = config["general"]["name"]
        simulation_type = SimulationType(
            config["general"]["simulation_type"])
        material_name = config["general"]["material_name"]
        if material_name == "pic255":
            material_data = pic255
        elif material_name == "pic181_temp":
            print("For simplicity wrong material used.")
            material_data = pic255
        else:
            raise SimulationException(
                f"Unknown material given {material_name}")

        simulation_data = SimulationData(
            delta_t=float(config["simulation"]["delta_t"]),
            number_of_time_steps=int(
                config["simulation"]["number_of_time_steps"]),
            gamma=float(config["simulation"]["gamma"]),
            beta=float(config["simulation"]["beta"])
        )
        excitation_type = ExcitationType(
            config["excitation"]["excitation_type"]
        )
        if excitation_type is ExcitationType.SINUSOIDAL:
            excitation_info = ExcitationInfo(
                amplitude=float(config["excitation"]["amplitude"]),
                frequency=float(config["excitation"]["frequency"]),
                excitation_type=excitation_type
            )

        elif excitation_type is ExcitationType.TRIANGULAR_PULSE:
            excitation_info = ExcitationInfo(
                amplitude=float(config["excitation"]["amplitude"]),
                excitation_type=excitation_type,
                frequency=0
            )
        else:
            raise SimulationException(
                f"Unknown excitation type given {excitation_type}")

        workspace_directory = new_working_diretory \
            if new_working_diretory != "" \
                else os.path.dirname(os.path.abspath(config_file_path))

        sim = PiezoSimulation(workspace_directory, simulation_name)

        model_type = ModelType(config["model"]["model_type"])
        if model_type is ModelType.DISC:
            mesh_file = os.path.join(
                workspace_directory,
                f"{simulation_name}_disc.msh")
            sim.gmsh_handler = GmshHandler(mesh_file, True)
            sim.gmsh_handler.model_type = model_type
            sim.gmsh_handler.width = float(config["model"]["width"])
            sim.gmsh_handler.height = float(config["model"]["height"])
            sim.gmsh_handler.mesh_size = float(config["model"]["mesh_size"])
        elif model_type is ModelType.RING:
            mesh_file = os.path.join(
                workspace_directory,
                f"{simulation_name}_ring.msh")
            sim.gmsh_handler = GmshHandler(mesh_file, True)
            sim.gmsh_handler.model_type = model_type
            sim.gmsh_handler.width = float(config["model"]["width"])
            sim.gmsh_handler.height = float(config["model"]["height"])
            sim.gmsh_handler.mesh_size = float(config["model"]["mesh_size"])
            sim.gmsh_handler.x_offset = float(config["model"]["x_offset"])
        else:
            raise IOError(f"Cannot deserialize ModelType {model_type}")
        nodes, elements = sim.gmsh_handler.get_mesh_nodes_and_elements()
        sim.mesh_data = MeshData(nodes, elements)

        sim.set_material_data(
            material_name,
            material_data
        )

        sim.set_simulation(
            delta_t=simulation_data.delta_t,
            number_of_time_steps=simulation_data.number_of_time_steps,
            gamma=simulation_data.gamma,
            beta=simulation_data.beta,
            simulation_type=simulation_type
        )

        if excitation_info.excitation_type is ExcitationType.SINUSOIDAL:
            sim.set_sinusoidal_excitation(
                amplitude=excitation_info.amplitude,
                frequency=excitation_info.frequency
            )
        elif excitation_info.excitation_type is \
                ExcitationType.TRIANGULAR_PULSE:
            sim.set_triangle_pulse_excitation(excitation_info.amplitude)
        else:
            raise IOError(
                "Cannot deserialize ExcitationType "
                f"{excitation_info.excitation_type}")

        sim.set_boundary_conditions()
        return sim

    def save_simulation_results(self):
        """Saves the simulation results u, q and the mechanical
        loss if the simulation of thermo-piezoelectric.
        The files are saved in the simulation working directory.
        """
        u_file_path = os.path.join(
            self.workspace_directory, f"{self.simulation_name}_u")
        q_file_path = os.path.join(
            self.workspace_directory, f"{self.simulation_name}_q")

        np.save(u_file_path, self.solver.u)
        np.save(q_file_path, self.solver.q)

        if self.simulation_type is SimulationType.THERMOPIEZOELECTRIC:
            mech_loss_file_path = os.path.join(
                self.workspace_directory,
                f"{self.simulation_name}_mech_loss")
            np.save(mech_loss_file_path, self.solver.mech_loss)

    def create_post_processing_views(self):
        """Writes the calculated fields to a gmsh *.msh file which can be
        opened in gmsh. The files are written in the working directory."""
        if self.simulation_type is SimulationType.PIEZOELECTRIC:
            self.gmsh_handler.create_u_default_post_processing_view(
                self.solver.u,
                self.simulation_data.number_of_time_steps,
                self.simulation_data.delta_t,
                False,
                False
            )
        elif self.simulation_type is SimulationType.THERMOPIEZOELECTRIC:
            self.gmsh_handler.create_u_default_post_processing_view(
                self.solver.u,
                self.simulation_data.number_of_time_steps,
                self.simulation_data.delta_t,
                True,
                False
            )
            self.gmsh_handler.create_element_post_processing_view(
                self.solver.mech_loss,
                self.simulation_data.number_of_time_steps,
                self.simulation_data.delta_t,
                1,
                "Mechanical loss",
                True
            )

        else:
            raise SimulationException(
                f"Simulation type {self.simulation_type} is not implemented.")

    def plot_impedence(self):
        """Plots the impedence of the simulation results."""
        if self.excitation is None:
            raise SimulationException(
                "In order to plot the impedance an excitation needs to be set")
        frequencies_fem, impedence_fem = calculate_impedance(
            self.solver.q, self.excitation, self.simulation_data.delta_t)

        plt.plot(frequencies_fem, np.abs(impedence_fem), label="MyFEM")
        plt.xlabel("Frequency f / Hz")
        plt.ylabel("Impedence |Z| / $\\Omega$")
        plt.yscale("log")
        plt.legend()
        plt.grid()
        plt.show()
