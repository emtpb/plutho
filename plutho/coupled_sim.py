"""Module for coupled simulations."""

# Python stanard libraries
from dataclasses import dataclass

# Third party libraries
import numpy as np
import numpy.typing as npt

# Local libraries
from .simulations import ThermoPiezoTime, ThermoTime, PiezoFreq
from .simulations.helpers import get_avg_temp_field_per_element
from .mesh.mesh import Mesh
from .materials import MaterialData
from .enums import FieldType


__all__ = [
    "SimulationData",
    "CoupledThermoPiezoTime",
    "CoupledPiezoThermoFreq"
]


@dataclass
class SimulationData:
    delta_t: float
    number_of_time_steps: int
    gamma: float
    beta: float


class CoupledThermoPiezoTime:
    """Couples a thermo-piezoelectric simulation with a heat conduction
    simulation where only a temperature field is calculated. The power
    losses from the thermo-piezoeletric simulation are assumed to be
    stationary at the last time steps, therefore they can be used as constant
    source in a heat conduction simulation. FOr this simulation a piezo
    with a 'Electrode' and 'Ground' physical group is assumed.
    Also a 'Symaxis' for the symmetric axis property is needed.

    Attributes:
        simulation_name: Name of the simulation and the simulation folder.
        piezo_sim: Thermo-piezoeletric simulation.
        heat_cond_sim: Heat conduction simulation.
    """

    # Basic settings
    simulation_name: str

    # Simulations
    thermo_piezo_sim: ThermoPiezoTime
    thermo_sim: ThermoTime

    def __init__(
        self,
        simulation_name: str,
        mesh: Mesh
    ):
        self.simulation_name = simulation_name

        # Init simulations
        thermo_piezo_sim = ThermoPiezoTime(
            simulation_name,
            mesh
        )

        thermo_sim = ThermoTime(
            simulation_name,
            mesh
        )

        self.thermo_piezo_sim = thermo_piezo_sim
        self.thermo_sim = thermo_sim

    def add_material(
            self,
            material_name: str,
            material_data: MaterialData,
            physical_group_name: str
    ):
        """Add a material to the simulation.

        Parameters:
            material_name: Name of the material.
            material_data: MaterialData object containing the material
                parameters
            physical_group_name: Name of the gmsh physical group for which
                this material shall be set.
        """
        self.thermo_piezo_sim.add_material(
            material_name,
            material_data,
            physical_group_name
        )
        self.thermo_sim.add_material(
            material_name,
            material_data,
            physical_group_name
        )

    def set_excitation(
        self,
        excitation: npt.NDArray,
        is_disc: bool
    ):
        """Sets the excitation to the given function (per time values).
        The given excitation function is set for all time steps at the
        'Electrode' physical group and 0 is set at the 'Ground' physical group.

        Parameters:
            excitation: Numpy array of time values for the excitation which
                will be set for every node on the 'Electrode' boundary.
            is_disc: Boolean to check if the model is a piezoelectric disc and
                therefore defined at the origin (r=0). In this case a
                additional boundary condition is set: u_r(r=0,z)=0.
        """
        self.thermo_piezo_sim.add_dirichlet_bc(
            FieldType.PHI,
            "Electrode",
            excitation
        )
        self.thermo_piezo_sim.add_dirichlet_bc(
            FieldType.PHI,
            "Ground",
            np.zeros(len(excitation))
        )
        if is_disc:
            self.thermo_piezo_sim.add_dirichlet_bc(
                FieldType.U_R,
                "Symaxis",
                np.zeros(len(excitation))
            )

    def set_convection_bc(
        self,
        alpha: float,
        outer_temperature: float
    ):
        """Sets a convction boundary condition for the simulation.

        Parameters:
            alpha: Heat transfer coefficient.
            outer_temperautre: Fixed temperature outside the model.
        """
        boundary_elements = \
            self.thermo_sim.mesh.get_elements_by_physical_groups(
                ["Electrode", "Ground", "RightBoundary"]
            )
        convective_bc_elements = np.vstack(
            [
                boundary_elements["Electrode"],
                boundary_elements["Ground"],
                boundary_elements["RightBoundary"]
            ]
        )

        self.thermo_sim.set_convection_bc(
            convective_bc_elements,
            alpha,
            outer_temperature
        )

    def assemble(self):
        """Assembles both underlying simulations."""
        self.thermo_piezo_sim.assemble()
        self.thermo_sim.assemble()

    def simulate(
        self,
        thermo_piezo_simulation_data: SimulationData,
        thermo_simulation_data: SimulationData,
        averaging_time_step_count: int = 100
    ):
        """Runs the coupled simulation. First the thermo-piezoelectric
        simulation is run. Then the power losses in the stationary state
        are calculated. Therefore an averaging user averinging_time_step_count
        number of time steps is done. After that the single thermo simulation
        is done.

        Parameters:
            thermo_piezo_simulation_data: Simulation data object, which
                contains the settings for the thermo-piezo simulation.
            thermo_simulation_data: Simulation data object, which contains
                the settings for the thermo simulation.
            averaging_time_step_count: Number of time steps starting from the
                last time step to calculate the average power loss.
        """

        # Run thermo_piezoelectric simulation first
        self.thermo_piezo_sim.simulate(
            thermo_piezo_simulation_data.delta_t,
            thermo_piezo_simulation_data.number_of_time_steps,
            thermo_piezo_simulation_data.gamma,
            thermo_piezo_simulation_data.beta,
        )

        # Calculate mech losses which are sources in heat cond sim
        avg_mech_loss = np.mean(
            self.thermo_piezo_sim.mech_loss[
                -averaging_time_step_count:,
                :
            ],
            axis=0
        )
        self.thermo_sim.set_constant_volume_heat_source(
            avg_mech_loss,
            thermo_simulation_data.number_of_time_steps
        )

        # Set the result temp field of piezo sim to start field of heat cond
        # sim
        number_of_nodes = len(self.thermo_piezo_sim.mesh_data.nodes)
        theta_start = self.thermo_piezo_sim.u[-1, 3*number_of_nodes:]

        self.thermo_sim.simulate(
            thermo_simulation_data.delta_t,
            thermo_simulation_data.number_of_time_steps,
            thermo_simulation_data.gamma,
            theta_start
        )

    def save_simulation_settings(self, directory: str):
        """Creates a folder 'simulation_name' in the given directory and saves
        the current simulation settings in it.

        Parameters:
            directory: Path to the directory in which the simulation folder is
                created.
        """
        self.thermo_piezo_sim.save_simulation_settings(
            directory,
            "thermo_piezo"
        )
        self.thermo_sim.save_simulation_settings(directory, "heat_cond")

    def save_simulation_results(self, directory: str):
        """Creates a folder 'simulation_name' in the given directory and saves
        the current simulation results in it.

        Parameters:
            directory: Path to the directory in which the simulation folder is
                created.
        """
        self.thermo_piezo_sim.save_simulation_results(
            directory,
            "thermo_piezo"
        )
        self.thermo_sim.save_simulation_results(
            directory,
            "heat_cond"
        )


class CoupledPiezoThermoFreq:

    simulation_name: str

    # Simulations
    piezo_freq: PiezoFreq
    thermo_time_sim: ThermoTime

    def __init__(
        self,
        simulation_name: str,
        mesh: Mesh
    ):
        self.simulation_name = simulation_name

        piezo_freq = PiezoFreq(
            simulation_name,
            mesh
        )

        thermo_time_sim = ThermoTime(
            simulation_name,
            mesh
        )

        self.piezo_freq = piezo_freq
        self.thermo_time_sim = thermo_time_sim

    def add_material(
        self,
        material_name: str,
        material_data: MaterialData,
        physical_group_name: str
    ):
        """Add a material to the simulation.

        Parameters:
            material_name: Name of the material.
            material_data: MaterialData object containing the material
                parameters
            physical_group_name: Name of the gmsh physical group for which
                this material shall be set.
        """
        self.piezo_freq.add_material(
            material_name,
            material_data,
            physical_group_name
        )
        self.thermo_time_sim.add_material(
            material_name,
            material_data,
            physical_group_name
        )

    def set_excitation(
        self,
        excitation: float,
        is_disc: bool = True
    ):
        """Sets the excitation for the simulation. Only pure sinusoidal
        excitation is possible.

        Parameters:
            excitation: Sets the amplitude for the sinusoidal excitation.
            is_disc: Set to true of the model is a disc.
        """
        self.piezo_freq.add_dirichlet_bc(
            FieldType.PHI,
            "Electrode",
            np.array([excitation])
        )
        self.piezo_freq.add_dirichlet_bc(
            FieldType.PHI,
            "Ground",
            np.zeros(1)
        )
        if is_disc:
            self.piezo_freq.add_dirichlet_bc(
                FieldType.U_R,
                "Symaxis",
                np.zeros(1)
            )

    def assemble(self):
        """Assembles the underlying simulations."""
        self.piezo_freq.assemble()
        self.thermo_time_sim.assemble()

    def simulate(
        self,
        piezo_freq_frequency: float,
        thermo_time_simulation_data: SimulationData,
        material_starting_temperature: float,
        temperature_dependent: bool
    ):
        """Runs the coupled simulation. The results are saved in the individual
        simulation classes.

        Parameters:
            piezo_freq_frequency: Frequency for which the frqeuency domain
                simulation is solved.
            thermo_time_simulation_data: Simulation data object for the
                    thermo time simulation.
            material_starting_temperature: Sets the temperature at time step
                0. Sets the starting temperature field as well as the material
                parameters at the start.
            temperature_dependent: Set to true if the materials shall be
                temperature dependent (Changing material parameters based on
                the current field temperature).
        """
        time_index = 0
        number_of_time_steps = thermo_time_simulation_data \
            .number_of_time_steps
        number_of_nodes = len(self.thermo_time_sim.mesh_data.nodes)
        number_of_elements = len(self.thermo_time_sim.mesh_data.elements)

        if temperature_dependent:
            temp_field_per_element = None
            while time_index < number_of_time_steps:
                # Run piezo_freq simulation
                if time_index == 0:
                    self.piezo_freq.material_manager.update_temperature(
                        material_starting_temperature * np.ones(
                            number_of_elements
                        )
                    )
                    self.piezo_freq.simulate(
                        np.array([piezo_freq_frequency]),
                        True
                    )
                else:
                    if temp_field_per_element is None:
                        print(
                            "Couldn't calculate temperature field in "
                            "previous time step"
                        )

                    self.piezo_freq.material_manager.update_temperature(
                        temp_field_per_element
                    )
                    self.piezo_freq.simulate(
                        np.array([piezo_freq_frequency]),
                        True,
                    )

                # Get mech losses
                mech_loss = np.real(
                    self.piezo_freq.mech_loss[time_index, :]
                )

                # Use mech losses for heat conduction simulation
                self.thermo_time_sim.f *= 0  # Reset load vector
                self.thermo_time_sim.set_constant_volume_heat_source(
                    mech_loss,
                    number_of_time_steps
                )

                # Run thermo time simulation. If the material parameters are
                # changed enough the simulation stops and a frequency domain
                # simulation is done.
                if time_index == 0:
                    self.thermo_time_sim.material_manager.update_temperature(
                        material_starting_temperature*np.ones(
                            number_of_elements)
                    )
                    time_index = self.thermo_time_sim \
                        .simulate_until_material_parameters_change(
                            thermo_time_simulation_data.delta_t,
                            thermo_time_simulation_data.number_of_time_steps,
                            thermo_time_simulation_data.gamma,
                            (
                                material_starting_temperature
                                * np.ones(number_of_nodes)
                            ),
                            0
                    )
                else:
                    time_index = self.thermo_time_sim \
                        .simulate_until_material_parameters_change(
                            thermo_time_simulation_data.delta_t,
                            thermo_time_simulation_data.number_of_time_steps,
                            thermo_time_simulation_data.gamma,
                            (
                                material_starting_temperature
                                * np.ones(number_of_nodes)
                            ),
                            time_index,
                        )

                if time_index == number_of_time_steps:
                    # Simulation is done
                    break

                # Simulation is not done. Get the temp per element and
                # in the next iteration set it for the piezo freq simulation
                # to calculate the new mech losses
                print(
                    f"Updating material parameters at time step {time_index}"
                )
                temp_field_per_element = get_avg_temp_field_per_element(
                    self.thermo_time_sim.theta[time_index, :],
                    self.thermo_time_sim.mesh_data.elements
                )
        else:
            # Run piezofreq simulation
            self.piezo_freq.simulate(
                np.array([piezo_freq_frequency]),
                True
            )

            # Get mech losses
            mech_loss = np.real(self.piezo_freq.mech_loss[time_index, :])

            # Use mech losses for heat conduction simulation
            self.thermo_time_sim.set_constant_volume_heat_source(
                mech_loss,
                number_of_time_steps
            )

            # Run thermo time domain simulation
            self.thermo_time_sim.simulate_until_material_parameters_change(
                thermo_time_simulation_data.delta_t,
                thermo_time_simulation_data.number_of_time_steps,
                thermo_time_simulation_data.gamma,
                material_starting_temperature*np.ones(number_of_nodes),
                0
            )

    def save_simulation_settings(self, directory: str):
        """Creates a folder 'simulation_name' in the given directory and saves
        the current simulation settings in it.

        Parameters:
            directory: Path to the directory in which the simulation folder is
                created.
        """
        self.piezo_freq.save_simulation_settings(directory, "piezo_freq")
        self.thermo_time_sim.save_simulation_settings(directory, "heat_cond")

    def save_simulation_results(self, directory: str):
        """Creates a folder 'simulation_name' in the given directory and saves
        the current simulation results in it.

        Parameters:
            directory: Path to the directory in which the simulation folder is
                created.
        """
        self.piezo_freq.save_simulation_results(directory, "piezo_freq")
        self.thermo_time_sim.save_simulation_results(directory, "heat_cond")
