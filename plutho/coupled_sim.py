"""Module for coupled simulations."""

# Python stanard libraries
from typing import Dict, Union

# Third party libraries
import numpy as np
import numpy.typing as npt

# Local libraries
# TODO Make relative
from plutho import SingleSimulation, SimulationData, Mesh, MaterialData, \
    FieldType
from plutho.simulation.base import get_avg_temp_field_per_element


class CoupledThermoPiezoThermoSim:
    """Couples a thermo-piezoelectric simulation with a heat conduction
    simulation where only a temperature field is calculated. The power
    losses from the thermo-piezoeletric simulation are assumed to be
    stationary at the last time steps, therefore they can be used as constant
    source in a heat conduction simulation. FOr this simulation a piezo
    with a 'Electrode' and 'Ground' physical group is assumed.
    Also a 'Symaxis' for the symmetric axis property is needed.

    Attributes:
        working_directory: Path to the directory where the simulation
            folder is created.
        simulation_name: Name of the simulation and the simulation folder.
        piezo_sim: Thermo-piezoeletric simulation.
        heat_cond_sim: Heat conduction simulation.
    """

    # Basic settings
    working_directory: str
    simulation_name: str

    # Simulations
    thermo_piezo_sim: SingleSimulation
    thermo_sim: SingleSimulation

    def __init__(
        self,
        working_directory: str,
        simulation_name: str,
        thermo_piezo_simulation_data: SimulationData,
        thermo_simulation_data: SimulationData,
        mesh: Mesh
    ):
        self.working_directory = working_directory
        self.simulation_name = simulation_name

        # Init simulations
        thermo_piezo_sim = SingleSimulation(
            working_directory,
            simulation_name,
            mesh
        )

        # TODO Unecessary to create a new simulation data object
        # in the function
        thermo_piezo_sim.setup_thermo_piezo_time_domain(
            thermo_piezo_simulation_data.number_of_time_steps,
            thermo_piezo_simulation_data.delta_t,
            thermo_piezo_simulation_data.gamma,
            thermo_piezo_simulation_data.beta
        )

        thermo_sim = SingleSimulation(
            working_directory,
            simulation_name,
            mesh
        )
        thermo_sim.setup_thermo_time_domain(
            thermo_simulation_data.delta_t,
            thermo_simulation_data.number_of_time_steps,
            thermo_simulation_data.gamma
        )

        self.thermo_piezo_sim = thermo_piezo_sim
        self.thermo_sim = thermo_sim

    def add_material(
            self,
            material_name: str,
            material_data: MaterialData,
            physical_group_name: Union[str, None]
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

        self.thermo_sim.solver.set_convection_bc(
            convective_bc_elements,
            alpha,
            outer_temperature
        )

    def simulate(
        self,
        averaging_time_step_count: int = 100,
        thermo_piezo_kwargs: Dict = {},
        thermo_kwargs: Dict = {}
    ):
        """Runs the coupled simulation. First the thermo-piezoelectric
        simulation is run. Then the power losses in the stationary state
        are calculated. Therefore an averaging user averinging_time_step_count
        number of time steps is done. After that the single thermo simulation
        is done.

        Parameters:
            averaging_time_step_count: Number of time steps starting from the
                last time step to calculate the average power loss.
            thermo_piezo_kwargs: Parameters given to the
                SingleSimulation.simulate() function for the thermo piezo
                simulation.
            thermo_kwargs: Parameters given to the SingleSimulation.simulate()
                function for the thermo simulation.

        """

        # Run thermo_piezoelectric simulation first
        self.thermo_piezo_sim.simulate(**thermo_piezo_kwargs)

        # Calculate mech losses which are sources in heat cond sim
        avg_mech_loss = np.mean(
            self.thermo_piezo_sim.solver.mech_loss[
                :,
                -averaging_time_step_count:
            ],
            axis=1
        )
        self.thermo_sim.solver.set_constant_volume_heat_source(
            avg_mech_loss,
            self.thermo_sim.solver.simulation_data.number_of_time_steps
        )

        # Set the result temp field of piezo sim to start field of heat cond
        # sim
        number_of_nodes = len(self.thermo_piezo_sim.mesh_data.nodes)
        theta_start = self.thermo_piezo_sim.solver.u[3*number_of_nodes:, -1]

        self.thermo_sim.simulate(
            initial_theta_field=theta_start,
            **thermo_kwargs
        )

    def save_simulation_results(self):
        """Saves the simulation results to the simulation folder."""
        self.thermo_piezo_sim.save_simulation_results("thermo_piezo")
        self.thermo_sim.save_simulation_results("heat_cond")

    def save_simulation_settings(self):
        """Saves the simulation settings to the simulation folder."""
        self.thermo_piezo_sim.save_simulation_settings("thermo_piezo")
        self.thermo_sim.save_simulation_settings("heat_cond")


class CoupledFreqPiezoTherm:

    # Basic setup
    working_directory: str
    simulation_name: str

    # Simulations
    piezo_freq: SingleSimulation
    thermo_time_sim: SingleSimulation

    def __init__(
        self,
        working_directory: str,
        simulation_name: str,
        piezo_freq_frequency: float,
        thermo_time_simulation_data: SimulationData,
        mesh: Mesh
    ):
        self.working_directory = working_directory
        self.simulation_name = simulation_name

        piezo_freq = SingleSimulation(
            working_directory,
            simulation_name,
            mesh
        )
        piezo_freq.setup_piezo_freq_domain(np.array([piezo_freq_frequency]))

        thermo_time_sim = SingleSimulation(
            working_directory,
            simulation_name,
            mesh
        )
        thermo_time_sim.setup_thermo_time_domain(
            thermo_time_simulation_data.delta_t,
            thermo_time_simulation_data.number_of_time_steps,
            thermo_time_simulation_data.gamma
        )

        self.piezo_freq = piezo_freq
        self.thermo_time_sim = thermo_time_sim

    def add_material(
        self,
        material_name: str,
        material_data: MaterialData,
        physical_group_name: Union[str, None]
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

    def simulate(
        self,
        material_starting_temperature: float,
        temperature_dependent: bool
    ):
        """Runs the coupled simulation. The results are saved in the individual
        simulation classes.

        Parameters:
            material_starting_temperature: Sets the temperature at time step 0.
                Sets the starting temperature field as well as the material
                parameters at the start.
            temperature_dependent: Set to true if the materials shall be
                temperature dependent (Changing material parameters based on the
                current field temperature).
        """
        time_index = 0
        number_of_time_steps = \
            self.thermo_time_sim.solver.simulation_data.number_of_time_steps
        number_of_nodes = len(self.thermo_time_sim.solver.mesh_data.nodes)

        if temperature_dependent:
            temp_field_per_element = None
            while time_index < number_of_time_steps:
                # Run piezo_freq simulation
                if time_index == 0:
                    self.piezo_freq.simulate(
                        calculate_mech_loss=True,
                        material_starting_temperature=material_starting_temperature
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
                        calculate_mech_loss=True,
                    )

                # Get mech losses
                mech_loss = np.real(
                    self.piezo_freq.solver.mech_loss[:, 0]
                )

                # Use mech losses for heat conduction simulation
                self.thermo_time_sim.solver.f *= 0  # Reset load vector
                self.thermo_time_sim.solver.set_constant_volume_heat_source(
                    mech_loss,
                    number_of_time_steps
                )

                # Run thermo time simulation. If the material parameters are
                # changed enough the simulation stops and a frequency domain
                # simulation is done.
                if time_index == 0:
                    time_index = self.thermo_time_sim.simulate(
                        initial_theta_time_step=0,
                        initial_theta_field=(
                            material_starting_temperature
                            * np.ones(number_of_nodes)
                        ),
                        material_starting_temperature=material_starting_temperature
                    )
                else:
                    time_index = self.thermo_time_sim.simulate(
                        initial_theta_time_step=time_index,
                        material_starting_temperature=material_starting_temperature
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
                    self.thermo_time_sim.solver.theta[:, time_index],
                    self.thermo_time_sim.solver.mesh_data.elements
                )
        else:
            # Run piezofreq simulation
            self.piezo_freq.simulate(
                calculate_mech_loss=True
            )

            # Get mech losses
            mech_loss = np.real(self.piezo_freq.solver.mech_loss[:, 0])

            # Use mech losses for heat conduction simulation
            self.thermo_time_sim.solver.set_constant_volume_heat_source(
                mech_loss,
                number_of_time_steps
            )

            # Run thermo time domain simulation
            self.thermo_time_sim.simulate(
                initial_theta_field=(
                    material_starting_temperature*np.ones(number_of_nodes)
                )
            )

    def save_simulation_results(self):
        """Saves the simulation results to the simulation folder."""
        self.piezo_freq.save_simulation_results("piezo_freq")
        self.thermo_time_sim.save_simulation_results("heat_cond")

    def save_simulation_settings(self):
        """Saves the simulation settings to the simulation folder."""
        self.piezo_freq.save_simulation_settings("piezo_freq")
        self.thermo_time_sim.save_simulation_settings("heat_cond")
