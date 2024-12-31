"""Module for coupled simulations."""

# Python stanard libraries

# Third party libraries
import numpy as np
import numpy.typing as npt

# Local libraries
from piezo_fem import SingleSimulation, SimulationData, Mesh, MaterialData, \
    FieldType
from piezo_fem.simulation.base import get_avg_temp_field_per_element


class CoupledThermPiezoHeatCond:
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
    piezo_sim: SingleSimulation
    heat_cond_sim: SingleSimulation

    def __init__(
            self,
            working_directory: str,
            simulation_name: str,
            thermo_piezo_simulation_data: SimulationData,
            heat_cond_simulation_data: SimulationData,
            mesh: Mesh):
        self.working_directory = working_directory
        self.simulation_name = simulation_name

        # Init simulations
        piezo_sim = SingleSimulation(
            working_directory,
            simulation_name,
            mesh
        )
        piezo_sim.setup_thermal_piezo_time_domain(
            thermo_piezo_simulation_data
        )

        heat_cond_sim = SingleSimulation(
            working_directory,
            simulation_name,
            mesh
        )
        heat_cond_sim.setup_heat_conduction_time_domain(
            heat_cond_simulation_data.delta_t,
            heat_cond_simulation_data.number_of_time_steps,
            heat_cond_simulation_data.gamma
        )

        self.piezo_sim = piezo_sim
        self.heat_cond_sim = heat_cond_sim

    def add_material(
            self,
            material_name: str,
            material_data: MaterialData,
            physical_group_name: str):
        """Add a material to the simulation.

        Parameters:
            material_name: Name of the material.
            material_data: MaterialData object containing the material
                parameters
            physical_group_name: Name of the gmsh physical group for which
                this material shall be set.
        """
        self.piezo_sim.add_material(
            material_name,
            material_data,
            physical_group_name
        )
        self.heat_cond_sim.add_material(
            material_name,
            material_data,
            physical_group_name
        )

    def set_excitation(
            self,
            excitation: npt.NDArray,
            is_disc: bool):
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
        self.piezo_sim.add_dirichlet_bc(
            FieldType.PHI,
            "Electrode",
            excitation
        )
        self.piezo_sim.add_dirichlet_bc(
            FieldType.PHI,
            "Ground",
            np.zeros(len(excitation))
        )
        if is_disc:
            self.piezo_sim.add_dirichlet_bc(
                FieldType.U_R,
                "Symaxis",
                np.zeros(len(excitation))
            )

    def simulate(self, **kwargs):
        """Runs the coupled simulation. First the thermo-piezoelectric
        simulation is run. Then the power losses in the stationary state
        are calculated. After that the heat conduction simulation is done.

        Parameters:
            kwargs:
                - 'averaging_time_step_cound': The number of time steps
                    starting from the last time step to calculate the average
                    power loss. Default is 100"""
        # Run thermo_piezoelectric simulation first
        self.piezo_sim.simulate(**kwargs)

        # Calculate mech losses which are sources in heat cond sim
        averaging_time_step_count = 100
        if "averaging_time_step_count" in kwargs:
            averaging_time_step_count = kwargs["averaging_time_step_count"]
        avg_mech_loss = np.mean(
            self.piezo_sim.solver.mech_loss[:, -averaging_time_step_count:],
            axis=1
        )
        self.heat_cond_sim.solver.set_constant_volume_heat_source(
            avg_mech_loss,
            self.heat_cond_sim.solver.simulation_data.number_of_time_steps
        )

        # Set the result temp field of piezo sim to start field of heat cond
        # sim
        number_of_nodes = len(self.piezo_sim.mesh_data.nodes)
        theta_start = self.piezo_sim.solver.u[3*number_of_nodes:, -1]

        self.heat_cond_sim.simulate(theta_start=theta_start)

    def save_simulation_results(self):
        """Saves the simulation results to the simulation folder."""
        self.piezo_sim.save_simulation_results("thermo_piezo")
        self.heat_cond_sim.save_simulation_results("heat_cond")


class CoupledFreqPiezoHeatCond:

    # Basic setup
    working_directory: str
    simulation_name: str

    # Simulations
    piezo_freq: SingleSimulation
    heat_cond_sim: SingleSimulation

    def __init__(
            self,
            working_directory: str,
            simulation_name: str,
            piezo_sim_frequency: float,
            heat_cond_simulation_data: SimulationData,
            mesh: Mesh):
        self.working_directory = working_directory
        self.simulation_name = simulation_name

        piezo_freq = SingleSimulation(
            working_directory,
            simulation_name,
            mesh
        )
        piezo_freq.setup_piezo_freq_domain(np.array([piezo_sim_frequency]))

        heat_cond_sim = SingleSimulation(
            working_directory,
            simulation_name,
            mesh
        )
        heat_cond_sim.setup_heat_conduction_time_domain(
            heat_cond_simulation_data.delta_t,
            heat_cond_simulation_data.number_of_time_steps,
            heat_cond_simulation_data.gamma
        )

        self.piezo_freq = piezo_freq
        self.heat_cond_sim = heat_cond_sim

    def add_material(
            self,
            material_name: str,
            material_data: MaterialData,
            physical_group_name: str):
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
        self.heat_cond_sim.add_material(
            material_name,
            material_data,
            physical_group_name
        )

    def set_excitation(
            self,
            excitation: float,
            is_disc: bool = True):
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

    def simulate_coupled(self, starting_temperature, is_temperature_dependent):

        time_index = 0
        number_of_time_steps = \
            self.heat_cond_sim.solver.simulation_data.number_of_time_steps

        if is_temperature_dependent:
            temp_field_per_element = None
            while time_index < number_of_time_steps:
                # Run piezofreq simulation
                if time_index == 0:
                    self.piezo_freq.simulate(
                        calculate_mech_loss=True,
                        starting_temperature=starting_temperature
                    )
                else:
                    self.piezo_freq.simulate(
                        calculate_mech_loss=True,
                        starting_temperature=temp_field_per_element
                    )
                # Get mech losses
                mech_loss = np.real(self.piezo_freq.solver.mech_loss[:, 0])

                # Use mech losses for heat conduction simulation
                self.heat_cond_sim.solver.f *= 0  # Reset load vector
                self.heat_cond_sim.solver.set_constant_volume_heat_source(
                    mech_loss,
                    number_of_time_steps
                )
                if time_index == 0:
                    time_index = self.heat_cond_sim.simulate(
                        time_step=0,
                        starting_temperature=starting_temperature
                    )
                else:
                    time_index = self.heat_cond_sim.simulate(
                        time_step=time_index,
                        starting_temperature=temp_field_per_element,
                        theta_start=self.heat_cond_sim.solver.theta[
                            :,
                            time_index-1
                        ]
                    )
                if time_index == number_of_time_steps:
                    break

                # Simulation is not done. Get the temp per element and
                # in the next iteration set it for the piezo freq simulation
                # to calculate the new mech losses
                print(
                    f"Updating material parameters at time step {time_index}"
                )
                temp_field_per_element = get_avg_temp_field_per_element(
                    self.heat_cond_sim.solver.theta[:, time_index],
                    self.heat_cond_sim.solver.mesh_data.elements
                )

                time_index += 1
        else:
            # Run piezofreq simulation
            self.piezo_freq.simulate(calculate_mech_loss=True)

            # Get mech losses
            mech_loss = np.real(self.piezo_freq.solver.mech_loss[:, 0])

            # Use mech losses for heat conduction simulation
            self.heat_cond_sim.solver.set_constant_volume_heat_source(
                mech_loss,
                number_of_time_steps
            )
            self.heat_cond_sim.simulate()

    def save_simulation_results(self):
        """Saves the simulation results to the simulation folder."""
        self.piezo_freq.save_simulation_results("piezo_freq")
        self.heat_cond_sim.save_simulation_results("heat_cond")
