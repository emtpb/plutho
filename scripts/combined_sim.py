"""Script for a thermal simulation after a thermo-piezo simulation
in order to calculate the Temperature for longer time intervals."""

# Python standard libraries
import os
import numpy as np
import numpy.typing as npt

# Third party libraries
from dotenv import load_dotenv
import gmsh

# Local libraries
import piezo_fem as pfem


def run_piezo_thermal_simulation(base_directory, name):
    """Example for a thermo piezoelectric simulation of a disc.

    Parameters:
        base_directory: Directory where the simulation directory is created.
    """
    piezo_delta_t = 1e-8
    frequency = 2e6

    sim_directory = os.path.join(base_directory, name)
    sim = pfem.Simulation(sim_directory, pfem.pic255, name)
    sim.create_disc_mesh(0.005, 0.001, 0.0001)
    sim.set_simulation(
        delta_t=piezo_delta_t,
        number_of_time_steps=10000,
        gamma=0.5,
        beta=0.25,
        simulation_type=pfem.SimulationType.THERMOPIEZOELECTRIC,
    )
    sim.set_sinusoidal_excitation(
        20,
        frequency
    )
    sim.set_boundary_conditions()
    sim.save_simulation_settings(
        "An example for a thermal piezo-electric simulation."
        "With convective boundary condition for iron.")
    sim.simulate()

    sim.save_simulation_results()


def calculate_avg_loss_density_per_element(
        mech_loss_density: npt.NDArray,
        time_steps_per_period: int):
    """Calculates the avg loss density per element at the last time step.

    Parameters:
        mech_loss_density: Array containing the mech loss density per 
            element per time.
        time_steps_per_period: How many time steps per mech_loss_density
            period (Half the excitation period).
        delta_t: Difference between time steps.
    """
    avg_losses = np.zeros(mech_loss_density.shape[0])
    for element_index in range(mech_loss_density.shape[0]):
        avg_losses[element_index] = np.mean(
            mech_loss_density[element_index, -time_steps_per_period:]
        )

    return avg_losses


if __name__ == "__main__":
    load_dotenv()

    CWD = os.getenv("piezo_fem_simulation_path")
    if CWD is None:
        print("Couldn't find simulation path.")
        exit(1)

    PIEZO_SIM_NAME = "real_model_30k"
    # run_piezo_thermal_simulation(CWD, PIEZO_SIM_NAME)

    # Load data from piezo sim
    piezo_sim_folder = os.path.join(CWD, PIEZO_SIM_NAME)
    u = np.load(os.path.join(piezo_sim_folder, f"{PIEZO_SIM_NAME}_u.npy"))
    piezo_mech_loss_density = np.load(
        os.path.join(piezo_sim_folder, f"{PIEZO_SIM_NAME}_mech_loss.npy"))
    piezo_sim = pfem.Simulation.load_simulation_settings(
        os.path.join(piezo_sim_folder, f"{PIEZO_SIM_NAME}.cfg")
    )

    TIME_STEPS_PER_PERIOD = 25  # Empirically

    # Heat conduction simulation settings
    SIMULATION_TIME = 1  # In seconds

    # Number of periods of the piezo sim which are simulated in one time step
    # in the heat conduction simulation
    SKIPPED_TIME_STEPS = 100000

    # Set heat conduction simulation settings
    heat_cond_delta_t = SKIPPED_TIME_STEPS*piezo_sim.simulation_data.delta_t
    number_of_time_steps = int(SIMULATION_TIME/heat_cond_delta_t)
    print("Total number of time steps:", number_of_time_steps)

    heat_simulation_data = pfem.SimulationData(
        delta_t=heat_cond_delta_t,
        number_of_time_steps=number_of_time_steps,
        gamma=0.5,
        beta=0  # Doesnt matter in heat conduction sim
    )

    # Create heat conduction sim
    heat_sim = pfem.HeatConductionSim(
        piezo_sim.mesh_data,
        piezo_sim.material_data,
        heat_simulation_data
    )

    # Set excitation for heat conduction sim
    # Multiplied with the number of skipped periods since the avg mech losses
    # represent the power over one period
    heat_sim.set_constant_volume_heat_source(
        SKIPPED_TIME_STEPS/TIME_STEPS_PER_PERIOD *
        calculate_avg_loss_density_per_element(
            piezo_mech_loss_density,
            TIME_STEPS_PER_PERIOD
        ),
        number_of_time_steps
    )

    # The start temperature field of the heat conduction sim is set to the
    # field of the last time step of the piezo sim
    number_of_nodes = len(piezo_sim.mesh_data.nodes)
    theta_start = u[3*number_of_nodes:, -1]
    heat_sim.assemble()
    heat_sim.solve_time(theta_start)

    theta = heat_sim.theta
    gmsh_handler = pfem.GmshHandler(
        os.path.join(piezo_sim_folder, "temperature_only.msh")
    )
    gmsh_handler.create_theta_post_processing_view(
        theta,
        number_of_time_steps,
        heat_cond_delta_t,
        False
    )

    #input_energy = pfem.calculate_electrical_input_energy(
    #    piezo_sim.excitation,
    #    np.load(os.path.join(piezo_sim_folder, f"{PIEZO_SIM_NAME}_q.npy")),
    #    piezo_sim.simulation_data.delta_t
    #)

    gmsh.fltk.run()
