"""Script for a thermal simulation after a thermo-piezo simulation
in order to calculate the Temperature for longer time intervals."""

# Python standard libraries
import os
import numpy as np
import numpy.typing as npt

# Third party libraries
import gmsh

# Local libraries
import piezo_fem as pfem


def calculate_avg_losses_per_element(
        mech_loss: npt.NDArray,
        number_of_elements: float,
        piezo_sim_frequency: float,
        piezo_sim_delta_t: float):
    """Calculates the average losses per element. Does only work properly for
    sinusoidal excitation"""
    # TODO 2 because the power has double frequency??
    time_steps_per_period = int(1/(2*piezo_sim_frequency*piezo_sim_delta_t))

    # Averages the losses for the last period.
    avg_losses = np.zeros(number_of_elements)
    for element_index in range(number_of_elements):
        avg_losses[element_index] = np.mean(
            mech_loss[element_index, -time_steps_per_period:])

    return avg_losses


def run_piezo_thermal_simulation(base_directory, name):
    """Example for a thermo piezoelectric simulation of a disc.

    Parameters:
        base_directory: Directory where the simulation directory is created.
    """
    piezo_delta_t = 1e-8
    frequency = 2e6

    sim_directory = os.path.join(base_directory, name)
    sim = pfem.Simulation(sim_directory, pfem.pic255, name)
    sim.create_disc_mesh(0.005, 0.001, 0.00015)
    sim.set_simulation(
        delta_t=piezo_delta_t,
        number_of_time_steps=100,
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
        "An example for a thermal piezo-electric simulation.")
    sim.simulate()

    sim.save_simulation_results()


if __name__ == "__main__":
    CWD = "/home/jonash/uni/Masterarbeit/simulations/"
    PIEZO_SIM_NAME = "double_sim"
    # run_piezo_thermal_simulation(CWD, PIEZO_SIM_NAME)

    # Load data from piezo sim
    piezo_sim_folder = os.path.join(CWD, PIEZO_SIM_NAME)
    u = np.load(os.path.join(piezo_sim_folder, f"{PIEZO_SIM_NAME}_u.npy"))
    mech_losses = np.load(
        os.path.join(piezo_sim_folder, f"{PIEZO_SIM_NAME}_mech_loss.npy"))
    piezo_sim = pfem.Simulation.load_simulation_settings(
        os.path.join(piezo_sim_folder, f"{PIEZO_SIM_NAME}.cfg")
    )

    avg_mech_losses = calculate_avg_losses_per_element(
        mech_losses,
        len(piezo_sim.mesh_data.elements),
        piezo_sim.excitation_info.frequency,
        piezo_sim.simulation_data.delta_t
    )

    # Heat conduction simulation settings
    SIMULATION_TIME = 1  # In seconds

    # Number of periods of the piezo sim which are simulated in one time step
    # in the heat conduction simulation
    SKIPPED_PERIODS = 1

    # Set heat conduction simulation settings
    delta_t = SKIPPED_PERIODS*piezo_sim.simulation_data.delta_t
    # number_of_time_steps = int(SIMULATION_TIME/delta_t)
    number_of_time_steps = 5
    heat_simulation_data = pfem.SimulationData(
        delta_t=delta_t,
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
    heat_sim.set_excitation(
        SKIPPED_PERIODS*avg_mech_losses
    )

    # The start temperature field of the heat conduction sim is set to the
    # field of the last time step of the piezo sim
    number_of_nodes = len(piezo_sim.mesh_data.nodes)
    theta_start = u[3*number_of_nodes:, -1]
    heat_sim.assemble()
    heat_sim.solve_time(theta_start)

    theta = heat_sim.theta
    piezo_sim.gmsh_handler.create_theta_post_processing_view(
        theta,
        number_of_time_steps,
        delta_t,
        False
    )
    gmsh.fltk.run()
