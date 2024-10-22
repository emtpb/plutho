"""Script for a thermal simulation after a thermo-piezo simulation
in order to calculate the Temperature for longer time intervals."""

# Python standard libraries
import os
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# Third party libraries
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
    sim.create_disc_mesh(0.005, 0.001, 0.00015)
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
        "An example for a thermal piezo-electric simulation.")
    sim.simulate()

    sim.save_simulation_results()


def interpolate_mech_loss(
        mech_loss,
        time_steps_per_period,
        number_of_periods,
        delta_t):
    time_values = np.arange(mech_loss.shape[1])*delta_t
    fits = np.zeros(shape=(mech_loss.shape[0], 2))
    for element_index in range(mech_loss.shape[0]):
        current_mech_loss = rolling_avg[
            -number_of_periods*time_steps_per_period:
        ]
        current_time_values = time_values[
                -number_of_periods*time_steps_per_period:
        ]

        fits[element_index] = np.polyfit(
            current_time_values,
            current_mech_loss,
            deg=1)
    return fits

def interpolate_avg_losses(
        mech_loss,
        delta_t,
        number_of_average_periods,
        time_steps_per_period):
    fits = np.zeros(shape=(mech_loss.shape[0], 2))
    time_values = np.arange(mech_loss.shape[1])*delta_t
    for element_index in range(mech_loss.shape[0]):
        current_mech_loss = mech_loss[element_index, :]

        # TODO Maybe take last x periods and calculate rolling average
        # Right now the whole rolling avg is calculated
        # Calculate rolling avg
        T = number_of_average_periods*time_steps_per_period
        rolling_avg = np.convolve(
            current_mech_loss,
            np.ones(T)/T,
            mode="valid"
        )

        # Take last periods and interpolate
        current_time_values = time_values[len(rolling_avg)-T:len(rolling_avg)]
        current_rolling_avg = rolling_avg[-T:]

        #fits[element_index] = Polynomial.fit(
        #    current_time_values,
        #    current_rolling_avg,
        #    1
        #)
        fit = np.polyfit(
            current_time_values, current_rolling_avg, deg=1
        )
        fits[element_index] = fit
        #print(fit)
        #pol = np.poly1d(fit)
        #plt.plot(time_values, current_mech_loss, "-", label="data")
        #plt.plot(time_values, pol(time_values), label="long fit")
        #plt.plot(current_time_values, pol(current_time_values), label="short fit")
        #plt.plot(time_values[:len(rolling_avg)], rolling_avg, label="rolling avg")
        #plt.grid()
        #plt.legend()
        #plt.show()

    return fits

def calculate_avg_loss_density_per_element(
        mech_loss_density,
        time_steps_per_period,
        delta_t):
    avg_losses = np.zeros(mech_loss_density.shape[0])
    for element_index in range(mech_loss_density.shape[0]):
        period = delta_t*time_steps_per_period
        avg_losses[element_index] = 1/period*np.trapezoid(
            mech_loss_density[element_index, -time_steps_per_period:])

    return avg_losses

if __name__ == "__main__":
    #CWD = "/home/jonash/uni/Masterarbeit/simulations/"
    CWD = "/upb/users/j/jonasho/scratch/piezo_fem/results/"
    #PIEZO_SIM_NAME = "double_sim"
    PIEZO_SIM_NAME = "real_model_15k"
    #run_piezo_thermal_simulation(CWD, PIEZO_SIM_NAME)

    # Load data from piezo sim
    piezo_sim_folder = os.path.join(CWD, PIEZO_SIM_NAME)
    u = np.load(os.path.join(piezo_sim_folder, f"{PIEZO_SIM_NAME}_u.npy"))
    mech_loss_density = np.load(
        os.path.join(piezo_sim_folder, f"{PIEZO_SIM_NAME}_mech_loss.npy"))
    piezo_sim = pfem.Simulation.load_simulation_settings(
        os.path.join(piezo_sim_folder, f"{PIEZO_SIM_NAME}.cfg")
    )

    NUMBER_OF_AVERAGE_PERIODS = 1
    TIME_STEPS_PER_PERIOD = 25  # Empirically
    T = NUMBER_OF_AVERAGE_PERIODS*TIME_STEPS_PER_PERIOD
    # fits = interpolate_avg_losses(
    #         mech_loss_density,
    #         piezo_sim.simulation_data.delta_t,
    #         number_of_average_periods,
    #         time_steps_per_period
    # )

    # Heat conduction simulation settings
    SIMULATION_TIME = 1  # In seconds

    # Number of periods of the piezo sim which are simulated in one time step
    # in the heat conduction simulation
    SKIPPED_PERIODS = 100000

    # Set heat conduction simulation settings
    delta_t = SKIPPED_PERIODS*piezo_sim.simulation_data.delta_t
    number_of_time_steps = int(SIMULATION_TIME/delta_t)
    print("Total number of time steps:", number_of_time_steps)

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

    # interpolations = []
    # for fit in fits:
    #     interpolations.append(np.poly1d(fit))

    # element_indices = [379, 1196]

    # Set excitation for heat conduction sim
    # Multiplied with the number of skipped periods since the avg mech losses
    # represent the power over one period
    heat_sim.set_constant_volume_heat_source(
        SKIPPED_PERIODS*calculate_avg_loss_density_per_element(
            mech_loss_density,
            TIME_STEPS_PER_PERIOD,
            piezo_sim.simulation_data.delta_t
        ),
        number_of_time_steps
    )
    #heat_sim.set_volume_heat_source(
    #    interpolations,
    #    number_of_time_steps,
    #    delta_t
    #)

    # The start temperature field of the heat conduction sim is set to the
    # field of the last time step of the piezo sim
    number_of_nodes = len(piezo_sim.mesh_data.nodes)
    theta_start = u[3*number_of_nodes:, -1]
    heat_sim.assemble()
    heat_sim.solve_time(theta_start)

    theta = heat_sim.theta
    gmsh_handler = pfem.GmshHandler(
        os.path.join(CWD, "temperature_only.msh")
    )
    gmsh_handler.create_theta_post_processing_view(
        theta,
        number_of_time_steps,
        delta_t,
        False
    )

    #input_energy = pfem.calculate_electrical_input_energy(
    #    piezo_sim.excitation,
    #    np.load(os.path.join(piezo_sim_folder, f"{PIEZO_SIM_NAME}_q.npy")),
    #    piezo_sim.simulation_data.delta_t
    #)

    gmsh.fltk.run()
