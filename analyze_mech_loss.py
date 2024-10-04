import os
import numpy as np 
import matplotlib.pyplot as plt

import scipy
import scipy.signal
import piezo_fem as pfem


def get_stationary_mech_loss(
        mech_loss,
        number_of_time_steps,
        delta_t,
        frequency,
        stationary_compare_distance = 1000,
        consecutive_indices = 10):
    # Find index of where the gradient of the average is stationary
    means = np.mean(mech_loss, axis=0)
    grad_means = np.gradient(np.mean(mech_loss, axis=0))
    means_hilbert = np.abs(scipy.signal.hilbert(grad_means))

    # Calculate running average (over time) over multiple periods
    time_steps_per_period = int(1/(frequency*delta_t))
    window_width = 5*time_steps_per_period
    running_avg = np.convolve(
        means_hilbert,
        np.ones(window_width)/(5*time_steps_per_period),
        mode="valid")

    stationary_indices = []
    for index, _ in enumerate(running_avg):
        if index < stationary_compare_distance:
            continue

        if np.isclose(
                running_avg[index],
                running_avg[index-stationary_compare_distance]):
            stationary_indices.append(index)

    # In order to find a proper stationary index iterate backwards thorugh the
    # list of possible indices and search for at least x consecutive indices
    stationary_index = -1
    for i in reversed(range(len(stationary_indices))):
        for j in range(consecutive_indices):
            if not np.isclose(stationary_indices[i], stationary_indices[i-j]):
                break

        stationary_index = i
        break

    if stationary_index == -1:
        raise ValueError("No proper stationary index found")

    print(stationary_index)

    stat_mech_loss_per_period = np.zeros(mech_loss.shape[0])
    for node_index in range(mech_loss.shape[0]):
        loss = 0
        for time_index in range(
                stationary_index,
                stationary_index+time_steps_per_period):
            loss += mech_loss[node_index, time_index]

        stat_mech_loss_per_period[node_index] = loss

    return stationary_index, stat_mech_loss_per_period


if __name__ == "__main__":
    MODEL_NAME = "real_model_30k"
    WD = os.path.join(
        "/home/jonash/uni/Masterarbeit/simulations", MODEL_NAME)
    LOSS_FILE = os.path.join(
        WD,
        f"{MODEL_NAME}_mech_loss.npy"
    )
    CONFIG_FILE = os.path.join(
        WD,
        f"{MODEL_NAME}.cfg"
    )
    DISPLACEMENT_FILE = os.path.join(
        WD,
        f"{MODEL_NAME}_u.npy"
    )
    sim = pfem.Simulation.load_simulation_settings(CONFIG_FILE)
    frequency = sim.excitation_info.frequency
    mech_loss = np.load(LOSS_FILE)
    u = np.load(DISPLACEMENT_FILE)

    stat_index, loss_per_period = get_stationary_mech_loss(
        mech_loss,
        sim.simulation_data.number_of_time_steps,
        sim.simulation_data.delta_t,
        sim.excitation_info.frequency
    )
    time_steps_per_period = int(1/(
        sim.excitation_info.frequency
        * sim.simulation_data.delta_t
    ))

    # Get theta field at stat_index
    number_of_periods = 10
    number_of_nodes = len(sim.mesh_data.nodes)
    theta_start = u[3*number_of_nodes:, stat_index]
    theta_end = u[3*number_of_nodes:, stat_index+number_of_periods*time_steps_per_period]

    stat_diff = theta_end-theta_start

    # Simulate x seconds
    SECONDS = 2
    future_time_steps = int(SECONDS/(time_steps_per_period*number_of_periods*sim.simulation_data.delta_t))
    print("Number of future time steps:", future_time_steps)
    theta = np.zeros(shape=(number_of_nodes, future_time_steps))
    theta[:, 0] = theta_start
    theta[:, 1] = theta_end
    for time_step in range(2, future_time_steps):
        theta[:, time_step] = theta[:, time_step-1] + stat_diff

    print(theta[:, -1])
