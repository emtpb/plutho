import os
import numpy as np 
import matplotlib.pyplot as plt

import scipy
import scipy.signal
import piezo_fem as pfem


def get_stationary_index(mech_loss, number_of_time_steps, delta_t, frequency):
    means = np.mean(mech_loss, axis=0)
    hilbert = scipy.signal.hilbert(means).imag
    #plt.plot(means, label="Averages")
    plt.plot(np.gradient(means), label="d/dt")
    #plt.plot(np.abs(hilbert), label="Hilbert")
    plt.plot(np.abs(scipy.signal.hilbert(np.gradient(means))), label="d/dt Hilbert")
    plt.grid()
    plt.legend()
    plt.show()
    # return np.mean(means[-10:])


def get_stationary_index_2(mech_loss, number_of_time_steps, delta_t, frequency):
    # Check when the average is stationary
    time_steps_per_period = int(1/(frequency*delta_t))
    print(time_steps_per_period)

    threshold = 1.148e-6  # empirically estimated
    threshold = 0.05

    # Average over all points per time step
    periods_per_box = 1
    means = np.mean(mech_loss, axis=0)
    avgs = np.zeros(int(len(means)/(periods_per_box*time_steps_per_period)))
    # Running average over 3 periods
    for avg_index in range(avgs.shape[0]):
        current_box = means[avg_index*time_steps_per_period:periods_per_box*time_steps_per_period*(avg_index+1)]
        avgs[avg_index] = np.mean(current_box)

    plt.plot(means)
    plt.show()

    stationary_indices = []
    for index, _ in enumerate(avgs):
        if index == 0:
            continue

        #if np.abs(avgs[index]-avgs[index-1]) < threshold:
        #    stationary_indices.append(index)
        if np.isclose(avgs[index], avgs[index-1]):
            stationary_indices.append(index)

    print(stationary_indices)

    plt.grid()
    plt.legend()
    #plt.show()


if __name__ == "__main__":
    MODEL_NAME = "real_model_30k"
    WD = os.path.join(
        "/upb/users/j/jonasho/scratch/piezo_fem/results/", MODEL_NAME)
    LOSS_FILE = os.path.join(
        WD,
        f"{MODEL_NAME}_mech_loss.npy"
    )
    CONFIG_FILE = os.path.join(
        WD,
        f"{MODEL_NAME}.cfg"
    )

    sim = pfem.Simulation.load_simulation_settings(CONFIG_FILE)
    frequency = sim.excitation_info.frequency
    mech_loss = np.load(LOSS_FILE)

    get_stationary_index(
        mech_loss,
        sim.simulation_data.number_of_time_steps,
        sim.simulation_data.delta_t,
        sim.excitation_info.frequency
    )
