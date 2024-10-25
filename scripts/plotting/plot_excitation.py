import os
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

if __name__ == "__main__":
    PLOT_FOLDER = "/home/jonash/uni/Masterarbeit/plots/"
    NUMBER_OF_TIME_STEPS = 100
    DELTA_T = 1e-8
    FREQUENCY = 2e6
    
    plt.rcParams.update({'font.size': 18})

    time_values = np.arange(NUMBER_OF_TIME_STEPS)*DELTA_T
    excitation = np.zeros(NUMBER_OF_TIME_STEPS)
    #excitation[1:10] = (
    #    1
    #    * np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    #)
    excitation = 20*np.sin(2*np.pi*time_values*FREQUENCY)
    plt.figure(figsize=(10,8))
    plt.plot(time_values, excitation)
    plt.xlabel("Time / s")
    plt.ylabel("Excitation / V")
    plt.grid()

    plt.savefig(os.path.join(
        PLOT_FOLDER,
        "excitation_sin.png"
    ))
    tikzplotlib.save(os.path.join(
        PLOT_FOLDER,
        "excitation_sin.tex"
    ))
