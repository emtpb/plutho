import os
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

if __name__ == "__main__":
    PLOT_FOLDER = "/home/jonash/uni/Masterarbeit/plots/"
    NUMBER_OF_TIME_STEPS = 10000
    DELTA_T = 1e-8    
    
    plt.rcParams.update({'font.size': 18})

    mech_loss_density = np.load(os.path.join(
        "/home/jonash/uni/Masterarbeit/simulations/real_model_10k/",
        "real_model_10k_mech_loss.npy"
    ))
    element_indices = [379, 1079]
    for element_index in element_indices:
        plt.figure(figsize=(12,6))
        plt.plot(
            np.arange(NUMBER_OF_TIME_STEPS)*DELTA_T,
            mech_loss_density[element_index, :],
            label=f"Element {element_index}")
        plt.xlabel("Time / s")
        plt.ylabel("Loss density / $\\frac{\\mathrm{W}}{\\mathrm{m}^{3}}$")
        plt.grid()
        plt.legend()

        plt.savefig(os.path.join(
            PLOT_FOLDER,
            f"mech_loss_density_{element_index}.png"
        ))
        tikzplotlib.save(os.path.join(
            PLOT_FOLDER,
            f"mech_loss_density_{element_index}.tex"
        ))
