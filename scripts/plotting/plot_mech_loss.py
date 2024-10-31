import os
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

if __name__ == "__main__":
    PLOT_FOLDER = "/home/jonash/uni/Masterarbeit/plots/"
    #PLOT_FOLDER = "smb://ad%5Cjonasho@fs-cifs.uni-paderborn.de/upb/departments/emt/Student/jonasho/Masterarbeit/simulations/"
    DELTA_T = 1e-8    
    
    plt.rcParams.update({'font.size': 18})

    sim_name = "real_model_30k"
    mech_loss_density = np.load(os.path.join(
        f"/home/jonash/uni/Masterarbeit/simulations/{sim_name}/",
        f"{sim_name}_mech_loss.npy"
    ))
    number_of_time_steps = mech_loss_density.shape[1]
    element_indices = [379, 1079]
    for element_index in element_indices:
        plt.figure(figsize=(12,6))
        plt.plot(
            np.arange(number_of_time_steps)*DELTA_T,
            mech_loss_density[element_index, :],
            label=f"Element {element_index}")
        plt.xlabel("Zeit / s")
        plt.ylabel("Verlustleistungsdichte / $\\frac{\\mathrm{W}}{\\mathrm{m}^{3}}$")
        plt.grid()
        plt.legend()
        #plt.show()
        plt.savefig(os.path.join(
            PLOT_FOLDER,
            f"mech_loss_density_{element_index}.png"
        ))
        tikzplotlib.save(os.path.join(
            PLOT_FOLDER,
            f"mech_loss_density_{element_index}.tex"
        ))
