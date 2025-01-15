
# Python standard libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# Third party libraries
import tikzplotlib
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()

    MODEL_NAME = "coupled_pic181_thermo_time_60k_temp_independent_20c"
    PLOT_FOLDER = os.environ["piezo_fem_plot_path"]
    SIMULATION_FOLDER = os.environ["piezo_fem_simulation_path"]
    DELTA_T = 1e-8

    plt.rcParams.update({'font.size': 18})

    mech_loss_density = np.load(os.path.join(
        SIMULATION_FOLDER,
        MODEL_NAME,
        "thermo_piezo_mech_loss.npy"
    ))
    number_of_time_steps = mech_loss_density.shape[1]
    element_indices = [428]
    #print(np.real(mech_loss_density[428]))
    print(
        "Mittlere Verlustleistungsdichte:",
        np.mean(
            mech_loss_density[428, -1000:]
        )
    )

    for element_index in element_indices:
        plt.figure(figsize=(12, 6))
        plt.plot(
            (np.arange(number_of_time_steps)*DELTA_T),
            mech_loss_density[element_index, :],
            label=f"Element {element_index}")
        plt.xlabel("Zeit $t$ / s")
        plt.ylabel(
            "Verlustleistungsdichte $\\dot{W}$ / "
            "$\\frac{\\mathrm{W}}{\\mathrm{m}^{3}}$"
        )
        plt.grid()
        plt.legend()
        plt.show()

        if False:
            plt.savefig(os.path.join(
                PLOT_FOLDER,
                f"mech_loss_density_{element_index}.png"
            ))
            tikzplotlib.save(os.path.join(
                PLOT_FOLDER,
                f"mech_loss_density_{element_index}.tex"
            ))
