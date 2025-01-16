
# Python standard libraries
import os

# Third party libraries
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from scipy import interpolate
from dotenv import load_dotenv

# Local libraries
import piezo_fem as pfem


def plot_mech_loss_density(
        mech_loss,
        time_values,
        element_indices,
        tikz_plot_path=""):

    plt.figure(figsize=(12, 6))
    for element_index in element_indices:
        plt.plot(
            time_values[-500:],
            mech_loss[element_index, -500:],
            label=f"Element {element_index}"
        )

    plt.xlabel("Zeit $t$ / s")
    plt.ylabel(
        "Verlustleistungsdichte $\\dot{W}$ / "
        "$\\frac{\\mathrm{W}}{\\mathrm{m}^{3}}$"
    )
    plt.grid()
    plt.legend()
    # plt.show()

    if tikz_plot_path != "":
        tikzplotlib.save(tikz_plot_path)


if __name__ == "__main__":
    load_dotenv()

    SIM_NAME = "pic255_thermo_time_20k"
    simulation_folder = os.environ["piezo_fem_simulation_path"]
    plot_folder = os.environ["piezo_fem_plot_path"]

    mech_file_name = "thermo_piezo_mech_loss.npy"

    mech_loss = np.load(os.path.join(
        simulation_folder,
        SIM_NAME,
        mech_file_name
    ))
    element_indices = [1079]

    number_of_time_steps = 20000
    delta_t = 1e-8
    time_values = np.arange(number_of_time_steps)*delta_t
    plot_mech_loss_density(
        mech_loss,
        time_values,
        element_indices,
        os.path.join(plot_folder, "mech_loss_density_1079_fine.tex")
    )

