"""Script to plot the impedence of the FEM simulation and or an
openCFS simulation.
"""

# Python standard libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# Third party libraries
from dotenv import load_dotenv
import tikzplotlib

# Local libraries
import piezo_fem as pfem


def plot_impedance_with_opencfs(
        sim: pfem.PiezoSimulation,
        open_cfs_hist_file: str):
    """Calculates and plots the impedence curves of the given FEM simulation
    together with OpenCFS results.

    Parameters:
        sim: Simulation object where the simulation was already done.
        open_cfs_hist_file: Hist file cotaining the charge from opencfs
            simulation.
    """
    frequencies_fem, impedance_fem = pfem.calculate_impedance(
        sim.solver.q, sim.excitation, sim.simulation_data.delta_t)

    # Get OpenCFS data
    _, charge_cfs = pfem.parse_charge_hist_file(open_cfs_hist_file)
    frequencies_cfs, impedance_cfs = pfem.calculate_impedance(
        charge_cfs, sim.excitation[:len(charge_cfs)], sim.simulation_data.delta_t)

    plt.rcParams.update({'font.size': 18})

    # Plot FEM and OpenCfs
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies_fem, np.abs(impedance_fem), label="Piezo FEM")
    plt.plot(frequencies_cfs, np.abs(impedance_cfs), "--", label="OpenCFS")
    #plt.plot(frequencies_fem, np.angle(impedance_fem), label="Phase piezo fem")
    plt.xlabel("Frequenz f / Hz")
    plt.ylabel("Impedanz |Z| / $\\Omega$")
    plt.yscale("log")
    #plt.xlim(0, 0.8e7)
    #plt.ylim(20, 2*1e4)
    plt.legend()
    plt.grid()
    plt.show()

    plot_folder = os.environ["piezo_fem_plot_path"]
    #plt.savefig(
    #    os.path.join(
    #        plot_folder,
    #        "compare_impedance.png"
    #    ),
    #    bbox_inches='tight'
    #)
    #tikzplotlib.save(os.path.join(
    #    plot_folder,
    #    "compare_impedance.tex"
    #))


if __name__ == "__main__":
    load_dotenv()
    MODEL_NAME = "temp_dep_mat_sim_16k"
    CWD = os.path.join(
        os.environ["piezo_fem_simulation_path"],
        MODEL_NAME
    )
    OPENCFS_FILE = os.path.join(CWD, "cfs_charge.hist")
    fem_sim = pfem.PiezoSimulation.load_simulation_settings(
        os.path.join(CWD, f"{MODEL_NAME}.cfg")
    )
    fem_sim.load_simulation_results()

    plot_impedance_with_opencfs(fem_sim, OPENCFS_FILE)
