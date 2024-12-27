"""Script to plot the impedence of the FEM simulation and or an
openCFS simulation.
"""

# Python standard libraries
import os
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

# Third party libraries
from dotenv import load_dotenv
import tikzplotlib

# Local libraries
import piezo_fem as pfem


def plot_impedance_with_opencfs(
        charge: npt.NDArray,
        open_cfs_hist_file: str,
        excitation: npt.NDArray,
        delta_t: float):
    """Calculates and plots the impedence curves of the given FEM simulation
    together with OpenCFS results.

    Parameters:
        sim: Simulation object where the simulation was already done.
        open_cfs_hist_file: Hist file cotaining the charge from opencfs
            simulation.
    """
    frequencies_fem, impedance_fem = pfem.calculate_impedance(
        charge,
        excitation,
        delta_t
    )

    # Get OpenCFS data
    _, charge_cfs = pfem.parse_charge_hist_file(open_cfs_hist_file)
    frequencies_cfs, impedance_cfs = pfem.calculate_impedance(
        charge_cfs, excitation[:len(charge_cfs)], delta_t)

    plt.rcParams.update({'font.size': 18})

    max_size=1000

    # Plot FEM and OpenCfs
    plt.figure(figsize=(12, 6))
    plt.plot((frequencies_fem/1e6)[:max_size], np.abs(impedance_fem)[:max_size], label="Python FEM")
    plt.plot((frequencies_cfs/1e6)[:max_size], np.abs(impedance_cfs)[:max_size], "--", label="OpenCFS")
    #plt.plot(frequencies_fem, np.angle(impedance_fem), label="Phase piezo fem")
    plt.xlabel("Frequenz $f$ / MHz")
    plt.ylabel("Impedanz $|Z|$ / $\\Omega$")
    plt.yscale("log")
    plt.xlim(0, 1e8)
    plt.ylim(15, 2*1e4)
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
    #tikzplotlib.save(
    #    os.path.join(
    #        plot_folder,
    #        "default_impedance_comparison.tex"
    #    ),
    #    axis_width='12cm'
    #)


if __name__ == "__main__":
    load_dotenv()
    MODEL_NAME = "impedance_pic255_thesis_fine_mesh"
    CWD = os.path.join(
        os.environ["piezo_fem_simulation_path"],
        MODEL_NAME
    )
    OPENCFS_FILE = os.path.join(CWD, "cfs_charge.hist")

    NUMBER_OF_TIME_STEPS = 8192
    DELTA_T = 1e-8
    excitation = np.zeros(NUMBER_OF_TIME_STEPS)
    excitation[1:10] = (
        1 * np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    )

    #pfem.io.create_vector_field_as_csv(
    #    fem_sim.solver.u,
    #    fem_sim.mesh_data.nodes,
    #    os.path.join(CWD, "csv_data"),
    #    False
    #)

    plot_impedance_with_opencfs(
        np.load(os.path.join(CWD, "q.npy")),
        OPENCFS_FILE,
        excitation,
        DELTA_T
    )
