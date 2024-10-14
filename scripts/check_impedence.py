"""Script to plot the impedence of the FEM simulation and or an
openCFS simulation.
"""

# Python standard libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# Local libraries
import piezo_fem as pfem


def plot_impedance_with_opencfs(sim: pfem.Simulation, open_cfs_hist_file: str):
    """Calculates and plots the impedence curves of the given FEM simulation
    together with OpenCFS results.

    Parameters:
        sim: Simulation object where the simulation was already done.
        open_cfs_hist_file: Hist file cotaining the charge from opencfs
            simulation.
    """
    delta_t = sim.simulation_data.delta_t
    excitation = sim.excitation

    frequencies_fem, impedence_fem = pfem.calculate_impedance(
        sim.solver.q, excitation, delta_t)

    # Get OpenCFS data
    _, charge_cfs = pfem.parse_charge_hist_file(open_cfs_hist_file)
    frequencies_cfs, impedence_cfs = pfem.calculate_impedance(
        charge_cfs, excitation, delta_t)

    # Plot FEM and OpenCfs
    plt.plot(frequencies_fem, np.abs(impedence_fem), label="MyFEM")
    plt.plot(frequencies_cfs, np.abs(impedence_cfs), "+", label="OpenCFS")
    plt.xlabel("Frequency f / Hz")
    plt.ylabel("Impedence |Z| / $\\Omega$")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    OPENCFS_FILE = "examples/cfs/charge.hist"
    MODEL_NAME = "real_model_impedence"
    WD = os.path.join(
        "/home/jonash/uni/Masterarbeit/simulations/", MODEL_NAME)
    fem_sim = pfem.Simulation.load_simulation_settings(
        os.path.join(WD, f"{MODEL_NAME}.cfg")
    )
    fem_sim.solver.q = np.load(os.path.join(WD, f"{MODEL_NAME}_q.npy"))

    # plot_impedence(fem_sim)
    plot_impedance_with_opencfs(fem_sim, OPENCFS_FILE)