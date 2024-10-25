"""Script to plot the impedence of the FEM simulation and or an
openCFS simulation.
"""

# Python standard libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# Third party libraries
import tikzplotlib

# Local libraries
import piezo_fem as pfem


def plot_impedance_with_opencfs(
        sim: pfem.Simulation,
        sim_term: pfem.Simulation,
        open_cfs_hist_file: str):
    """Calculates and plots the impedence curves of the given FEM simulation
    together with OpenCFS results.

    Parameters:
        sim: Simulation object where the simulation was already done.
        open_cfs_hist_file: Hist file cotaining the charge from opencfs
            simulation.
    """
    frequencies_fem, impedence_fem = pfem.calculate_impedance(
        sim.solver.q, sim.excitation, sim.simulation_data.delta_t)
    frequencies_fem_therm, impedence_fem_therm = pfem.calculate_impedance(
        sim_term.solver.q, sim_term.excitation, sim_term.simulation_data.delta_t)

    # Get OpenCFS data
    _, charge_cfs = pfem.parse_charge_hist_file(open_cfs_hist_file)
    frequencies_cfs, impedence_cfs = pfem.calculate_impedance(
        charge_cfs, sim.excitation, sim.simulation_data.delta_t)

    plt.rcParams.update({'font.size': 18})

    # Plot FEM and OpenCfs
    plt.plot(frequencies_fem, np.abs(impedence_fem), label="Piezo FEM")
    plt.plot(frequencies_cfs, np.abs(impedence_cfs), "--", label="OpenCFS")
    plt.xlabel("Frequency f / Hz")
    plt.ylabel("Impedance |Z| / $\\Omega$")
    plt.yscale("log")
    plt.xlim(0, 0.8e7)
    plt.ylim(20, 2*1e4)
    plt.legend()
    plt.grid()
    plt.show()
    #plt.savefig("/home/jonash/uni/Masterarbeit/plots/compare_impedance.png", bbox_inches='tight')

    tikzplotlib.save("/home/jonash/uni/Masterarbeit/plots/compare_impedance.tex")


if __name__ == "__main__":
    OPENCFS_FILE = "examples/cfs/charge.hist"
    MODEL_NAME = "real_model_impedence"
    THERM_MODEL_NAME = "real_model_impedence_thermal"
    WD = os.path.join(
        "/home/jonash/uni/Masterarbeit/simulations/", MODEL_NAME)
    WD_THERM = os.path.join(
        "/home/jonash/uni/Masterarbeit/simulations/", THERM_MODEL_NAME)
    fem_sim = pfem.Simulation.load_simulation_settings(
        os.path.join(WD, f"{MODEL_NAME}.cfg")
    )
    fem_sim.solver.q = np.load(os.path.join(WD, f"{MODEL_NAME}_q.npy"))

    fem_sim_therm = pfem.Simulation.load_simulation_settings(
        os.path.join(WD_THERM, f"{THERM_MODEL_NAME}.cfg")
    )
    fem_sim_therm.solver.q = np.load(
        os.path.join(WD_THERM, f"{THERM_MODEL_NAME}_q.npy"))

    # plot_impedence(fem_sim)
    plot_impedance_with_opencfs(fem_sim, fem_sim_therm, OPENCFS_FILE)
