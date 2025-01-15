
# Python standard libraries
import os

# Third party libraries
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt

# Local libraries
import piezo_fem as pfem


if __name__ == "__main__":
    load_dotenv()
    MODEL_NAME = "impedance_pic181_20c_coarse"
    CWD = os.path.join(
        os.environ["piezo_fem_simulation_path"],
        MODEL_NAME
    )
    plot_folder = os.environ["piezo_fem_plot_path"]
    TIME_SIM_RESULTS = os.path.join(CWD, "time_results_q.npy")
    sim = pfem.SingleSimulation.load_simulation_settings(CWD)
    sim.load_simulation_results()

    NUMBER_OF_TIME_STEPS = 8192
    DELTA_T = 1e-8
    excitation = np.zeros(NUMBER_OF_TIME_STEPS)
    excitation[1:10] = (
        1 * np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    )

    # Get Time data
    # charge_time = np.load(TIME_SIM_RESULTS)
    # frequencies_time, impedance_time = pfem.calculate_impedance(
    #     charge_time,
    #     excitation[:len(charge_time)],
    #     DELTA_T)

    # Impedance python fem freq
    impedance_fem = np.abs(1/(1j*2*np.pi*sim.solver.frequencies*sim.solver.q))
    plt.figure(figsize=(12, 6))
    plt.plot(
        (sim.solver.frequencies/1e6)[:1000],
        impedance_fem[:1000],
        label="Frequenzsimulation"
    )
    #plt.plot(
    #    (frequencies_time/1e6)[:1000],
    #    np.abs(impedance_time)[:1000],
    #    "--",
    #    label="Zeitbereichssimulation"
    #)
    plt.xlabel("Frequenz $f$ / MHz")
    plt.ylabel("Impedanz $|Z|$ / $\\Omega$")
    plt.yscale("log")
    # plt.xlim(0, 8)
    # plt.ylim(10, 3*1e4)
    plt.legend()
    plt.grid()
    plt.show()

    #import tikzplotlib
    #    tikzplotlib.save(os.path.join(plot_folder, "compare_time_freq.tex"))

