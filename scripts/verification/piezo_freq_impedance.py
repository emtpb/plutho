
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
    MODEL_NAME = "piezo_sim_freq_test_new_model"
    CWD = os.path.join(
        os.environ["piezo_fem_simulation_path"],
        MODEL_NAME
    )
    OPENCFS_FILE = os.path.join(CWD, "cfs_charge.hist")
    sim = pfem.SingleSimulation.load_simulation_settings(CWD)
    sim.load_simulation_results()

    NUMBER_OF_TIME_STEPS_CFS = 8192
    DELTA_T_CFS = 1e-8
    excitation = np.zeros(NUMBER_OF_TIME_STEPS_CFS)
    excitation[1:10] = (
        1 * np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    )

    # Get OpenCFS data
    _, charge_cfs = pfem.parse_charge_hist_file(OPENCFS_FILE)
    frequencies_cfs, impedance_cfs = pfem.calculate_impedance(
        charge_cfs, excitation[:len(charge_cfs)], DELTA_T_CFS)

    # Impedance python fem freq
    impedance_fem = np.abs(1/(1j*2*np.pi*sim.solver.frequencies*sim.solver.q))

    plt.figure(figsize=(12, 6))
    plt.plot(
        (sim.solver.frequencies/1e6),
        impedance_fem,
        label="Python FEM Freq"
    )
    plt.plot(
        (frequencies_cfs/1e6),
        np.abs(impedance_cfs),
        "--",
        label="OpenCFS"
    )
    plt.xlabel("Frequenz $f$ / MHz")
    plt.ylabel("Impedanz $|Z|$ / $\\Omega$")
    plt.yscale("log")
    plt.xlim(0, 8)
    plt.ylim(10, 2*1e4)
    plt.legend()
    plt.grid()
    plt.show()
