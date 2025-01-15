
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
    MODEL_NAME = "nonlinear_time_test"
    CWD = os.path.join(
        os.environ["piezo_fem_simulation_path"],
        MODEL_NAME
    )
    OPENCFS_FILE = os.path.join(CWD, "cfs_charge.hist")
    CHARGE_FILE = os.path.join(CWD, "q.npy")
    MEAS_FILE = os.path.join(CWD, "pic255_measurement.npy")

    NUMBER_OF_TIME_STEPS = 8192
    DELTA_T = 1e-8
    frequencies_fem = np.linspace(0, 1e7, 1000)
    excitation = np.zeros(NUMBER_OF_TIME_STEPS)
    excitation[1:10] = (
        1 * np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    )

    # Get meas data
    #meas = np.load(MEAS_FILE)
    #frequencies_meas = meas[0]
    #impedance_meas = meas[1]

    # Get OpenCFS data
    #_, charge_cfs = pfem.parse_charge_hist_file(OPENCFS_FILE)
    #frequencies_cfs, impedance_cfs = pfem.calculate_impedance(
    #    charge_cfs, excitation[:len(charge_cfs)], DELTA_T)

    # Impedance python fem freq
    q = np.load(CHARGE_FILE)
    print(q)
    frequencies_fem, impedance_fem = pfem.calculate_impedance(
        q,
        excitation,
        DELTA_T
    )

    plt.figure(figsize=(12, 6))
    plt.plot(
        (frequencies_fem/1e6),
        np.abs(impedance_fem),
        label="Python FEM"
    )
    #plt.plot(
    #    (frequencies_cfs/1e6),
    #    np.abs(impedance_cfs),
    #    "--",
    #    label="OpenCFS"
    #)
    #plt.plot(
    #    (frequencies_meas/1e6),
    #    np.abs(impedance_meas),
    #    "--",
    #    label="Messung PIC255"
    #)
    plt.xlabel("Frequenz $f$ / MHz")
    plt.ylabel("Impedanz $|Z|$ / $\\Omega$")
    plt.yscale("log")
    #plt.xlim(0, 1e8)
    #plt.ylim(15, 2*1e4)
    plt.legend()
    plt.grid()
    plt.show()
