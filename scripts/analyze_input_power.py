
# Python standard libraries
import os
import numpy as np

# Third party libraries
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Local libraries
import piezo_fem as pfem

def plot_power(excitation, charge, delta_t):
    current = np.gradient(charge, delta_t)

    time_values = np.arange(len(excitation))*delta_t

    power = current * excitation
    running_avg = np.convolve(power, np.ones(50)/50, mode="valid")
    print("Total energy:", np.trapezoid(power, dx=delta_t), "Ws")
    plt.plot(time_values, current, label="Strom")
    plt.plot(time_values, excitation, label="Spannung")
    plt.plot(time_values, power, label="elektrische Leistung")
    plt.plot(time_values[:len(running_avg)], running_avg, label="Gleitender Mittelwert elektrische Leistung")
    plt.xlabel("Zeit in s")
    plt.ylabel("Leistung in W")
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":

    #MODEL_NAME = "real_model_20k_check_energy"
    MODEL_NAME = "real_model_20k_check_energy"
    load_dotenv()

    CWD = os.getenv("piezo_fem_simulation_path")
    if CWD is None:
        print("Couldn't find simulation path.")
        exit(1)

    SIM_FOLDER = os.path.join(CWD, MODEL_NAME)

    sim = pfem.Simulation.load_simulation_settings(
        os.path.join(SIM_FOLDER, f"{MODEL_NAME}.cfg")
    )
    sim.load_simulation_results()
    sim.create_post_processing_views()

    plot_power(sim.excitation, sim.solver.q, sim.simulation_data.delta_t)
