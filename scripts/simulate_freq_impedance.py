
# Python standard libraries
import os
import numpy as np

# Third party libraries
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Local libraries
import piezo_fem as pfem


def simulate_freq(sim_directory, sim_name, sim_frequencies):
    sim = pfem.PiezoSimulation(
        sim_directory,
        sim_name)
    sim.create_disc_mesh(0.005, 0.001, 0.0001)
    sim.set_material_data(
        "pic255",
        pfem.pic255,
    )
    sim.material_manager.material_data.alpha_m = 1.267e5
    sim.set_frequency_simulation()
    sim.set_sinusoidal_excitation(
        np.ones(len(sim_frequencies)),
        sim_frequencies
    )
    sim.set_boundary_conditions()
    sim.save_simulation_settings(
        "Frequency simulation to check impedance curve.")

    sim.simulate()
    sim.save_simulation_results()


if __name__ == "__main__":
    load_dotenv()

    CWD = os.getenv("piezo_fem_simulation_path")
    if CWD is None:
        print("Couldn't find simulation path.")
        exit(1)

    frequencies = np.linspace(0, 1e7, 1000)[1:]

    # Simulate freq
    SIM_NAME = "frequency_sim_disc"
    sim_directory = os.path.join(CWD, SIM_NAME)
    # simulate_freq(sim_directory, SIM_NAME, frequencies)

    # Load freq results
    freq_sim = pfem.PiezoSimulation.load_simulation_settings(
        os.path.join(sim_directory, f"{SIM_NAME}.cfg")
    )
    freq_sim.load_simulation_results()
    impedance = np.abs(1/(1j*2*np.pi*frequencies*freq_sim.solver.q))

    # Load example time results
    time_sim = pfem.PiezoSimulation.load_simulation_settings(
        os.path.join(CWD, "check_impedance_test", "check_impedance_test.cfg")
    )
    time_sim.load_simulation_results()
    frequencies_time, impedance_time = pfem.calculate_impedance(
        time_sim.solver.q,
        time_sim.excitation,
        time_sim.simulation_data.delta_t)

    # Plot
    plt.plot(frequencies, impedance, label="MyFEM freq")
    plt.plot(frequencies_time, np.abs(impedance_time), "--", label="MyFEM time")
    plt.xlabel("Frequenz f / Hz")
    plt.ylabel("Impedanz |Z| / $\\Omega$")
    plt.legend()
    plt.yscale("log")
    plt.grid()
    plt.show()
