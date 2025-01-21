
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

    CWD = os.getenv("piezo_fem_simulation_path")
    if CWD is None:
        print("Couldn't find simulation path")
        exit(1)

    sim_name_20 = "pic181_impedance_20c"
    sim_name_90 = "pic181_impedance_90c"

    mesh = pfem.Mesh(
        os.path.join(CWD, "ring_mesh_0DOT00004.msh"),
        True
    )
    
    sim_20_working_directory = os.path.join(CWD, sim_name_20)
    sim_90_working_directory = os.path.join(CWD, sim_name_90)

    sim_20 = pfem.SingleSimulation.load_simulation_settings(
        sim_20_working_directory
    )
    sim_20.load_simulation_results()
    sim_90 = pfem.SingleSimulation.load_simulation_settings(
        sim_90_working_directory
    )
    sim_90.load_simulation_results()

    impedance_20 = np.abs(
        1/(1j*2*np.pi*sim_20.solver.frequencies*sim_20.solver.q)
    )
    impedance_90 = np.abs(
        1/(1j*2*np.pi*sim_90.solver.frequencies*sim_90.solver.q)
    )

    plt.plot(
        sim_20.solver.frequencies/1e6,
        impedance_20,
        label="PIC181 $20\\,\\mathrm{°C}$"
    )
    plt.plot(
        sim_90.solver.frequencies/1e6,
        impedance_90,
        label="PIC181 $90\\,\\mathrm{°C}$"
    )
    plt.xlabel("Frequenz $f$ / MHz")
    plt.ylabel("Impedanz $|Z|$ / $\\Omega$")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()

