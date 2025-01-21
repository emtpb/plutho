
# Python standard libraries
import os

# Third party libraries
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt

# Local libraries
import piezo_fem as pfem


def run_simulation(working_directory, simulation_name, mesh):
    sim = pfem.SingleSimulation(
        working_directory,
        simulation_name,
        mesh
    )

    frequencies = np.linspace(0, 1e7, 8000)[1:]
    sim.setup_piezo_freq_domain(frequencies)

    sim.add_material(
        "pic181_c",
        pfem.materials.pic181_90_extrapolated,
        None
    )

    sim.add_dirichlet_bc(
        pfem.FieldType.PHI,
        "Electrode",
        np.ones(len(frequencies))
    )
    sim.add_dirichlet_bc(
        pfem.FieldType.PHI,
        "Ground",
        np.zeros(len(frequencies))
    )

    electrode_elements = mesh.get_elements_by_physical_groups(
        ["Electrode"]
    )["Electrode"]

    sim.simulate(
        electrode_elements=electrode_elements
    )
    sim.save_simulation_results()
    sim.save_simulation_settings()


if __name__ == "__main__":
    load_dotenv()

    CWD = os.getenv("piezo_fem_simulation_path")
    if CWD is None:
        print("Couldn't find simulation path.")
        exit(1)

    sim_name = "pic181_impedance_90c"
    mesh = pfem.Mesh(
        os.path.join(CWD, "ring_mesh_0DOT00004.msh"),
        True
    )
    run_simulation(CWD, sim_name, mesh)
    exit(0)
    working_directory = os.path.join(CWD, sim_name)
    # Analyze results
    sim = pfem.SingleSimulation.load_simulation_settings(working_directory)
    sim.load_simulation_results()

    impedance_fem = np.abs(1/(1j*2*np.pi*sim.solver.frequencies*sim.solver.q))
    plt.plot(
        sim.solver.frequencies/1e6,
        impedance_fem,
        label="Impedanz PIC181"
    )

    plt.xlabel("Frequenz $f$ / MHz")
    plt.ylabel("Impedanz $|Z|$ / $\\Omega$")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()

