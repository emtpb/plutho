
# Python standard libraries
import os

# Third party libraries
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt

# Local libraries
import piezo_fem as pfem


def run_simulation(
        working_directory,
        simulation_name,
        mesh,
        delta_t,
        number_of_time_steps,
        excitation):
    sim = pfem.SingleSimulation(
        working_directory,
        simulation_name,
        mesh
    )

    sim.setup_piezo_time_domain(
        pfem.SimulationData(
            delta_t,
            number_of_time_steps,
            0.5,
            0.25
        )
    )

    sim.add_material(
        "pic181_c",
        pfem.materials.pic181_20_extrapolated,
        None
    )

    sim.add_dirichlet_bc(
        pfem.FieldType.PHI,
        "Electrode",
        excitation
    )
    sim.add_dirichlet_bc(
        pfem.FieldType.PHI,
        "Ground",
        np.zeros(number_of_time_steps)
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

    sim_name = "pic181_impedance_time_40k"
    mesh = pfem.Mesh(
        os.path.join(CWD, "ring_mesh_0DOT00004.msh"),
        True
    )
    working_directory = os.path.join(CWD, sim_name)

    delta_t = 1e-8
    number_of_time_steps = 40000

    # Add boundary conditions
    excitation = np.zeros(number_of_time_steps)
    excitation[1:10] = (
        1 * np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    )

    if False:
        run_simulation(
            CWD,
            sim_name,
            mesh,
            delta_t,
            number_of_time_steps,
            excitation
        )

    if True:
        # Analyze results
        sim = pfem.SingleSimulation.load_simulation_settings(working_directory)
        sim.load_simulation_results()

        frequencies_fem, impedance_fem = pfem.calculate_impedance(
            sim.solver.q,
            excitation,
            delta_t
        )

        q_fd = np.load(os.path.join(working_directory, "freq_q.npy"))
        frequencies_fd = np.linspace(0, 1e7, 1000)[1:]
        impedance_fd = np.abs(1/(1j*2*np.pi*frequencies_fd*q_fd))

        plt.plot(
            frequencies_fem/1e6,
            np.abs(impedance_fem),
            label="Impedanz PIC181 Zeitbereichssimulation"
        )
        plt.plot(
        frequencies_fd/1e6,
        np.abs(impedance_fd),
        label="Impedanz PIC181 Frequenzbereichssimulation"
        )

        plt.xlabel("Frequenz $f$ / MHz")
        plt.ylabel("Impedanz $|Z|$ / $\\Omega$")
        plt.yscale("log")
        plt.legend()
        plt.grid()
        plt.show()

