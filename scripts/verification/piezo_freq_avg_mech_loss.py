
# Python standard libraries
import os

# Third party libraries
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt

# Local libraries
import piezo_fem as pfem


def run_simulation(working_directory, sim_name):

    mesh = pfem.Mesh(
        os.path.join(
            os.environ["piezo_fem_simulation_path"],
            "disc_mesh_0DOT0001.msh"
        ),
        True
    )

    amplitude = 20
    frequency = 2e6

    sim = pfem.SingleSimulation(
        working_directory,
        sim_name,
        mesh
    )
    sim.setup_piezo_freq_domain([frequency])
    sim.add_material(
        "pic255",
        pfem.pic255,
        None
    )
    sim.add_dirichlet_bc(
        pfem.FieldType.PHI,
        "Electrode",
        np.array([amplitude])
    )
    sim.add_dirichlet_bc(
        pfem.FieldType.PHI,
        "Ground",
        np.zeros(1)
    )
    sim.add_dirichlet_bc(
        pfem.FieldType.U_R,
        "Symaxis",
        np.zeros(1)
    )

    sim.simulate(calculate_mech_loss=True)
    sim.save_simulation_results()


if __name__ == "__main__":
    load_dotenv()
    SIM_NAME = "check_piezo_freq_avg_losses"
    CWD = os.path.join(
        os.environ["piezo_fem_simulation_path"],
    )
    run_simulation(CWD, SIM_NAME)

    # Load simulation results
    mech_loss_time = np.load(os.path.join(CWD, SIM_NAME, "mech_loss_time.npy"))
    mech_loss_freq = np.load(os.path.join(CWD, SIM_NAME, "mech_loss.npy"))

    print(np.mean(mech_loss_time[0, -100:]))
    print(np.real(mech_loss_freq[0, -1]))

    plt.plot(np.mean(mech_loss_time[:, -100:], axis=1), "+", label="time")
    plt.plot(np.real(mech_loss_freq[:, -1]), "+", label="freq")
    plt.grid()
    plt.legend()
    plt.show()
