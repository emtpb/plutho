
# Python standard libraries
import os

# Third party libraries
from dotenv import load_dotenv
import numpy as np

# Local libraries
import piezo_fem as pfem


def run_simulation(
        working_directory,
        simulation_name,
        mesh,
        mech_loss_density_file):
    delta_t = 0.001
    number_of_time_steps = 1000

    sim = pfem.SingleSimulation(working_directory, simulation_name, mesh)
    sim.add_material(
        "pic255",
        pfem.pic255,
        None
    )

    sim.setup_heat_conduction_time_domain(
        delta_t,
        number_of_time_steps,
        0.5
    )

    mech_loss_density = np.load(mech_loss_density_file)
    sim.solver.set_constant_volume_heat_source(
        np.mean(mech_loss_density[:, -100:], axis=1),
        number_of_time_steps
    )

    elements = mesh.get_elements_by_physical_groups(
        ["Electrode", "Ground", "RightBoundary"]
    )
    convective_boundary_elements = np.vstack(
        [
            elements["Electrode"],
            elements["Ground"],
            elements["RightBoundary"]
        ]
    )
    sim.solver.set_convection_bc(
        convective_boundary_elements,
        80,
        20
    )

    sim.simulate()
    sim.save_simulation_results()
    sim.save_simulation_settings()


if __name__ == "__main__":
    load_dotenv()
    SIM_NAME = "pic255_heat_cond_convective"
    CWD = os.path.join(os.environ["piezo_fem_simulation_path"])
    mech_loss_path = os.path.join(
        CWD,
        "pic255_thermo_time_20k",
        "thermo_piezo_mech_loss.npy"
    )

    mesh = pfem.Mesh(
        os.path.join(
            CWD,
            "disc_mesh_0DOT00004.msh"
        ),
        True
    )

    run_simulation(CWD, SIM_NAME, mesh, mech_loss_path)

