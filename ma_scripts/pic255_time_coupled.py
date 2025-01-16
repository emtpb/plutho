
# Python standard libraries
import os

# Third party libraries
from dotenv import load_dotenv
import numpy as np

# Local libraries
import piezo_fem as pfem


def run_simulation(working_directory, simulation_name, mesh):
    piezo_delta_t = 1e-8
    piezo_number_of_time_steps = 20000

    heat_cond_delta_t = 0.001
    heat_cond_number_of_time_steps = 1000

    coupled_sim = pfem.CoupledThermPiezoHeatCond(
        working_directory,
        simulation_name,
        pfem.SimulationData(
            piezo_delta_t,
            piezo_number_of_time_steps,
            0.5,
            0.25
        ),
        pfem.SimulationData(
            heat_cond_delta_t,
            heat_cond_number_of_time_steps,
            0.5,
            None
        ),
        mesh
    )

    coupled_sim.add_material(
        "pic255",
        pfem.pic255,
        None
    )

    amplitude = 20
    frequency = 2e6
    time_values = np.arange(piezo_number_of_time_steps)*piezo_delta_t
    coupled_sim.set_excitation(
        excitation=amplitude*np.sin(2*np.pi*frequency*time_values),
        is_disc=True
    )

    coupled_sim.simulate()
    # coupled_sim.save_simulation_results()
    # coupled_sim.save_simulation_settings()


if __name__ == "__main__":
    load_dotenv()
    SIM_NAME = "pic255_thermo_time_20k"
    CWD = os.path.join(
        os.environ["piezo_fem_simulation_path"]
    )

    mesh = pfem.Mesh(
        os.path.join(
            os.environ["piezo_fem_simulation_path"],
            "disc_mesh_0DOT00004.msh"
        ),
        True
    )

    run_simulation(
        CWD,
        SIM_NAME,
        mesh
    )

