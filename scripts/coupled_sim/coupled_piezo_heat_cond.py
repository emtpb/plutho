
# Python standard libraries
import os

# Third party libraries
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt

# Local libraries
import piezo_fem as pfem


def run_simulation(working_directory, sim_name, mesh):

    piezo_delta_t = 1e-8
    piezo_number_of_time_steps = 20000

    heat_cond_delta_t = 0.001
    heat_cond_number_of_time_steps = 1000

    coupled_sim = pfem.CoupledThermPiezoHeatCond(
        working_directory,
        sim_name,
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
        amplitude*np.sin(2*np.pi*frequency*time_values)
    )

    coupled_sim.simulate()
    coupled_sim.save_simulation_results()


if __name__ == "__main__":
    load_dotenv()
    SIM_NAME = "coupled_therm_piezo_heat_cond"
    CWD = os.path.join(
        os.environ["piezo_fem_simulation_path"],
    )

    run_simulation(
        CWD,
        SIM_NAME,
        pfem.Mesh(
            os.path.join(
                os.environ["piezo_fem_simulation_path"],
                "disc_mesh_0DOT0001.msh"
            ),
            True
        )
    )

    
