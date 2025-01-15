
# Python standard libraries
import os

# Third party libraries
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import json

# Local libraries
import piezo_fem as pfem


def run_simulation(
        working_directory: str,
        sim_name: str,
        mesh: pfem.Mesh,
        material_data: pfem.MaterialData):

    frequency = 2e6
    amplitude = 20
    heat_cond_delta_t = 0.0005
    heat_cond_number_of_time_steps = 2000

    coupled_sim = pfem.CoupledFreqPiezoHeatCond(
        working_directory,
        sim_name,
        frequency,
        pfem.SimulationData(
            heat_cond_delta_t,
            heat_cond_number_of_time_steps,
            0.5,
            None
        ),
        mesh
    )

    coupled_sim.add_material(
        "pic181",
        material_data,
        None
    )

    coupled_sim.set_excitation(amplitude)

    coupled_sim.simulate_coupled(
        starting_temperature=20,
        is_temperature_dependent=True)
    coupled_sim.save_simulation_results()


if __name__ == "__main__":
    load_dotenv()
    SIM_NAME = "coupled_piezo_freq_heat_cond"
    CWD = os.path.join(
        os.environ["piezo_fem_simulation_path"],
    )

    material_data_folder = os.getenv("material_data_path")
    if material_data_folder is None:
        print("Couldn't find material data path.")
        exit(1)

    temp_dep_material_data_path = os.path.join(
        material_data_folder,
        "pic181_time_dependent.json"
    )

    temp_dep_mat_data = None
    with open(temp_dep_material_data_path, "r", encoding="UTF-8") as fd:
        temp_dep_mat_data = pfem.MaterialData.from_dict(json.load(fd))

    if temp_dep_mat_data is None:
        raise IOError("Couldn't deserialize given material data.")

    run_simulation(
        CWD,
        SIM_NAME,
        pfem.Mesh(
            os.path.join(
                os.environ["piezo_fem_simulation_path"],
                "disc_mesh_0DOT0001.msh"
            ),
            True
        ),
        temp_dep_mat_data
    )
