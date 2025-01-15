
# Python standard libraries
import os

# Third party libraries
from dotenv import load_dotenv
import numpy as np
import json

# Local libraries
import piezo_fem as pfem


def run_simulation(
        working_directory,
        simulation_name,
        mesh):
    # piezo_frequency = 2.2022e6  # Fine mesh 0.00004
    piezo_frequency = 2.2322e6   # Coarse mesh 0.0001
    amplitude = 20

    heat_cond_time_steps = 1000
    heat_cond_delta_t = 0.001

    coupled_sim = pfem.CoupledFreqPiezoHeatCond(
        working_directory,
        simulation_name,
        piezo_frequency,
        pfem.SimulationData(
            heat_cond_delta_t,
            heat_cond_time_steps,
            0.5,
            None
        ),
        mesh
    )
    # Load temperature dependent material data
    material_data_folder = os.getenv("material_data_path")
    if material_data_folder is None:
        print("Couldn't find material data path.")
        exit(1)

    temp_dep_material_data_path = os.path.join(
        material_data_folder,
        "pic181_temperature_dependent.json"
    )

    temp_dep_mat_data = None
    with open(temp_dep_material_data_path, "r", encoding="UTF-8") as fd:
        temp_dep_mat_data = pfem.MaterialData.from_dict(
            json.load(fd)
        )

    if temp_dep_mat_data is None:
        raise IOError("Couldn't deserialize given material data.")

    coupled_sim.add_material(
        "pic181_20°C",
        temp_dep_mat_data,
        None
    )

    coupled_sim.set_excitation(
        excitation=amplitude,
        is_disc=False
    )

    coupled_sim.simulate_coupled(
        starting_temperature=20,
        is_temperature_dependent=False
    )

    coupled_sim.save_simulation_results()


if __name__ == "__main__":
    load_dotenv()
    SIM_NAME = "coupled_pic181_thermo_freq_temp_independent_20c_coarse"
    CWD = os.path.join(
        os.environ["piezo_fem_simulation_path"],
    )

    mesh = pfem.Mesh(
        os.path.join(
            os.environ["piezo_fem_simulation_path"],
            "ring_mesh_0DOT0001.msh"
        ),
        True
    )

    run_simulation(
        CWD,
        SIM_NAME,
        mesh
    )
