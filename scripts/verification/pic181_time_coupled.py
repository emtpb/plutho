
# Python standard libraries
import os

# Third party libraries
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import json

# Local libraries
import piezo_fem as pfem


def run_simulation(working_directory, sim_name, mesh):
    piezo_delta_t = 1e-8
    piezo_number_of_time_steps = 60000

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

    # Load temperature dependent materials
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

    amplitude = 20
    frequency = 2.2322e6
    time_values = np.arange(piezo_number_of_time_steps)*piezo_delta_t
    coupled_sim.set_excitation(
        excitation=amplitude*np.sin(2*np.pi*frequency*time_values),
        is_disc=False
    )

    coupled_sim.piezo_sim.material_manager.is_temperature_dependent = False
    coupled_sim.heat_cond_sim.material_manager.is_temperature_dependent = False

    coupled_sim.simulate(
        starting_temperature=20
    )
    coupled_sim.save_simulation_results()


if __name__ == "__main__":
    load_dotenv()
    SIM_NAME = "coupled_pic181_thermo_time_60k_temp_independent_20c"
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
    #mesh.generate_rectangular_mesh(
    #    width=0.00635-0.0026,
    #    height=0.001,
    #    mesh_size=0.0001,
    #    x_offset=0.0026
    #)

    run_simulation(
        CWD,
        SIM_NAME,
        mesh
    )
