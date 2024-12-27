
# Python standard libraries
import os

# Third party libraries
from dotenv import load_dotenv
import numpy as np

# Local libraries
import piezo_fem as pfem

def load_material_data(td_mat_data_path):
    td_dir = os.path.join(td_mat_data_path, "pic181_time_dependent")

    # Load temperature dependent material parameters
    temp_file_names = [
        "opt_values_2_25.yaml",
        "opt_values_2_30.yaml",
        "opt_values_2_35.yaml",
        "opt_values_2_40.yaml",
        "opt_values_2_45.yaml",
        "opt_values_2_50.yaml",
        "opt_values_2_55.yaml",
        "opt_values_2_60.yaml",
        "opt_values_2_65.yaml",
        "opt_values_2_70.yaml",
        "opt_values_2_75.yaml",
        "opt_values_2_80.yaml",
        "opt_values_2_85.yaml",
    ]

    for index, temp_file_name in enumerate(temp_file_names):
        temp_file_names[index] = os.path.join(td_dir, temp_file_name)

    material_data = pfem.io.load_temperature_dependent_material_data_pic181(
        temp_file_names
    )

    return material_data


if __name__ == "__main__":
    load_dotenv()

    CWD = os.getenv("piezo_fem_simulation_path")
    if CWD is None:
        print("Couldn't find simulation path.")
        exit(1)
    TD_MATERIAL_DATA_PATH = os.getenv("time_dep_material_data_path")
    MODEL_NAME = "temp_dep_mat_sim_250k"
    fem_sim = pfem.PiezoSimulation.load_simulation_settings(
        os.path.join(CWD, MODEL_NAME, f"{MODEL_NAME}.cfg")
    )
    fem_sim.load_simulation_results()

    material_manager = pfem.materials.MaterialManager(
        "pic181",
        load_material_data(TD_MATERIAL_DATA_PATH),
        len(fem_sim.mesh_data.elements),
        25
    )

    number_of_nodes = len(fem_sim.mesh_data.nodes)

    theta_field = fem_sim.solver.u[3*number_of_nodes:, -1]
    avg_temperatures = pfem.simulation.piezo_temp_time. \
        get_avg_temp_field_per_element(
            theta_field,
            fem_sim.mesh_data.elements
        )
    print(theta_field)
    print("Max field value", np.max(theta_field))

    update = material_manager.update_temperature(
        avg_temperatures
    )
    print(fem_sim.solver.u.shape)

    _, number_of_time_steps = fem_sim.solver.u.shape
    SKIPPED_TIME_STEPS = 1000
    new_number_of_time_steps = number_of_time_steps//SKIPPED_TIME_STEPS
    fem_sim.gmsh_handler.create_u_default_post_processing_view(
        fem_sim.solver.u[:, ::SKIPPED_TIME_STEPS],
        new_number_of_time_steps,
        fem_sim.simulation_data.delta_t,
        True,
        False
    )
