
# Python standard libraries
import os
import numpy as np

# Third party libraries
from dotenv import load_dotenv
import yaml

# Local libraries
from piezo_fem import MaterialData

def load_temperature_dependent_material_data_pic181(files):
    c11 = []
    c12 = []
    c13 = []
    c33 = []
    c44 = []
    e15 = []
    e31 = []
    e33 = []
    eps11 = []
    eps33 = []
    alpha_m = []
    alpha_k = []
    temperatures = []
    density = 7850
    thermal_conductivity = 1.1
    heat_capacity = 350

    material_name = "pic181"

    for file in files:
        file_name, ext = os.path.splitext(os.path.basename(file))
        if ext != "yaml":
            raise IOError(f"*.yaml file is expected but {ext} is given.")

        _, _, _, temperature = file_name.split("_")

        data = None
        with open(file, "r", encoding="UTF-8") as fd:
            data = yaml.safe_load(fd)

        c11.append(data["c11"])
        c12.append(data["c12"])
        c13.append(data["c13"])
        c33.append(data["c33"])
        c44.append(data["c44"])
        e15.append(data["e15"])
        e31.append(data["e31"])
        e33.append(data["e33"])
        eps11.append(data["eps11"])
        eps33.append(data["eps33"])
        alpha_m.append(data["alpha_M"])
        alpha_k.append(data["alpha_K"])
        temperatures.append(temperature)

    mat_data = MaterialData(
        material_name,
        {
            "c11": c11,
            "c12": c12,
            "c13": c13,
            "c33": c33,
            "c44": c44,
            "e15": e15,
            "e31": e31,
            "e33": e33,
            "eps11": eps11,
            "eps33": eps33,
            #"alpha_m": alpha_m,
            "alpha_m": 0,
            "alpha_k": alpha_k,
            "temperatures": temperatures,
            "heat_capacity": heat_capacity,
            "density": density,
            "thermal_conductivity": thermal_conductivity
        }
    )

    return mat_data


if __name__ == "__main__":
    load_dotenv()

    CWD = os.getenv("piezo_fem_simulation_path")
    if CWD is None:
        print("Couldn't find simulation path.")
        exit(1)

    td_dir = os.path.join(
        CWD,
        "..",
        "pic181_time_dependent"
    )
    PIEZO_SIM_NAME = "pic181_td_mat_param"

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

    material_data = load_temperature_dependent_material_data_pic181(
        temp_file_names
    )
