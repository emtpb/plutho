
# Python standard libraries
import os
import numpy as np

# Third party libraries
from dotenv import load_dotenv
import yaml

# Local libraries
import piezo_fem as pfem

def load_temperature_dependent_material_data_pic181(files):
    """From the given file paths load the material data.
    Extract the temperatures from the file names.
    Expects that the files are already sorted after the temperature.

    Parameters:
        files: List of file paths.
    
    Returns:
        MaterialData object.
    """
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
        if ext != ".yaml":
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
        temperatures.append(float(temperature))

    mat_data = pfem.MaterialData(
        **{
            "c11": np.array(c11),
            "c12": np.array(c12),
            "c13": np.array(c13),
            "c33": np.array(c33),
            "c44": np.array(c44),
            "e15": np.array(e15),
            "e31": np.array(e31),
            "e33": np.array(e33),
            "eps11": np.array(eps11),
            "eps33": np.array(eps33),
            #"alpha_m": alpha_m,
            "alpha_m": 0,
            "alpha_k": 1.289813815258054e-10,
            "temperatures": np.array(temperatures),
            "heat_capacity": heat_capacity,
            "density": density,
            "thermal_conductivity": thermal_conductivity
        }
    )

    return mat_data


def simulation_model(base_directory, temp_dep_material_data):
    """Create a simulation model with the given temperature
    dependent material data.

    Parameters:
        base_directory: Directory where the simulation directory is
            created
        temp_dep_material_data: Temperature dependent material data.
    """
    sim_name = "temp_dep_mat_sim_130k"
    sim_directory = os.path.join(base_directory, sim_name)
    starting_temperature = 25
    sim = pfem.PiezoSimulation(
        sim_directory,
        sim_name
    )
    sim.create_ring_mesh(
        inner_radius=0.0026,
        outer_radius=0.00635,
        height=0.001,
        mesh_size=0.0001
    )
    sim.set_material_data(
        material_name="pic181_temp",
        material_data=temp_dep_material_data,
        starting_temperature=starting_temperature
    )
    # Set the starting theta field to starting temperature too
    sim.set_simulation(
        delta_t=1e-8,
        number_of_time_steps=130000,
        gamma=0.5,
        beta=0.25,
        simulation_type=pfem.SimulationType.THERMOPIEZOELECTRIC
    )
    sim.set_triangle_pulse_excitation(1)
    sim.set_boundary_conditions()

    sim.save_simulation_settings(
        "Simulation with temperature dependent material properties."
        "Alpha_k is constant 1.289813815258054e-10. Alpha_m is constant 0."
    )
    sim.simulate(
        np.ones(len(sim.mesh_data.nodes))*starting_temperature
    )

    sim.save_simulation_results()


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
    print(CWD)
    print(td_dir)
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

    simulation_model(CWD, material_data)
