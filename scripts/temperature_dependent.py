
# Python standard libraries
import os
import sys
from io import StringIO

# Third party libraries
from dotenv import load_dotenv
import numpy as np

# Local libraries
import piezo_fem as pfem


def simulation_model(base_directory, temp_dep_material_data):
    """Create a simulation model with the given temperature
    dependent material data.

    Parameters:
        base_directory: Directory where the simulation directory is
            created
        temp_dep_material_data: Temperature dependent material data.
    """

    result = StringIO()
    sys.stdout = result

    sim_name = "temp_dep_mat_sim_500k"
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
        delta_t=4e-8,
        number_of_time_steps=500000,
        gamma=0.5,
        beta=0.25,
        simulation_type=pfem.SimulationType.THERMOPIEZOELECTRIC
    )
    #sim.set_triangle_pulse_excitation(1)
    sim.set_sinusoidal_excitation(100, 2.14439e6)
    sim.set_boundary_conditions()

    sim.save_simulation_settings(
        "Simulation with temperature dependent material properties."
        "Alpha_k is constant 1.289813815258054e-10. Alpha_m is constant 0."
    )

    sim.simulate(
        np.ones(len(sim.mesh_data.nodes))*starting_temperature
    )

    sim.save_simulation_results()

    with open(
            os.path.join(sim_directory, "console_log.txt"),
            "w",
            encoding="UTF-8") as fd:
        fd.write(result.getvalue())

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
    td_dir = "/upb/departments/emt/Student/jonasho/Masterarbeit/pic181_time_dependent/"
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

    material_data = pfem.io.load_temperature_dependent_material_data_pic181(
        temp_file_names
    )

    simulation_model(CWD, material_data)
