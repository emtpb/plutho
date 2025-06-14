"""Module for io functions for the resulting fields and values."""

# Python standard libraries
from typing import Tuple
import os

# Third party libraries
import numpy as np
import numpy.typing as npt
import yaml

# Local libraries
from .simulation.base import MaterialData


def parse_charge_hist_file(file_path: str) -> Tuple[npt.NDArray, npt.NDArray]:
    """Reads the charge file from an OpenCFS simulation.

    Parameters:
        file_path: Path to the charge (*.hist) file.

    Returns:
        Tuple containing the time list and charge list read from the
        given file.
    """
    lines = []
    with open(file_path, "r", encoding="UTF-8") as fd:
        lines = fd.readlines()[3:]

    time = []
    charge = []
    for line in lines:
        current_time, current_charge = line.split()
        time.append(float(current_time))
        charge.append(float(current_charge))

    return np.array(time), np.array(charge)


def parse_charge_freq_hist_file(
    file_path: str
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Reads the charge file from an OpenCFS simulation.

    Parameters:
        file_path: Path to the charge (*.hist) file.

    Returns:
        Tuple containing the time list and charge list read from the
        given file.
    """
    lines = []
    with open(file_path, "r", encoding="UTF-8") as fd:
        lines = fd.readlines()[3:]

    frequencies = []
    charge = []
    for line in lines:
        current_frequency, current_charge, _ = line.split()
        frequencies.append(float(current_frequency))
        charge.append(float(current_charge))

    return np.array(frequencies), np.array(charge)


def parse_displacement_hist_file(
    file_path: str
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Reads the displacement file from and OpenCFS simulation.

    Parameters:
        file_path: Path to the displacement (*.hist) file.

    Returns:
        Tuple containg the time steps, u_r and u_z.
    """
    lines = []
    with open(file_path, "r", encoding="UTF-8") as fd:
        lines = fd.readlines()
    time_steps = []
    u_r = []
    u_z = []
    for _, line in enumerate(lines[3:]):
        current_time_step, current_u_r, current_u_z = line.split()
        time_steps.append(float(current_time_step))
        u_r.append(float(current_u_r))
        u_z.append(float(current_u_z))

    return np.array(time_steps), np.array(u_r), np.array(u_z)


def create_scalar_field_as_csv(
    field_name: str,
    field: npt.NDArray,
    nodes: npt.NDArray,
    folder_path: str
):
    """Creates a series of *.csv files in the given folder each for
    a specific time step of the given field.

    Parameters:
        field_name: Name of the field -> Name of the file.
        field: Field which shall be exported.
        nodes: Nodes of the mesh.
        folder_path: Path of the folder in which all the files are created.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    number_of_time_steps = field.shape[1]

    for time_step in range(number_of_time_steps):
        current_file_path = os.path.join(
            folder_path, f"{field_name}_{time_step}.csv")

        text = f"r,z,{field_name}\n"
        for node_index, node in enumerate(nodes):
            current_field = field[node_index, time_step]

            r = node[0]
            z = node[1]
            text += f"{r},{z},{current_field}\n"

        with open(current_file_path, "w", encoding="UTF-8") as fd:
            fd.write(text)


def create_vector_field_as_csv(
    u: npt.NDArray,
    nodes: npt.NDArray,
    folder_path: str,
    contains_theta: bool
):
    """Creates a new *.csv for each time step containing the data from u.

    Parameters:
        u: Contains the data (simulation output) u[time_index, node_index].
        nodes: List of nodes used in the simulation.
        folder_path: Path to the folder where the *.csv files are stored.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    number_of_nodes = len(nodes)
    number_of_time_steps = u.shape[1]

    for time_step in range(number_of_time_steps):
        current_file_path = os.path.join(folder_path, f"u_{time_step}.csv")

        if contains_theta:
            text = "r,z,u_r,u_z,v,theta\n"
        else:
            text = "r,z,u_r,u_z,v\n"
        for node_index, node in enumerate(nodes):
            current_u_r = u[2*node_index, time_step]
            current_u_z = u[2*node_index+1, time_step]
            current_v = u[2*number_of_nodes+node_index, time_step]
            if contains_theta:
                current_theta = u[3*number_of_nodes+node_index, time_step]
                theta = current_theta

            r = node[0]
            z = node[1]
            u_r = current_u_r
            u_z = current_u_z
            v = current_v
            if contains_theta:
                text += f"{r},{z},{u_r},{u_z},{v},{theta}\n"
            else:
                text += f"{r},{z},{u_r},{u_z},{v}\n"

        with open(current_file_path, "w", encoding="UTF-8") as fd:
            fd.write(text)


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

    mat_data = MaterialData(
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
            # "alpha_m": alpha_m,
            "alpha_m": np.zeros(len(alpha_k)),
            "alpha_k": np.array(alpha_k),
            "temperatures": np.array(temperatures),
            "heat_capacity": heat_capacity,
            "density": density,
            "thermal_conductivity": thermal_conductivity
        }
    )

    return mat_data
