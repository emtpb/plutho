
# Python standard libraries
from dataclasses import Field
import os
from enum import Enum

# Third party libraries
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from scipy import interpolate
from dotenv import load_dotenv

# Local libraries
import piezo_fem as pfem

class FieldType(Enum):
    U_R = "u_r"
    U_Z = "u_z"
    THETA = "theta"

def plot_scalar_field(field, nodes, field_type, tikz_output_path: str = None):
    if field_type == FieldType.U_R or field_type == FieldType.U_Z:
        label = f"Verschiebung ${field_type.value}$ / m"
        cmap = "viridis"
        if field_type == FieldType.U_R:
            field = field[:2*len(nodes):2]
        else:
            field = field[1:2*len(nodes):2]
    elif field_type == FieldType.THETA:
        label = "Temperatur $\\vartheta$ / K"
        cmap = "plasma"
        field = field[:3*len(nodes)]
    else:
        raise NotImplementedError(
            f"Not implemented for field type {field_type.value}"
        )

    # Bigger mesh
    r_grid = np.linspace(nodes[:, 0].min(), nodes[:, 0].max(), 300)
    z_grid = np.linspace(nodes[:, 1].min(), nodes[:, 1].max(), 300)
    r_grid, z_grid = np.meshgrid(r_grid, z_grid)

    field_grid = interpolate.griddata(
        nodes,
        field,
        (r_grid, z_grid),
        method="linear"
    )

    # Plot
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(r_grid, z_grid, field_grid, cmap=cmap)

    # Add colorbar with specific ticks
    cbar = plt.colorbar(
        label=label
    )
    colorbar_ticks = np.linspace(
        np.min(field),
        np.max(field),
        10,
        endpoint=True
    )
    cbar.set_ticks(colorbar_ticks)

    plt.xlabel("Radius $r$ / m")
    plt.ylabel("Höhe $z$ / m")
    plt.show()

    # if tikz_output_path != "":
    #     tikzplotlib.save(tikz_output_path)


if __name__ == "__main__":
    load_dotenv()

    SIM_NAME = "pic181_thermo_freq_temp_independent_20c"
    simulation_folder = os.environ["piezo_fem_simulation_path"]
    plot_folder = os.environ["piezo_fem_plot_path"]

    u_file_name = "piezo_freq_u.npy"

    u = np.load(os.path.join(
        simulation_folder,
        SIM_NAME,
        u_file_name
    ))
    mesh = pfem.Mesh(
        os.path.join(simulation_folder, "ring_mesh_0DOT00004.msh"),
        True
    )

    nodes, _ = mesh.get_mesh_nodes_and_elements()
    number_of_nodes = len(nodes)

    plot_scalar_field(
        np.real(u[:, -1]),
        nodes,
        FieldType.U_Z,
        os.path.join(
            plot_folder,
            "pic255_temp_field_20k_sinusoidal.tex"
        )
    )

