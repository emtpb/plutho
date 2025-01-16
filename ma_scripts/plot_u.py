
# Python standard libraries
import os

# Third party libraries
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from scipy import interpolate
from dotenv import load_dotenv

# Local libraries
import piezo_fem as pfem


def plot_scalar_field(theta, nodes, tikz_output_path: str = None):
    label = "Temperatur $\\vartheta$ / K"
    cmap = "plasma"
    # Bigger mesh
    r_grid = np.linspace(nodes[:, 0].min(), nodes[:, 0].max(), 300)
    z_grid = np.linspace(nodes[:, 1].min(), nodes[:, 1].max(), 300)
    r_grid, z_grid = np.meshgrid(r_grid, z_grid)

    field_grid = interpolate.griddata(
        nodes,
        theta,
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
        np.min(theta),
        np.max(theta),
        10,
        endpoint=True
    )
    cbar.set_ticks(colorbar_ticks)

    plt.xlabel("Radius $r$ / m")
    plt.ylabel("Höhe $z$ / m")
    # plt.show()

    if tikz_output_path != "":
        tikzplotlib.save(tikz_output_path)


if __name__ == "__main__":
    load_dotenv()

    SIM_NAME = "pic255_thermo_time_20k"
    simulation_folder = os.environ["piezo_fem_simulation_path"]
    plot_folder = os.environ["piezo_fem_plot_path"]

    u_file_name = "thermo_piezo_u.npy"

    u = np.load(os.path.join(
        simulation_folder,
        SIM_NAME,
        u_file_name
    ))
    mesh = pfem.Mesh(
        os.path.join(simulation_folder, "disc_mesh_0DOT00004.msh"),
        True
    )

    nodes, _ = mesh.get_mesh_nodes_and_elements()
    number_of_nodes = len(nodes)

    plot_scalar_field(
        u[3*number_of_nodes:, -1],
        nodes,
        os.path.join(
            plot_folder,
            "pic255_temp_field_20k_sinusoidal.tex"
        )
    )

