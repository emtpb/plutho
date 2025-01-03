
# Python standard lbiraries
import os

# Third party libraries
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
from scipy import interpolate
from dotenv import load_dotenv

# Local libraries
import piezo_fem as pfem

def plot_scalar_field(
        theta,
        nodes,
        tikz_output_path: str = None):
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
    plt.show()


if __name__ == "__main__":
    load_dotenv()

    MODEL_NAME = "coupled_piezo_freq_heat_cond"
    simulation_folder = os.environ["piezo_fem_simulation_path"]
    plot_folder = os.environ["piezo_fem_plot_path"]

    current_sim_folder = os.path.join(
        simulation_folder,
        #"2MHz",
        MODEL_NAME
    )

    # Load single temp field
    theta = np.load(os.path.join(
        current_sim_folder,
        #"theta_1ms",
        f"heat_cond_theta.npy"
    ))
    mech_loss = np.load(os.path.join(
        current_sim_folder,
        "piezo_freq_mech_loss.npy"
    ))

    gmsh_handler = pfem.gmsh_handler.GmshHandler(
        #os.path.join(current_sim_folder, f"{MODEL_NAME}_disc.msh"),
        os.path.join(simulation_folder, f"disc_mesh_0DOT0001.msh"),
        True
    )
    nodes, elements = gmsh_handler.get_mesh_nodes_and_elements()

    #total_mech_loss = 0
    #for element_index, element in enumerate(elements):
    #    current_volume = 
    plot_scalar_field(
        theta[:, -1],
        nodes,
        os.path.join(
            plot_folder,
            "theta_energy_check.tex"
        )
    )
