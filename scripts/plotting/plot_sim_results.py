
# Python standard libraries
import os
from enum import Enum

# Third party libraries
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import tikzplotlib
import h5py
import matplotlib.patches as patches

# Local libraries
import piezo_fem as pfem


class FieldType(Enum):
    U_R = "u_r"
    U_Z = "u_z"
    POWER_LOSS = "power_loss"
    POTENTIAL = "elec_potential"
    THETA = "theta"

def plot_fem_cfs_displacement_comparison(
        sim: pfem.PiezoSimulation,
        cfs_file,
        time_step,
        tikz_output_path):
    h5 = h5py.File(cfs_file, "r")

    cfs_nodes = h5["Mesh"]["Nodes"]["Coordinates"][:]

    cfs_r_grid = np.linspace(cfs_nodes[:, 0].min(), cfs_nodes[:, 0].max(), 300)
    cfs_z_grid = np.linspace(cfs_nodes[:, 1].min(), cfs_nodes[:, 1].max(), 300)
    cfs_r_grid, cfs_z_grid = np.meshgrid(cfs_r_grid, cfs_z_grid)

    fig, axs = plt.subplots(2, 2)

    # Plot cfs
    # plt.figure(figsize=(8, 6))
    for index, component in enumerate(["u_r", "u_z"]):
        field = h5["Results"]["Mesh"]["MultiStep_1"][f"Step_{time_step}"] \
            ["mechDisplacement"]["piezo"]["Nodes"]["Real"]

        field_grid = interpolate.griddata(
            cfs_nodes[:, :2],
            field[:, 0 if component == "u_r" else 1],
            (cfs_r_grid, cfs_z_grid),
            method="linear"
        )
        im = axs[index, 0].pcolormesh(
            cfs_r_grid,
            cfs_z_grid,
            field_grid,
            cmap="viridis"
        )

        if index == 0:
            axs[0, 0].set(
                ylabel="Höhe $z$ / m"
            )
            axs[0, 0].set_title("OpenCFS")
        else:
            axs[1, 0].set(
                xlabel="Radius $r$ / m",
                ylabel="Höhe $z$ / m"
            )

    pfem_r_grid = np.linspace(
        cfs_nodes[:, 0].min(),
        cfs_nodes[:, 0].max(),
        300
    )
    pfem_z_grid = np.linspace(
        cfs_nodes[:, 1].min(),
        cfs_nodes[:, 1].max(),
        300
    )
    pfem_r_grid, pfem_z_grid = np.meshgrid(pfem_r_grid, pfem_z_grid)

    nodes = sim.mesh_data.nodes
    number_of_nodes = len(nodes)
    for index, component in enumerate(["u_r", "u_z"]):
        if index == 0:
            field = sim.solver.u[:2*number_of_nodes:2, time_step]
        else:
            field = sim.solver.u[1:2*number_of_nodes:2, time_step]

        field_grid = interpolate.griddata(
            nodes,
            field,
            (pfem_r_grid, pfem_z_grid),
            method="linear"
        )

        # Plot
        im = axs[index, 1].pcolormesh(
            pfem_r_grid,
            pfem_z_grid,
            field_grid,
            cmap="viridis"
        )

        if index == 1:
            axs[1, 1].set(
                xlabel="Radius $r$ / m"
                #ylabel="Höhe $z$ / m"
            )
        else:
            axs[0, 1].set_title("Python FEM")

        # Add colorbar with specific ticks
        cbar = plt.colorbar(
            im,
            ax=axs[index, 1],
            label=f"Verschiebung ${component}$ / m"
        )
        colorbar_ticks = np.linspace(
            np.min(field),
            np.max(field),
            10,
            endpoint=True
        )
        cbar.set_ticks(colorbar_ticks)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
    #plt.show()
    #exit(0)
    if tikz_output_path is not None:
        tikzplotlib.save(tikz_output_path)

def get_nodes_r_z(nodes):
    return nodes[:, 0], nodes[:, 1]


def get_displacement_r_z(u):
    return u[::2], u[1::2]


def get_rect_params(nodes):
    min_x = np.min(nodes[:, 0])
    min_y = np.min(nodes[:, 1])
    width = np.max(nodes[:, 0]) - min_x
    height = np.max(nodes[:, 1]) - min_y
    return min_x, min_y, width, height


def plot_average_amplitude(
        sim: pfem.PiezoSimulation,
        period,
        tikz_output_path):
    pass


def plot_displacement_vector_field(
        sim: pfem.PiezoSimulation,
        time_steps,
        tikz_output_path = None):
    nodes = sim.mesh_data.nodes
    r, z = get_nodes_r_z(nodes)
    points = np.transpose(np.vstack((r, z)))
    number_of_nodes = len(r)


    fig, axs = plt.subplots(2, 2)
    axis = [[0, 0], [0, 1], [1, 0], [1, 1]]

    u_r_max = np.max(sim.solver.u[:2*number_of_nodes:2, time_steps])
    u_z_max = np.max(sim.solver.u[1:2*number_of_nodes:2, time_steps])

    for index, time_step in enumerate(time_steps):
        u_r, u_z = get_displacement_r_z(
            sim.solver.u[:2*number_of_nodes, time_step]
        )

        r_grid = np.linspace(nodes[:, 0].min(), nodes[:, 0].max(), 45)
        z_grid = np.linspace(nodes[:, 1].min(), nodes[:, 1].max(), 45)
        r_grid, z_grid = np.meshgrid(r_grid, z_grid)

        interp_u_r = interpolate.griddata(
            points,
            u_r,
            (r_grid, z_grid),
            method="linear"
        )
        interp_u_z = interpolate.griddata(
            points,
            u_z,
            (r_grid, z_grid),
            method="linear"
        )

        # Add outline to plot
        x, y, width, height = get_rect_params(nodes)
        outline = patches.Rectangle(
            (x, y),
            width=width,
            height=height,
            linewidth=2,
            edgecolor="black",
            facecolor="none"
        )

        abs_field = np.sqrt(interp_u_r**2+interp_u_z**2)
        c, r = axis[index]
        quiver = axs[c, r].quiver(
            r_grid,
            z_grid,
            interp_u_r/u_r_max,
            interp_u_z/u_z_max,
            abs_field,
            cmap=plt.cm.viridis,
            #scale=15,
            #width=0.005
        )
        axs[c, r].add_patch(outline)

    cbar = plt.colorbar(
        quiver,
        label="Betrag der Verschiebung $|u|$"
    )
    colorbar_ticks = np.linspace(
        np.min(abs_field),
        np.max(abs_field),
        10,
        endpoint=True
    )
    cbar.set_ticks(colorbar_ticks)

    plt.grid()
    plt.xlabel("Radius $r$ / m")
    plt.ylabel("Höhe $z$ / m")
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
    plt.show()
    exit(0)
    if tikz_output_path is not None:
        tikzplotlib.save(tikz_output_path)


def plot_scalar_field(
        sim: pfem.PiezoSimulation,
        field_type: FieldType,
        time_step: int,
        tikz_output_path: str = None):
    nodes = sim.mesh_data.nodes
    number_of_nodes = len(nodes)
    field = np.zeros(number_of_nodes)
    label = ""
    cmap = "viridis"
    if field_type is FieldType.U_R:
        field = sim.solver.u[:2*number_of_nodes:2, time_step]
        label="Verschiebung $u_r$ / m"
    elif field_type is FieldType.U_Z:
        field = sim.solver.u[1:2*number_of_nodes:2, time_step]
        label="Verschiebung $u_z$ / m"
    elif field_type is FieldType.POTENTIAL:
        field = sim.solver.u[2*number_of_nodes:3*number_of_nodes, time_step]
        label="Elektrisches Potenzial $\\phi$ / V"
        cmap = "cividis"
    elif field_type is FieldType.THETA:
        field = sim.solver.u[3*number_of_nodes:, time_step]
        label="Temperatur $\\vartheta$ / K"
        cmap="plasma"
    elif field_type is FieldType.POWER_LOSS:
        raise NotImplementedError("Power loss is not implemented yet")
    else:
        raise ValueError(f"Unknown field_type given {field_type}")

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
    #plt.show()
    #exit(0)
    if tikz_output_path is not None:
        tikzplotlib.save(tikz_output_path)

if __name__ == "__main__":
    load_dotenv()

    MODEL_NAME = "energy_check_sinusoidal_20k"
    simulation_folder = os.environ["piezo_fem_simulation_path"]
    plot_folder = os.environ["piezo_fem_plot_path"]

    current_sim_folder = os.path.join(
        simulation_folder,
        "2MHz",
        MODEL_NAME
    )
    simulation = pfem.PiezoSimulation.load_simulation_settings(
        os.path.join(
            current_sim_folder,
            f"{MODEL_NAME}.cfg"
        )
    )
    simulation.load_simulation_results()

    #plot_fem_cfs_displacement_comparison(
    #    simulation,
    #    os.path.join(
    #        current_sim_folder,
    #        "impedance_255_fine.cfs"
    #    ),
    #    2500,
    #    os.path.join(plot_folder, "python_cfs_comparison_2500.tex")
    #)

    print(
        "Number of time steps:",
        simulation.simulation_data.number_of_time_steps
    )
    plot_displacement_vector_field(
        simulation,
        [19924, 19935, 19948, 19960],
        os.path.join(plot_folder, "u_vector_field_19962_sin.tex")
    )

    #plot_scalar_field(
    #    simulation,
    #    FieldType.THETA,
    #    -1,
    #    os.path.join(plot_folder, "temp_field_20k_sin.tex")
    #)
