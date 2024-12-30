"""Script for the simulation of a real piezo model."""

# Python standard libraries
import os

# Third party libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from dotenv import load_dotenv

# Local libraries
import piezo_fem as pfem


def plot_scalar_field(
        field,
        nodes):
    """Plots the given field using matplotlib.

    Parameters:
        theta: Scalar field defined on every node.
        nodes: List of node coordinates."""
    cmap = "plasma"
    # Bigger mesh
    r_grid = np.linspace(nodes[:, 0].min(), nodes[:, 0].max(), 150)
    z_grid = np.linspace(nodes[:, 1].min(), nodes[:, 1].max(), 150)
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
    cbar = plt.colorbar()
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


def example_heat_cond(base_directory):
    """Example for a heat conduction simulation."""
    sim_name = "heat_cond_test_new_model"
    mesh = pfem.Mesh(
        os.path.join(base_directory, "disc_mesh.msh"),
        True
    )

    delta_t = 0.01
    number_of_time_steps = 100

    # Create single simulation object
    sim = pfem.SingleSimulation(
        base_directory,
        sim_name,
        mesh
    )

    # Setup heat cond sim
    sim.setup_heat_conduction_time_domain(
        delta_t,
        number_of_time_steps,
        0.5
    )

    # Add materials
    sim.add_material(
        "pic255",
        pfem.materials.pic255,
        None
    )

    # Add boundary condition
    #sim.add_dirichlet_bc(
    #    pfem.VariableType.THETA,
    #    "Electrode",
    #    np.ones(number_of_time_steps)
    #)
    sim.solver.set_constant_volume_heat_source(
        np.ones(len(sim.mesh_data.elements)),
        number_of_time_steps
    )

    # Run simulation
    sim.simulate()
    sim.save_simulation_results()
    print(np.max(sim.solver.theta[:, -1]))
    print(np.min(sim.solver.theta[:, -1]))
    plot_scalar_field(sim.solver.theta[:, -1], sim.mesh_data.nodes)


def example_piezo_sim(base_directory):
    sim_name = "piezo_sim_test_new_model"
    mesh = pfem.Mesh(
        os.path.join(base_directory, "disc_mesh.msh"),
        True
    )

    delta_t = 1e-8
    number_of_time_steps = 8192

    # Create single simulation object
    sim = pfem.SingleSimulation(
        base_directory,
        sim_name,
        mesh
    )

    # Setup piezo sim
    sim.setup_piezo_time_domain(
        pfem.SimulationData(
            delta_t,
            number_of_time_steps,
            0.5,
            0.25
        )
    )

    # Add materials
    sim.add_material(
        "pic255",
        pfem.materials.pic255_alpha_m_nonzero,
        None
    )

    # Add boundary conditions
    excitation = np.zeros(number_of_time_steps)
    excitation[1:10] = (
        1 * np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    )
    sim.add_dirichlet_bc(
        pfem.FieldType.PHI,
        "Electrode",
        excitation
    )
    sim.add_dirichlet_bc(
        pfem.FieldType.PHI,
        "Ground",
        np.zeros(number_of_time_steps)
    )
    sim.add_dirichlet_bc(
        pfem.FieldType.U_R,
        "Symaxis",
        np.zeros(number_of_time_steps)
    )

    electrode_elements = mesh.get_elements_by_physical_groups(
        ["Electrode"]
    )["Electrode"]

    # Run simulation
    sim.simulate(electrode_elements=electrode_elements)
    sim.save_simulation_results()


def example_piezo_freq_sim(base_directory):
    sim_name = "piezo_sim_freq_test_new_model"
    mesh = pfem.Mesh(
        os.path.join(base_directory, "disc_mesh.msh"),
        True
    )
    mesh.generate_rectangular_mesh(mesh_size=0.00004)

    frequencies = np.linspace(0, 1e7, 1000)[1:]

    # Create single simulation object
    sim = pfem.SingleSimulation(
        base_directory,
        sim_name,
        mesh
    )

    # Setup piezo sim
    sim.setup_piezo_freq_domain(frequencies)

    # Add materials
    sim.add_material(
        "pic255",
        pfem.materials.pic255_alpha_m_nonzero,
        None
    )

    # Add boundary conditions
    sim.add_dirichlet_bc(
        pfem.FieldType.PHI,
        "Electrode",
        np.ones(len(frequencies))
    )
    sim.add_dirichlet_bc(
        pfem.FieldType.PHI,
        "Ground",
        np.zeros(len(frequencies))
    )
    sim.add_dirichlet_bc(
        pfem.FieldType.U_R,
        "Symaxis",
        np.zeros(len(frequencies))
    )

    electrode_elements = mesh.get_elements_by_physical_groups(
        ["Electrode"]
    )["Electrode"]

    # Run simulation
    sim.simulate(
        electrode_elements=electrode_elements
    )
    sim.save_simulation_results()
    sim.save_simulation_settings()


def example_therm_piezo_sim(base_directory):
    sim_name = "test_temp"
    mesh = pfem.Mesh(
        os.path.join(base_directory, "disc_mesh_default.msh"),
        True
    )

    delta_t = 1e-8
    number_of_time_steps = 100

    # Create single simulation object
    sim = pfem.SingleSimulation(
        base_directory,
        sim_name,
        mesh
    )

    # Setup piezo sim
    sim.setup_thermal_piezo_time_domain(
        pfem.SimulationData(
            delta_t,
            number_of_time_steps,
            0.5,
            0.25
        )
    )

    # Add materials
    sim.add_material(
        "pic255",
        pfem.materials.pic255_alpha_m_nonzero,
        None
    )

    # Add boundary conditions
    excitation = np.zeros(number_of_time_steps)
    excitation[1:10] = (
        1 * np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    )
    sim.add_dirichlet_bc(
        pfem.FieldType.PHI,
        "Electrode",
        excitation
    )
    sim.add_dirichlet_bc(
        pfem.FieldType.PHI,
        "Ground",
        np.zeros(number_of_time_steps)
    )
    sim.add_dirichlet_bc(
        pfem.FieldType.U_R,
        "Symaxis",
        np.zeros(number_of_time_steps)
    )

    electrode_elements = mesh.get_elements_by_physical_groups(
        ["Electrode"]
    )["Electrode"]

    # Run simulation
    sim.simulate(electrode_elements=electrode_elements)
    sim.save_simulation_results()
    sim.save_simulation_settings()


def real_model(base_directory):
    """Real thermo piezoelectric simulation of a disc.

    Parameters:
        base_directory: Directory where the simulation directory is created.
    """
    sim_name = "runtime_check_20k"
    sim_directory = os.path.join(base_directory, sim_name)
    sim = pfem.PiezoSimulation(
        sim_directory,
        sim_name)
    sim.create_disc_mesh(0.005, 0.001, 0.0001)
    sim.set_material_data(
        "pic255",
        pfem.pic255,
    )
    #sim.material_manager.material_data.alpha_m = 1.267e5
    sim.set_simulation(
        delta_t=1e-8,
        number_of_time_steps=20000,
        gamma=0.5,
        beta=0.25,
        simulation_type=pfem.SimulationType.THERMOPIEZOELECTRIC,
    )
    sim.set_sinusoidal_excitation(20, 2e6) # 1.889e5
    #sim.set_triangle_pulse_excitation(1)
    sim.set_boundary_conditions()
    sim.save_simulation_settings(
        "Simulation for runtime evaluation.")
    sim.simulate()

    sim.save_simulation_results()
    # sim.create_post_processing_views()


if __name__ == "__main__":
    load_dotenv()

    CWD = os.getenv("piezo_fem_simulation_path")
    if CWD is None:
        print("Couldn't find simulation path.")
        exit(1)

    # real_model(CWD)
    # example_heat_cond(CWD)
    # example_piezo_sim(CWD)
    # example_piezo_freq_sim(CWD)
    example_therm_piezo_sim(CWD)
