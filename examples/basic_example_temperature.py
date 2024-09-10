"""Module for a basic exmaple on how to use the temperature piezo_fem."""

# Python standard libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# Third party libraries
import gmsh

# Local libraries
import piezo_fem as pfem


def simulate(gmsh_handler, time_step_count, delta_t, excitation):
    # Get nodes and elements from mesh
    nodes, elements = gmsh_handler.get_mesh_nodes_and_elements()
    pg_elements = gmsh_handler.get_elements_by_physical_groups(["Electrode"])
    electrode_triangles = pg_elements["Electrode"]

    print("Number of nodes:", len(nodes))
    print("Number of elements:", len(elements))

    # Time integration parameters
    GAMMA = 0.5
    BETA = 0.25

    # Material parameters
    # TODO Use emt package for materials
    rho = 7800
    alpha_M = 1.267e5
    alpha_M = 0
    alpha_K = 6.259e-10
    elasticity_matrix = np.array([
        [1.19e11, 0.83e11,       0, 0.84e11],
        [0.83e11, 1.17e11,       0, 0.83e11],
        [0, 0, 0.21e11, 0],
        [0.84e11, 0.83e11,       0, 1.19e11]
    ])
    permittivity_matrix = np.diag([8.15e-9, 6.58e-9])
    piezo_matrix = np.array([
        [0, 0, 12.09, 0],
        [-6.03, 15.49, 0, -6.03]
    ])
    tau = 1.99e-11  # TODO What is the name of this?

    thermal_conductivity = 1.1
    heat_capacity = 350

    mesh_data = pfem.MeshData(nodes, elements)
    material_data = pfem.MaterialData(
        elasticity_matrix,
        permittivity_matrix,
        piezo_matrix,
        rho,
        thermal_conductivity,
        heat_capacity,
        alpha_M,
        alpha_K,
        tau
    )
    simulation_data = pfem.SimulationData(
        delta_t,
        time_step_count,
        GAMMA,
        BETA
    )

    # Excitation
    pg_nodes = gmsh_handler.get_nodes_by_physical_groups(
        ["Electrode", "Symaxis", "Ground"])
    excitation_nodes, excitation_values = pfem.create_node_excitation(
        pg_nodes["Electrode"],
        pg_nodes["Symaxis"],
        pg_nodes["Ground"],
        excitation,
        time_step_count)

    # Solve
    M, C, K = pfem.assemble(mesh_data, material_data)
    u, q, power_loss = pfem.solve_time(
        M, C, K,
        mesh_data,
        material_data,
        simulation_data,
        excitation_nodes,
        excitation_values,
        electrode_triangles
    )

    # pfem.create_vector_field_as_csv(u, nodes, os.path.join(data_directory, 
    # "field"))

    return u, q, power_loss


if __name__ == "__main__":
    data_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                  "data")
    mesh_file_path = os.path.join(data_directory, "mesh.msh")

    if not os.path.exists(data_directory):
        os.mkdir(data_directory)

    # Simulation parameters
    TIME_STEP_COUNT = 50
    DELTA_T = 1e-8

    # Excitation
    excitation = np.zeros(TIME_STEP_COUNT)
    excitation[1:10] = np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])

    # Create mesh
    gmsh_handler = pfem.mesh.GmshHandler(mesh_file_path)
    gmsh_handler.generate_rectangular_mesh(mesh_size=0.0001)

    # Run simulation and save results
    u, q, power_loss = simulate(gmsh_handler,
                                TIME_STEP_COUNT,
                                DELTA_T,
                                excitation)

    # print("Creating post processing views")
    gmsh_handler.create_post_processing_views(u, TIME_STEP_COUNT, DELTA_T)
    gmsh_handler.create_power_loss_post_processing_view(power_loss,
                                                        TIME_STEP_COUNT,
                                                        DELTA_T)

    gmsh.fltk.run()

    # np.save(os.path.join(data_directory, "displacement"), u)
    # np.save(os.path.join(data_directory, "charge"), q)

    exit(0)

    # Load simulation
    u = np.load(os.path.join(data_directory, "displacement.npy"))
    q = np.load(os.path.join(data_directory, "charge.npy"))
    frequencies_fem, impedence_fem = pfem.calculate_impedance(
        q, excitation, DELTA_T)

    # Get OpenCFS data
    time_list_cfs, charge_cfs = pfem.read_charge_open_cfs(
        os.path.join(data_directory, "charge_opencfs.hist"))
    frequencies_cfs, impedence_cfs = pfem.calculate_impedance(
        charge_cfs, excitation, DELTA_T)

    # Plot FEM and OpenCfs
    plt.plot(frequencies_fem, np.abs(impedence_fem), label="MyFEM")
    plt.plot(frequencies_cfs, np.abs(impedence_cfs), "+", label="OpenCFS")
    plt.xlabel("Frequency f / Hz")
    plt.ylabel("Impedence |Z| / $\\Omega$")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()
    


