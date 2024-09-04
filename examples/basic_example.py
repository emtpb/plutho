import os
import numpy as np
import matplotlib.pyplot as plt
import gmsh
import piezo_fem as pfem

def simulate(mesh_file_path, time_step_count, delta_t, excitation):
    # Create mesh and load it
    pfem.mesh.generate_rectangular_mesh(mesh_file_path)

    # Get nodes and elements from parser
    nodes, elements = pfem.mesh.get_mesh_nodes_and_elements(mesh_file_path)
    pg_elements = pfem.mesh.get_elements_by_physical_groups(mesh_file_path, ["Electrode"])
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
    alpha_K = 6.259e-10
    elasticity_matrix = np.array([
        [1.19e11, 0.83e11,       0, 0.84e11],
        [0.83e11, 1.17e11,       0, 0.83e11],
        [      0,       0, 0.21e11,       0],
        [0.84e11, 0.83e11,       0, 1.19e11]
    ])
    permittivity_matrix = np.diag([8.15e-9, 6.58e-9])
    piezo_matrix = np.array([
        [0, 0, 12.09, 0],
        [-6.03, 15.49, 0, -6.03]
    ])

    thermal_conductivity = 1.1
    heat_capacity = 350
    thermal_diffusivity = thermal_conductivity/(rho*heat_capacity)

    # Excitation
    excitation_nodes, excitation_values = pfem.create_node_excitation(mesh_file_path, excitation, time_step_count)

    material_data = pfem.MaterialData(
        elasticity_matrix,
        permittivity_matrix,
        piezo_matrix,
        rho,
        thermal_diffusivity,
        heat_capacity,
        alpha_M,
        alpha_K
    )

    mesh_data = pfem.MeshData(
        nodes,
        elements
    )

    simulation_data = pfem.SimulationData(
        delta_t,
        time_step_count,
        GAMMA,
        BETA
    )

    # Solve
    M, C, K = pfem.assemble(mesh_data, material_data)
    u, q = pfem.solve_time(
        M,
        C,
        K,
        mesh_data,
        material_data,
        simulation_data,
        excitation_nodes,
        excitation_values,
        electrode_triangles)


    return u, q

if __name__ == "__main__":
    data_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
    mesh_file_path = os.path.join(data_directory, "mesh.msh")

    if not os.path.exists(data_directory):
        os.mkdir(data_directory)

    # Simulation parameters
    TIME_STEP_COUNT = 8192
    DELTA_T = 1e-8

    # Excitation
    excitation = np.zeros(TIME_STEP_COUNT)
    excitation[1:10] = np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])

    # Run simulation and save results
    u, q = simulate(mesh_file_path, TIME_STEP_COUNT, DELTA_T, excitation)


    print(u.shape)
    np.save(os.path.join(data_directory, "displacement"), u)
    np.save(os.path.join(data_directory, "charge"), q)

    # Load simulation
    #u = np.load(os.path.join(data_directory, "displacement.npy"))
    #q = np.load(os.path.join(data_directory, "charge.npy"))
    
    print("Creating post processing views")
    pfem.mesh.create_post_processing_views(mesh_file_path, u, 1000, DELTA_T)
    
    frequencies_fem, impedence_fem = pfem.calculate_impedance(q, excitation, DELTA_T)

    # Get OpenCFS data
    #time_list_cfs, charge_cfs = pfem.read_charge_open_cfs(os.path.join(data_directory, "charge_opencfs.hist"))
    #frequencies_cfs, impedence_cfs = pfem.calculate_impedance(charge_cfs, excitation, DELTA_T)

    # Plot FEM and OpenCfs
    plt.plot(frequencies_fem, np.abs(impedence_fem), label="MyFEM")
    #plt.plot(frequencies_cfs, np.abs(impedence_cfs), "+", label="OpenCFS")
    plt.xlabel("Frequency f / Hz")
    plt.ylabel("Impedence |Z| / $\\Omega$")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()
    
    gmsh.open(os.path.join(data_directory, "mesh_results.msh"))
    gmsh.fltk.run()


