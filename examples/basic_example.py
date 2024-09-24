"""Module for a basic example on how to use piezo_fem."""

# Python standard libraries
import os
import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt
import gmsh

# Local libraries
import piezo_fem as pfem


def simulate(
        gmsh_handler: pfem.GmshHandler,
        time_step_count: float,
        delta_t: float,
        excitation: npt.NDArray):
    """Setup and runs the simulation with the given mesh and simulation
    parameters.

    Parameters:
        gmsh_handler: Contains the mesh information.
        time_step_count: Total number of time steps.
        delta_t: Time difference between time steps.
        excitation: Excitation values for each time step.
    """
    # Get nodes and elements from mesh
    nodes, elements = gmsh_handler.get_mesh_nodes_and_elements()
    pg_elements = gmsh_handler.get_elements_by_physical_groups(["Electrode"])
    electrode_triangles = pg_elements["Electrode"]

    print("Number of nodes:", len(nodes))
    print("Number of elements:", len(elements))

    # Time integration parameters
    gamma = 0.5
    beta = 0.25

    # Material parameters
    # TODO Use emt package for materials
    rho = 7800
    alpha_m = 1.267e5
    alpha_k = 6.259e-10
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

    # Actually not neeeded for this simulation.
    # TODO Maybe add different material classes for different models?
    thermal_conductivity = 1.1
    heat_capacity = 350

    material_data = pfem.MaterialData(
        elasticity_matrix,
        permittivity_matrix,
        piezo_matrix,
        rho,
        thermal_conductivity,
        heat_capacity,
        alpha_m,
        alpha_k
    )

    mesh_data = pfem.MeshData(
        nodes,
        elements
    )

    simulation_data = pfem.SimulationData(
        delta_t,
        time_step_count,
        gamma,
        beta,
        pfem.SimulationType.RING
    )

    solver = pfem.PiezoSim(
        mesh_data,
        material_data,
        simulation_data
    )

    # Excitation
    pg_nodes = gmsh_handler.get_nodes_by_physical_groups(
        ["Electrode", "Symaxis", "Ground"])
    solver.set_dirichlet_boundary_conditions(
        pg_nodes["Electrode"],
        # pg_nodes["Symaxis"],
        None,  # Can be set to none if symmetrical boundary condition
        pg_nodes["Ground"],
        excitation,
        time_step_count
    )

    solver.assemble()
    solver.solve_time(electrode_triangles)

    return solver.u, solver.q


if __name__ == "__main__":
    data_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                  "data")
    mesh_file_path = os.path.join(data_directory, "mesh.msh")

    if not os.path.exists(data_directory):
        os.mkdir(data_directory)

    # Simulation parameters
    TIME_STEP_COUNT = 8192
    DELTA_T = 1e-8

    # Excitation
    excitation = np.zeros(TIME_STEP_COUNT)
    excitation[1:10] = np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])

    # Create mesh
    gmsh_handler = pfem.GmshHandler(mesh_file_path)
    # For a disc use:
    # gmsh_handler.generate_rectangular_mesh()
    # For a ring add the x_offset parameter:
    gmsh_handler.generate_rectangular_mesh(x_offset=0.0005)

    # Run simulation and save results
    # u, q = simulate(gmsh_handler,
    #                 TIME_STEP_COUNT,
    #                 DELTA_T,
    #                 excitation)

    # print("Creating post processing views")
    # gmsh_handler.create_u_default_post_processing_view(
    #     u, TIME_STEP_COUNT, DELTA_T, False)

    # Open gmsh to show the fields
    # gmsh.fltk.run()

    # Save results as npz files
    # np.save(os.path.join(data_directory, "displacement"), u)
    # np.save(os.path.join(data_directory, "charge"), q)

    # Load simulation
    u = np.load(os.path.join(data_directory, "displacement.npy"))
    q = np.load(os.path.join(data_directory, "charge.npy"))
    frequencies_fem, impedence_fem = pfem.calculate_impedance(
        q, excitation, DELTA_T)

    # Get OpenCFS data
    time_list_cfs, charge_cfs = pfem.parse_charge_hist_file(
        os.path.join(data_directory, "charge_opencfs.hist"))
    frequencies_cfs, impedence_cfs = pfem.calculate_impedance(
        charge_cfs, excitation, DELTA_T)

    # Plot FEM and OpenCfs
    plt.plot(frequencies_fem, np.angle(impedence_fem), "b+", label="MyFEM")
    plt.plot(frequencies_cfs, np.angle(impedence_cfs), "r+", label="OpenCFS")
    plt.xlabel("Frequency f / Hz")
    #plt.ylabel("Impedence |Z| / $\\Omega$")
    plt.ylabel("Phase arg(Z)")
    #plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()
