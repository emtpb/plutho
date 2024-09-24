"""Module for a basic exmaple on how to use the temperature piezo_fem."""

# Python standard libraries
import os
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy import integrate

# Third party libraries
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

    mesh_data = pfem.MeshData(nodes, elements)
    material_data = pfem.pic255

    simulation_data = pfem.SimulationData(
        delta_t=delta_t,
        number_of_time_steps=time_step_count,
        gamma=0.5,
        beta=0.25,
        model_type=pfem.ModelType.RING
    )

    # Create solver object and run simulation
    solver = pfem.PiezoSimTherm(
        mesh_data,
        material_data,
        simulation_data)

    # Excitation
    solver.set_dirichlet_boundary_conditions(
        gmsh_handler,
        excitation,
        time_step_count
    )

    solver.assemble()
    solver.solve_time(electrode_triangles)

    return solver.u, solver.q, solver.mech_loss, solver.temp_field_energy

def create_power_loss_field_for_open_cfs(
        field_values,
        mesh_data,
        simulation_data,
        output_path):
    nodes = mesh_data.nodes
    elements = mesh_data.elements
    number_of_nodes = len(nodes)
    number_of_time_steps = simulation_data.number_of_time_steps

    resulting_field = np.zeros(shape=(number_of_time_steps, number_of_nodes))
    for time_index in range(number_of_time_steps):
        for element_index, element in enumerate(elements):
            field_value = field_values[element_index][time_index]
            for node_index in element:
                resulting_field[time_index][node_index] += 1/3*field_value

    output_text = "time,r,z,heat_density\n"
    for time_index, node_values in enumerate(resulting_field):
        for node_index, node_value in enumerate(node_values):
            r, z = nodes[node_index]
            output_text += f"{time_index},{r},{z},{node_value}\n"

    with open(output_path, "w", encoding="UTF-8") as fd:
        fd.write(output_text)



if __name__ == "__main__":
    data_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                  "data")

    simulation_directory = os.path.join(
        data_directory, "..", "simulations", "energy_check_big")
    mesh_file_path = os.path.join(simulation_directory, "mesh.msh")

    if not os.path.exists(data_directory):
        os.mkdir(data_directory)

    # Simulation parameters
    TIME_STEP_COUNT = 100000
    DELTA_T = 1e-8

    # Excitation
    # Old excitation
    excitation = np.zeros(TIME_STEP_COUNT)
    excitation[1:10] = np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])

    # New excitation
    # excitation = sinusodial_excitation(2001953, TIME_STEP_COUNT, DELTA_T)

    # Create mesh
    gmsh_handler = pfem.GmshHandler(mesh_file_path)
    gmsh_handler.generate_rectangular_mesh(mesh_size=0.000075)

    # Run simulation and save results
    u, q, power_loss, temp_field_energy = simulate(gmsh_handler,
                                TIME_STEP_COUNT,
                                DELTA_T,
                                excitation)

    np.savetxt(
        os.path.join(simulation_directory, "temp_field_energy.txt"),
        temp_field_energy)
    # print("Creating post processing views")
    #gmsh_handler.create_u_default_post_processing_view(
    #    u, TIME_STEP_COUNT, DELTA_T, True)
    #gmsh_handler.create_element_post_processing_view(
    #    power_loss,
    #    TIME_STEP_COUNT,
    #    DELTA_T,
    #    1,
    #    "Mechanical loss")

    # Open gmsh to show the fields
    #gmsh.fltk.run()

    eie = calculate_electrical_input_energy(excitation, q, DELTA_T)
    with open(
            os.path.join(simulation_directory, "excitation_energy.txt"),
            "w",
            encoding="UTF-8") as fd:
        fd.write(str(eie))
    # print("Electrical input energy:", eie)

    # Save results as npz files
    # np.save(os.path.join(simulation_directory, "displacement"), u)
    np.save(os.path.join(simulation_directory, "charge"), q)

    print("Script finished.")
    # Load simulation
    # u = np.load(os.path.join(data_directory, "displacement.npy"))
    # q = np.load(os.path.join(data_directory, "charge.npy"))
    # frequencies_fem, impedence_fem = pfem.calculate_impedance(
    #     q, excitation, DELTA_T)

    # Get OpenCFS data
    # time_list_cfs, charge_cfs = pfem.read_charge_open_cfs(
    #     os.path.join(data_directory, "charge_opencfs.hist"))
    # frequencies_cfs, impedence_cfs = pfem.calculate_impedance(
    #     charge_cfs, excitation, DELTA_T)


    # Plot FEM and OpenCfs
    # plt.plot(frequencies_fem, np.abs(impedence_fem), label="MyFEM")
    # plt.plot(frequencies_cfs, np.abs(impedence_cfs), "+", label="OpenCFS")
    # plt.xlabel("Frequency f / Hz")
    # plt.ylabel("Impedence |Z| / $\\Omega$")
    # plt.yscale("log")
    # plt.legend()
    # plt.grid()
    # plt.show()
