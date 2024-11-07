"""Module to check the energy conversation of the FEM simulation"""

# Python standard libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# Third party libraries
from dotenv import load_dotenv

# Local libraries
import piezo_fem as pfem
from piezo_fem.simulation.base import *


def integral_ed(
    node_points,
    u_e,
    v_e,
    piezo_matrix,
    permittivity_matrix,
    jacobian_inverted_t):
    def inner(s, t):
        r = local_to_global_coordinates(node_points, s, t)[0]
        dn = gradient_local_shape_functions()
        global_dn = np.dot(jacobian_inverted_t, dn)
        b_opt = b_operator_global(node_points, s, t, jacobian_inverted_t)

        e_e = -np.dot(global_dn, v_e)
        #d_e = (
        #    np.dot(piezo_matrix, np.dot(b_opt, u_e))
        #     + np.dot(permittivity_matrix, e_e)
        #)
        d_e = np.dot(permittivity_matrix, e_e)

        return 1/2*np.dot(d_e.T, e_e)*r

    return quadratic_quadrature(inner)

def integral_ts(
        node_points,
        u_e,
        elasticity_matrix,
        jacobian_inverted_t):
    def inner(s, t):
        r = local_to_global_coordinates(node_points, s, t)[0]
        dn = gradient_local_shape_functions()
        global_dn = np.dot(jacobian_inverted_t, dn)
        b_opt = b_operator_global(node_points, s, t, jacobian_inverted_t)

        s_e = np.dot(b_opt, u_e)
        t_e = np.dot(elasticity_matrix, s_e)

        return 1/2*np.dot(t_e, s_e)*r

    return quadratic_quadrature(inner)


def calculate_stored_electrical_energy(sim):
    nodes = sim.mesh_data.nodes
    number_of_nodes = len(nodes)

    energy = np.zeros(sim.simulation_data.number_of_time_steps)
    for time_step in range(sim.simulation_data.number_of_time_steps):
        u = sim.solver.u[:, time_step]
        for element_index, element in enumerate(sim.mesh_data.elements):
            dn = gradient_local_shape_functions()
            node_points = np.array([
                [nodes[element[0]][0],
                 nodes[element[1]][0],
                 nodes[element[2]][0]],
                [nodes[element[0]][1],
                 nodes[element[1]][1],
                 nodes[element[2]][1]]
            ])
            jacobian = np.dot(node_points, dn.T)
            jacobian_det = np.linalg.det(jacobian)
            jacobian_inverted_t = np.linalg.inv(jacobian).T

            u_e = np.array([u[2*element[0]],
                            u[2*element[0]+1],
                            u[2*element[1]],
                            u[2*element[1]+1],
                            u[2*element[2]],
                            u[2*element[2]+1]])
            v_e = np.array([u[element[0]+2*number_of_nodes],
                            u[element[1]+2*number_of_nodes],
                            u[element[2]+2*number_of_nodes]])
            energy[time_step] += (
                integral_ed(
                    node_points,
                    u_e,
                    v_e,
                    sim.material_data.piezo_matrix,
                    sim.material_data.permittivity_matrix,
                    jacobian_inverted_t)
                #+ integral_ts(
                #    node_points,
                #    u_e,
                #    sim.material_data.elasticity_matrix,
                #    jacobian_inverted_t)
                ) * 2*np.pi*jacobian_det


    return energy


def model(sim_directory, sim_name):
    """Real thermo piezoelectric simulation of a disc.

    Parameters:
        base_directory: Directory where the simulation directory is created.
    """
    sim = pfem.PiezoSimulation(
        sim_directory,
        pfem.pic255,
        sim_name)
    sim.create_disc_mesh(0.005, 0.001, 0.0001)

    sim.set_simulation(
        delta_t=1e-8,
        number_of_time_steps=1500,
        gamma=0.5,
        beta=0.25,
        simulation_type=pfem.SimulationType.PIEZOELECTRIC,
    )
    sim.set_triangle_pulse_excitation(1)
    sim.set_boundary_conditions()
    sim.save_simulation_settings(
        "Simulation to check electric energy.")
    sim.simulate()

    sim.save_simulation_results()
    sim.create_post_processing_views()

    return sim


def load_hist_file(file):
    times = []
    values = []
    with open(file, "r", encoding="UTF-8") as fd:
        for line in fd.readlines()[3:]:
            time, value = line.split("  ")
            times.append(float(time.strip()))
            values.append(float(value.strip()))

    return np.array(times), np.array(values)


if __name__ == "__main__":
    MODEL_NAME = "electric_energy_check"
    load_dotenv()

    CWD = os.getenv("piezo_fem_simulation_path")
    if CWD is None:
        print("Couldn't find simulation path.")
        exit(1)

    CHARGE_FILE_PATH = os.path.join(
        CWD, f"{MODEL_NAME}_q.npy"
    )
    DISPLACEMENT_FILE_PATH = os.path.join(
        CWD, f"{MODEL_NAME}_u.npy"
    )
    CONFIG_FILE_PATH = os.path.join(CWD, f"{MODEL_NAME}.cfg")

    if True:
        # Run simulation
        simulation = model(CWD, MODEL_NAME)

        pfem.create_vector_field_as_csv(
            simulation.solver.u,
            simulation.mesh_data.nodes,
            os.path.join(CWD, "csv"),
            False
        )
    else:
        # Load data
        simulation = pfem.PiezoSimulation.load_simulation_settings(CONFIG_FILE_PATH)
        simulation.solver.q = np.load(CHARGE_FILE_PATH)
        simulation.solver.u = np.load(DISPLACEMENT_FILE_PATH)
    exit(0)
    energy = calculate_stored_electrical_energy(simulation)

    input_energy = pfem.calculate_electrical_input_energy(
        simulation.excitation,
        simulation.solver.q,
        simulation.simulation_data.delta_t)

    time_values = np.arange(simulation.simulation_data.number_of_time_steps)*simulation.simulation_data.delta_t
    for time_value, energy_value in zip(time_values, energy):
        print(time_value, energy_value)
    print("Input energy:", input_energy)

    plt.plot(energy)
    plt.grid()
    plt.show()