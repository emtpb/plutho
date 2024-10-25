"""Module to check the energy conversation of the FEM thermal
piezo simulation
"""

# Python standard libraries
import os
import numpy as np

# Local libraries
import piezo_fem as pfem


def compare_energies(sim: pfem.Simulation):
    """Prints the input energy and the energy stored in the thermal field
    at the end of the simulation.

    Parameters:
        sim: Simulation object where the simulation is already done.
    """
    input_energy = pfem.calculate_electrical_input_energy(
        sim.excitation, sim.solver.q, sim.simulation_data.delta_t)
    thermal_energy = sim.solver.temp_field_energy

    print("Input energy:", input_energy)
    print("Energy in thermal field at last time step:", thermal_energy[-2])


def compare_loss_energies(sim: pfem.Simulation):
    """Prints the input energy and the loss energy of the simulation.

    Parameters:
        sim: Simulation object where the simulation is already done.
    """
    mech_loss = sim.solver.mech_loss

    number_of_time_steps = sim.simulation_data.number_of_time_steps
    total_power = np.zeros(number_of_time_steps)
    elements = sim.mesh_data.elements
    nodes = sim.mesh_data.nodes

    # Calculate the volume of each element
    volumes = np.zeros(len(elements))
    for element_index, element in enumerate(elements):
        dn = pfem.simulation.base.gradient_local_shape_functions()
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
        volumes[element_index] = (
            pfem.simulation.base.integral_volume(
                node_points)
            * 2*np.pi*jacobian_det)

    for time_step in range(number_of_time_steps):
        total_power[time_step] = np.dot(mech_loss[:, time_step], volumes)

    input_energy = pfem.calculate_electrical_input_energy(
        sim.excitation, sim.solver.q, sim.simulation_data.delta_t)

    loss_energy = np.trapezoid(total_power, None, sim.simulation_data.delta_t)

    # Calculate thermal energy in last time step
    thermal_energy = pfem.postprocessing.calculate_stored_thermal_energy(
        sim.solver.u[3*len(nodes):, -2],
        sim.mesh_data.nodes,
        sim.mesh_data.elements,
        sim.material_data.heat_capacity,
        sim.material_data.density
    )

    print("Input energy:", input_energy)
    print("Loss energy:", loss_energy)
    print("Energy in thermal field at last time step:", thermal_energy)


def model(sim_directory, sim_name):
    """Real thermo piezoelectric simulation of a disc.

    Parameters:
        base_directory: Directory where the simulation directory is created.
    """
    sim = pfem.Simulation(
        sim_directory,
        pfem.pic255,
        sim_name)
    sim.create_disc_mesh(0.005, 0.001, 0.00015)
    sim.set_simulation(
        delta_t=1e-8,
        number_of_time_steps=5000,
        gamma=0.5,
        beta=0.25,
        simulation_type=pfem.SimulationType.THERMOPIEZOELECTRIC,
    )
    sim.set_triangle_pulse_excitation(1)
    sim.set_boundary_conditions()
    sim.save_simulation_settings(
        "Simulation to check the energy conservation. "
        "The mech losses are powers and not power densities."
        "Different calculation for dt_S. Divide by volume."
    )
    sim.simulate()

    sim.save_simulation_results()
    sim.create_post_processing_views()

    return sim


if __name__ == "__main__":
    MODEL_NAME = "energy_check"
    #CWD = os.path.join(
    #    "/upb/departments/emt/Student/jonasho/Masterarbeit/piezo_fem_results/",
    #    MODEL_NAME
    #)
    CWD = os.path.join(
        "/home/jonash/uni/Masterarbeit/simulations/", MODEL_NAME
    )
    CONFIG_FILE_PATH = os.path.join(CWD, f"{MODEL_NAME}.cfg")

    if False:
        # Run simulation
        simulation = model(CWD, MODEL_NAME)

        csv_path = os.path.join(CWD, "csv")
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)

        pfem.io.create_vector_field_as_csv(
            simulation.solver.u,
            simulation.mesh_data.nodes,
            csv_path,
            True
        )
    else:
        # Load data
        simulation = pfem.Simulation.load_simulation_settings(CONFIG_FILE_PATH)
        simulation.load_simulation_results()

    compare_loss_energies(simulation)
