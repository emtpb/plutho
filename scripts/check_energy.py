"""Module to check the energy conversation of the FEM simulation"""

# Python standard libraries
import os
import numpy as np
import matplotlib.pyplot as plt

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


def compare_loss_energies(sim: pfem.Simulation, is_power_density: False):
    """Prints the input energy and the loss energy of the simulation.
    Assumes the mech_loss in the simulation is not a power density but just
    the power per element per time step.

    Parameters:
        sim: Simulation object where the simulation is already done.
    """
    mech_loss = sim.solver.mech_loss

    number_of_time_steps = sim.simulation_data.number_of_time_steps
    total_power = np.zeros(number_of_time_steps)
    elements = sim.mesh_data.elements
    nodes = sim.mesh_data.nodes

    volumes = np.zeros(len(elements))
    if is_power_density:
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
                pfem.simulation.fem_piezo_temp_time.integral_volume(
                    node_points)
                * 2*np.pi*jacobian_det)

    for time_step in range(number_of_time_steps):
        if is_power_density:
            total_power[time_step] = np.dot(mech_loss[:, time_step], volumes)
        else:
            total_power[time_step] = np.sum(mech_loss[:, time_step])

    input_energy = pfem.calculate_electrical_input_energy(
        sim.excitation, sim.solver.q, sim.simulation_data.delta_t)

    loss_energy = np.trapezoid(total_power, None, sim.simulation_data.delta_t)

    thermal_energy = sim.solver.temp_field_energy

    print("Input energy:", input_energy)
    print("Loss energy:", loss_energy)
    if is_power_density:
        # Since the power density is used the thermal simulation does work fine
        print("Energy in thermal field at last time step:", thermal_energy[-2])


def get_max_temp_value(sim: pfem.Simulation):
    node_count = len(sim.mesh_data.nodes)
    theta = sim.solver.u[3*node_count:, :]
    print(theta.shape)
    print(
        f"Max temperature value {np.max(theta)} "
        f"at time step {np.argmax(theta)}"
    )


def model(sim_directory, sim_name):
    """Real thermo piezoelectric simulation of a disc.

    Parameters:
        base_directory: Directory where the simulation directory is created.
    """
    sim = pfem.Simulation(
        sim_directory,
        pfem.pic255,
        sim_name)
    sim.create_disc_mesh(0.005, 0.001, 0.0001)
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
        "Simulation to check the energy conservation. The mech losses are powers and not power densities. Different calculation for dt_S. Divide by volume. e erm added to calculation.")
    sim.simulate()

    sim.save_simulation_results()
    sim.create_post_processing_views()

    return sim

if __name__ == "__main__":
    MODEL_NAME = "energy_check_eds"
    CWD = os.path.join(
        "/upb/departments/emt/Student/jonasho/Masterarbeit/piezo_fem_results/",
        MODEL_NAME
    )
    #CWD = os.path.join(
    #    "/home/jonash/uni/Masterarbeit/simulations/", MODEL_NAME
    #)
    THERM_ENERGY_FILE_PATH = os.path.join(
        CWD, f"{MODEL_NAME}_temp_field_energy.npy"
    )
    CHARGE_FILE_PATH = os.path.join(
        CWD, f"{MODEL_NAME}_q.npy"
    )
    DISPLACEMENT_FILE_PATH = os.path.join(
        CWD, f"{MODEL_NAME}_u.npy"
    )
    CONFIG_FILE_PATH = os.path.join(CWD, f"{MODEL_NAME}.cfg")

    if False:
        # Run simulation
        simulation = model(CWD, MODEL_NAME)
    else:
        # Load data
        simulation = pfem.Simulation.load_simulation_settings(CONFIG_FILE_PATH)
        simulation.solver.q = np.load(CHARGE_FILE_PATH)
        simulation.solver.temp_field_energy = np.load(THERM_ENERGY_FILE_PATH)
        simulation.solver.u = np.load(DISPLACEMENT_FILE_PATH)
        simulation.solver.mech_loss = np.load(os.path.join(
            CWD, f"{MODEL_NAME}_mech_loss.npy"
        ))

    # get_max_temp_value(simulation)
    # compare_energies(simulation)
    compare_loss_energies(simulation, True)

    # print(np.trapezoid(
    #     np.mean(simulation.solver.mech_loss, axis=0),
    #     None,
    #     simulation.simulation_data.delta_t))

    #plt.plot(np.mean(simulation.solver.mech_loss, axis=0), label="Mech loss")
    #plt.plot(simulation.solver.temp_field_energy, label="Temp field energy")
    #plt.show()
