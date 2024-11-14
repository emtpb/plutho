"""Script for a thermal simulation after a thermo-piezo simulation
in order to calculate the Temperature for longer time intervals."""

# Python standard libraries
import os
import numpy as np
import numpy.typing as npt

# Third party libraries
from dotenv import load_dotenv
import gmsh

# Local libraries
import piezo_fem as pfem


def run_piezo_thermal_simulation(base_directory, name):
    """Example for a thermo piezoelectric simulation of a disc.

    Parameters:
        base_directory: Directory where the simulation directory is created.
    """
    piezo_delta_t = 1e-8
    frequency = 2e5

    sim_directory = os.path.join(base_directory, name)
    sim = pfem.PiezoSimulation(sim_directory, pfem.pic255, name)
    sim.create_disc_mesh(0.005, 0.001, 0.0001)
    sim.set_simulation(
        delta_t=piezo_delta_t,
        number_of_time_steps=20000,
        gamma=0.5,
        beta=0.25,
        simulation_type=pfem.SimulationType.THERMOPIEZOELECTRIC,
    )
    sim.set_sinusoidal_excitation(
        20,
        frequency
    )
    sim.set_boundary_conditions()
    sim.save_simulation_settings(
        "An example for a thermal piezo-electric simulation.")
    sim.simulate()

    sim.save_simulation_results()


def analyze_energies(
        piezo_sim: pfem.PiezoSimulation,
        heat_sim: pfem.HeatConductionSim,
        avg_mech_loss: npt.NDArray):
    """Given a thermal-piezoeelectric simulation and the heat conduction
    simulation the energies of both simulations are calculated and printed.
    The input energy is calculated using the energy given to the 
    thermo-piezoelectric and the energy given to the heat conduction sim.
    The total mechanical loss energy is calculated using the mechanical loss
    energy of the thermo-pieoelectric simulation and the mechanical loss energy
    of the heat conduction simulation.

    Parameters:
        piezo_sim: Thermo-piezoelectric simulation.
        heat_sim: Heat conduction simulation."""
    # Calculate input energy
    heat_sim_duration = heat_sim.simulation_data.delta_t * \
        heat_sim.simulation_data.number_of_time_steps
    input_energy_piezo_sim = pfem.calculate_electrical_input_energy(
        piezo_sim.excitation,
        piezo_sim.solver.q,
        piezo_sim.simulation_data.delta_t
    )
    input_energy_therm_sim = pfem.calculate_electrical_input_energy(
        piezo_sim.excitation[-25:],
        piezo_sim.solver.q[-25:],
        piezo_sim.simulation_data.delta_t
    ) * heat_sim_duration/(25*piezo_sim.simulation_data.delta_t)

    total_input_energy = input_energy_piezo_sim + input_energy_therm_sim
    print("Total input energy is:", total_input_energy)

    # Calculate mech loss energy
    nodes = piezo_sim.mesh_data.nodes
    elements = piezo_sim.mesh_data.elements
    piezo_sim_mech_loss_energy = calculate_mech_loss_energy(
        piezo_sim.solver.mech_loss,
        nodes,
        elements,
        piezo_sim.simulation_data.delta_t
    )
    extrapolated_mech_loss_energy = calculate_mech_loss_energy(
        np.tile(
            avg_mech_loss.reshape(-1, 1),
            (1, heat_sim.simulation_data.number_of_time_steps)
        ),
        nodes,
        elements,
        heat_sim.simulation_data.delta_t
    )

    total_mech_loss_energy = piezo_sim_mech_loss_energy + extrapolated_mech_loss_energy
    print("Total mech loss energy is:", total_mech_loss_energy)

    # Calculate stored thermal energy
    thermal_stored_energy = pfem.postprocessing.calculate_stored_thermal_energy(
        heat_sim.theta[:, -1],
        piezo_sim.mesh_data.nodes,
        piezo_sim.mesh_data.elements,
        piezo_sim.material_manager.heat_capacity,
        piezo_sim.material_manager.density
    )

    print("Stored energy in thermal field:", thermal_stored_energy)


def calculate_mech_loss_energy(
        mech_loss: npt.NDArray,
        nodes: npt.NDArray,
        elements: npt.NDArray,
        delta_t: float):
    """Given the mech_loss_density[element_index, time_index], which is a
    power density, the total energy is calculated.

    Parameters:
        mech_loss: Power loss density.
        nodes: Nodes of the mesh.
        elements: Elements of the mesh.
        delta_t: Time step difference of the data.

    Returns:
        Total energy."""
    number_of_time_steps = mech_loss.shape[1]
    total_power = np.zeros(number_of_time_steps)

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

    return np.trapezoid(total_power, None, delta_t)


def print_temperature_at_right_boundary(
        nodes,
        theta):
    """Prints the average temperature at the nodes of theright boundary
    of the given temperature field.

    Parameters:
        nodes: Nodes of the mesh
        theta: Temperature field theta[element_index]
    """
    # Find nodes which are most right
    radius = np.max(nodes[:, 1])
    print(radius)

    right_node_indices = []
    for index, node in enumerate(nodes):
        if np.isclose(node[1], radius):
            right_node_indices.append(index)

    print(
        "Average temperature at right boundary:",
        np.mean(theta[right_node_indices])
    )


if __name__ == "__main__":
    load_dotenv()

    CWD = os.getenv("piezo_fem_simulation_path")
    if CWD is None:
        print("Couldn't find simulation path.")
        exit(1)

    PIEZO_SIM_NAME = "real_model_1_889kHz_20k"
    # run_piezo_thermal_simulation(CWD, PIEZO_SIM_NAME)

    # Load data from piezo sim
    piezo_sim_folder = os.path.join(CWD, "2kHz", PIEZO_SIM_NAME)
    piezo_sim = pfem.PiezoSimulation.load_simulation_settings(
        os.path.join(piezo_sim_folder, f"{PIEZO_SIM_NAME}.cfg")
    )

    if not isinstance(piezo_sim.solver, pfem.PiezoSimTherm):
        raise IOError("The given simulation must be thermo-piezoelectric")

    piezo_sim.load_simulation_results()
    u = piezo_sim.solver.u

    TIME_STEPS_PER_PERIOD = 25  # Empirically

    # Simulation time of the heat conduction simulation
    SIMULATION_TIME = 20 # In seconds

    # Number of periods of the piezo sim which are simulated in one time step
    # in the heat conduction simulation
    SKIPPED_TIME_STEPS = 400000

    # Set heat conduction simulation settings
    heat_cond_delta_t = SKIPPED_TIME_STEPS*piezo_sim.simulation_data.delta_t
    number_of_time_steps = int(SIMULATION_TIME/heat_cond_delta_t)
    print("Total number of heat conduction time steps:", number_of_time_steps)

    heat_simulation_data = pfem.SimulationData(
        delta_t=heat_cond_delta_t,
        number_of_time_steps=number_of_time_steps,
        gamma=0.5,
        beta=0  # Doesnt matter in heat conduction sim
    )

    # Create heat conduction sim
    heat_sim = pfem.HeatConductionSim(
        piezo_sim.mesh_data,
        piezo_sim.material_manager,
        heat_simulation_data
    )

    # Set constant heat source
    import matplotlib.pyplot as plt
    plt.plot(piezo_sim.solver.mech_loss[342, :])
    plt.plot(piezo_sim.solver.mech_loss[566, :])
    plt.show()
    avg_mech_loss = np.mean(piezo_sim.solver.mech_loss[:, -100:], axis=1)
    heat_sim.set_constant_volume_heat_source(
        avg_mech_loss,
        number_of_time_steps,
    )

    # Get boundary elements for convection boundary condition
    # el = piezo_sim.gmsh_handler.get_elements_by_physical_groups(["Electrode", "Ground"])
    # boundary_elements = np.concatenate([el["Electrode"], el["Ground"]])

    # The start temperature field of the heat conduction sim is set to the
    # field of the last time step of the piezo sim
    number_of_nodes = len(piezo_sim.mesh_data.nodes)
    theta_start = u[3*number_of_nodes:, -1]

    # Simulate
    heat_sim.assemble()
    heat_sim.solve_time(theta_start)

    analyze_energies(
        piezo_sim,
        heat_sim,
        avg_mech_loss
    )

    # Save temperature only results
    np.save(
        os.path.join(piezo_sim_folder, f"{PIEZO_SIM_NAME}_theta_only.npy"),
        heat_sim.theta)
    gmsh_handler = pfem.GmshHandler(
        os.path.join(piezo_sim_folder, "temperature_only.msh")
    )
    gmsh_handler.create_theta_post_processing_view(
        heat_sim.theta,
        number_of_time_steps,
        heat_cond_delta_t,
        False
    )

    print_temperature_at_right_boundary(
        piezo_sim.mesh_data.nodes,
        heat_sim.theta
    )

    # Show the results in gmsh
    gmsh.fltk.run()
