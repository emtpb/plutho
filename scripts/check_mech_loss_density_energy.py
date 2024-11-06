
# Python standard libraries
import os
import numpy as np

# Third party libraries
from dotenv import load_dotenv

# Local libraries
import piezo_fem as pfem

def calculate_mech_loss_energy(
        mech_loss,
        nodes,
        elements,
        delta_t
    ):
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

    loss_energy = np.trapezoid(total_power, None, delta_t)
    return loss_energy


def compare_energies(sim: pfem.Simulation, max_duration: float):
    """Prints the input energy and the energy stored in the thermal field
    at the end of the simulation.

    Parameters:
        sim: Simulation object where the simulation is already done.
    """
    if isinstance(sim.solver, pfem.PiezoSimTherm):
        input_energy = pfem.calculate_electrical_input_energy(
            sim.excitation, sim.solver.q, sim.simulation_data.delta_t)
        thermal_energy = pfem.postprocessing.calculate_stored_thermal_energy(
            sim.solver.u[3*len(sim.mesh_data.nodes):, -1],
            sim.mesh_data.nodes,
            sim.mesh_data.elements,
            sim.material_data.heat_capacity,
            sim.material_data.density
        )

        max_factor = max_duration/ \
            (sim.simulation_data.delta_t
                * sim.simulation_data.number_of_time_steps)

        print("Input energy:", input_energy)
        print("Energy in thermal field at last time step:", thermal_energy)
        print(
            f"Predicted input energy after {max_duration} s",
            input_energy*max_factor
        )


if __name__ == "__main__":
    load_dotenv()

    CWD = os.getenv("piezo_fem_simulation_path")
    if CWD is None:
        print("Couldn't find simulation path.")
        exit(1)

    PIEZO_SIM_NAME = "real_model_20k_check_energy"

    piezo_sim_folder = os.path.join(CWD, PIEZO_SIM_NAME)
    piezo_sim = pfem.Simulation.load_simulation_settings(
        os.path.join(piezo_sim_folder, f"{PIEZO_SIM_NAME}.cfg")
    )
    piezo_sim.load_simulation_results()

    nodes = piezo_sim.mesh_data.nodes
    elements = piezo_sim.mesh_data.elements
    MAX_DURATION = 1
    PIEZO_SIM_DURATION = piezo_sim.simulation_data.delta_t*piezo_sim.simulation_data.number_of_time_steps
    mech_loss = piezo_sim.solver.mech_loss
    compare_energies(piezo_sim, MAX_DURATION)
    calc_mech_loss_energy = calculate_mech_loss_energy(
        mech_loss,
        nodes,
        elements,
        piezo_sim.simulation_data.delta_t
    )
    print("Calculated mech loss energy:", calc_mech_loss_energy)
    print("Predicted mech loss energy:", MAX_DURATION/PIEZO_SIM_DURATION*calc_mech_loss_energy)

    SKIPPED_TIME_STEPS = 40000
    heat_sim_delta_t = SKIPPED_TIME_STEPS*piezo_sim.simulation_data.delta_t
    heat_sim_number_of_time_steps = int(MAX_DURATION/heat_sim_delta_t)
    print(heat_sim_delta_t, heat_sim_number_of_time_steps)
    avg_stationary_losses = np.mean(mech_loss[:, -25:], axis=1)
    avg_stationary_losses = np.tile(avg_stationary_losses.reshape(-1, 1), (1, heat_sim_number_of_time_steps))
    interp_mech_loss_energy = calculate_mech_loss_energy(
        avg_stationary_losses,
        nodes,
        elements,
        heat_sim_delta_t
    )
    print("Interpolated mech loss energy:", interp_mech_loss_energy+)