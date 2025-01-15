"""Module to check the energy conversation of the FEM thermal
piezo simulation
"""

# Python standard libraries
import os
import numpy as np

# Third party libraries
from dotenv import load_dotenv

# Local libraries
import piezo_fem as pfem


def compare_loss_energies(sim: pfem.PiezoSimulation):
    """Prints the input energy and the loss energy of the simulation.

    Parameters:
        sim: Simulation object where the simulation is already done.
    """
    if isinstance(sim.solver, pfem.PiezoSimTherm):
        mech_loss = sim.solver.mech_loss

        number_of_time_steps = sim.solver.simulation_data.number_of_time_steps
        total_power = np.zeros(number_of_time_steps)
        elements = sim.mesh_data.elements
        nodes = sim.mesh_data.nodes

        # Calculate the volume of each element
        volumes = np.zeros(len(elements))
        for element_index, element in enumerate(elements):
            dn = pfem.simulation.base.gradient_local_shape_functions()
            node_points = np.array([
                [
                    nodes[element[0]][0],
                    nodes[element[1]][0],
                    nodes[element[2]][0]
                ],
                [
                    nodes[element[0]][1],
                    nodes[element[1]][1],
                    nodes[element[2]][1]
                ]
            ])
            jacobian = np.dot(node_points, dn.T)
            jacobian_det = np.linalg.det(jacobian)
            volumes[element_index] = (
                pfem.simulation.base.integral_volume(
                    node_points)
                * 2*np.pi*jacobian_det)

        for time_step in range(number_of_time_steps):
            total_power[time_step] = np.dot(mech_loss[:, time_step], volumes)

        excitation = sim.boundary_conditions[0]["values"]
        input_energy = pfem.calculate_electrical_input_energy(
            excitation, sim.solver.q, sim.solver.simulation_data.delta_t)

        loss_energy = np.trapezoid(
            total_power,
            None,
            sim.solver.simulation_data.delta_t
        )

        # Calculate thermal energy in last time step
        thermal_energy = pfem.postprocessing.calculate_stored_thermal_energy(
            sim.solver.u[3*len(nodes):, -1],
            sim.solver.mesh_data.nodes,
            sim.solver.mesh_data.elements,
            sim.material_manager.get_heat_capacity(0),
            sim.material_manager.get_density(0)
        )

        time_duration = sim.solver.simulation_data.delta_t * \
            sim.solver.simulation_data.number_of_time_steps

        max_duration = 1
        max_factor = max_duration/time_duration
        print("Time duration")

        print("Input energy:", input_energy)
        print("Loss energy:", loss_energy)
        print("Energy in thermal field at last time step:", thermal_energy)

        print("Input energy 1s:", input_energy*max_factor)


def model(working_directory, sim_name):
    """Real thermo piezoelectric simulation of a disc.

    Parameters:
        base_directory: Directory where the simulation directory is created.
    """
    mesh = pfem.Mesh(
        os.path.join(working_directory, "disc_mesh_0DOT0001.msh"),
        True
    )

    sim = pfem.SingleSimulation(
        working_directory,
        sim_name,
        mesh
    )

    # Simulation parameters
    number_of_time_steps = 20000
    delta_t = 1e-8

    sim.setup_thermal_piezo_time_domain(
        pfem.SimulationData(
            delta_t,
            number_of_time_steps,
            0.5,
            0.25
        )
    )
    sim.add_material(
        "pic255",
        pfem.pic255,
        None
    )

    time_steps = np.arange(number_of_time_steps)*delta_t
    #excitation = amplitude*np.sin(2*np.pi*frequency*time_steps)
    from scipy import signal
    _, exc = signal.gausspulse(time_steps-25*delta_t, fc=1/(10*delta_t), retenv=True)

    # excitation = np.zeros(number_of_time_steps)
    # excitation[1:10] = (
    #     1 * np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    # )

    sim.add_dirichlet_bc(
        pfem.FieldType.PHI,
        "Electrode",
        exc
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

    sim.simulate(electrode_elements=electrode_elements)
    sim.save_simulation_settings()
    sim.save_simulation_results()

    return sim


if __name__ == "__main__":
    load_dotenv()

    CWD = os.getenv("piezo_fem_simulation_path")
    if CWD is None:
        print("Couldn't find simulation path.")
        exit(1)
    MODEL_NAME = "energy_check_triangular_20k"
    SIM_FOLDER = os.path.join(
        CWD,
        MODEL_NAME
    )

    # model(CWD, MODEL_NAME)

    # Load data
    simulation = pfem.SingleSimulation.load_simulation_settings(SIM_FOLDER)
    simulation.load_simulation_results()

    compare_loss_energies(simulation)
