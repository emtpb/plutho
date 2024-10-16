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
        "Simulation to check the energy conservation.")
    sim.simulate()

    sim.save_simulation_results()
    sim.create_post_processing_views()

    return sim

if __name__ == "__main__":
    MODEL_NAME = "energy_check_eeds"
    CWD = os.path.join(
        "/upb/users/j/jonasho/scratch/piezo_fem/results/", MODEL_NAME
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

    if True:
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
    compare_energies(simulation)

    print(np.trapezoid(
        np.mean(simulation.solver.mech_loss, axis=0),
        None,
        simulation.simulation_data.delta_t))

    #plt.plot(np.mean(simulation.solver.mech_loss, axis=0), label="Mech loss")
    #plt.plot(simulation.solver.temp_field_energy, label="Temp field energy")
    #plt.show()
