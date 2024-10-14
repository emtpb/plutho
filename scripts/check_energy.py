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


if __name__ == "__main__":
    MODEL_NAME = "real_model_30k_triangle"
    #CWD = os.path.join(
    #    "/upb/users/j/jonasho/scratch/piezo_fem/results/", MODEL_NAME)
    CWD = os.path.join(
        "/home/jonash/uni/Masterarbeit/simulations/", MODEL_NAME
    )
    TEMP_ENERGY_FILE_PATH = os.path.join(
        CWD, f"{MODEL_NAME}_temp_field_energy.npy"
    )
    CHARGE_FILE_PATH = os.path.join(
        CWD, f"{MODEL_NAME}_q.npy"
    )
    DISPLACEMENT_FILE_PATH = os.path.join(
        CWD, f"{MODEL_NAME}_u.npy"
    )
    CONFIG_FILE_PATH = os.path.join(CWD, f"{MODEL_NAME}.cfg")

    simulation = pfem.Simulation.load_simulation_settings(CONFIG_FILE_PATH)
    simulation.solver.q = np.load(CHARGE_FILE_PATH)
    simulation.solver.temp_field_energy = np.load(TEMP_ENERGY_FILE_PATH)
    simulation.solver.u = np.load(DISPLACEMENT_FILE_PATH)

    # get_max_temp_value(simulation)
    compare_energies(simulation)
