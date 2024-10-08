"""Script for a thermal simulation after a thermo-piezo simulation
in order to calculate the Temperature for longer time intervals."""

# Python standard libraries
import os
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

# Third party libraries
import gmsh

# Local libraries
import piezo_fem as pfem


def calculate_avg_losses_per_node(
        mech_loss: npt.NDArray,
        number_of_nodes: float,
        frequency: float,
        delta_t: float):
    """Calculates the average losses per node. Does only work properly for
    sinusoidal excitation"""
    # TODO 2 because the power has double frequency??
    time_steps_per_period = int(1/(2*frequency*delta_t))

    avg_losses = np.zeros(number_of_nodes)
    for node_index in range(number_of_nodes):
        avg_losses[node_index] = np.mean(
            mech_loss[node_index, -time_steps_per_period:])

    return avg_losses


def run_thermal_simulation(base_directory, name):
    """Example for a thermo piezoelectric simulation of a disc.

    Parameters:
        base_directory: Directory where the simulation directory is created.
    """
    delta_t = 1e-8
    frequency = 2e6

    sim_directory = os.path.join(base_directory, name)
    sim = pfem.Simulation(sim_directory, pfem.pic255, name)
    sim.create_disc_mesh(0.005, 0.001, 0.00015)
    sim.set_simulation(
        delta_t=1e-8,
        number_of_time_steps=100,
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

    return sim.solver.u, calculate_avg_losses_per_node(
        sim.solver.mech_loss,
        len(sim.mesh_data.nodes),
        frequency,
        delta_t
    )


if __name__ == "_main__":
    cwd = "/home/jonash/uni/Masterarbeit/simulations/"

    u, avg_mech_losses = run_thermal_simulation(cwd, "double_sim")
