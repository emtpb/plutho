"""Implements examples on how to use the simulation class of piezo_fem."""

# Python standard libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# Third party libraries
import gmsh

# Local libraries
import piezo_fem as pfem


def run_ring_simulation(base_directory):
    """Example for a piezoelectric ring simulation.

    Parameters:
        base_directory: Directory where the simulation directory is created.
    """
    sim_directory = os.path.join(base_directory, "ring_sim")
    sim = pfem.Simulation(sim_directory, pfem.pic255, "Ring sim")
    sim.create_ring_mesh(0.002, 0.005, 0.001, 0.00015)
    sim.set_simulation(
        delta_t=1e-8,
        number_of_time_steps=100,
        gamma=0.5,
        beta=0.25,
        simulation_type=pfem.SimulationType.PIEZOELECTRIC,
    )
    sim.set_triangle_pulse_excitation(1)
    sim.set_boundary_conditions()
    sim.save_simulation_settings(
        "An example for a piezo-electric simulation of a ring.")
    sim.simulate()
    sim.create_post_processing_views()
    # gmsh.fltk.run()


def run_disc_simulation(base_directory):
    """Example for a piezoelectric disc simulation.

    Parameters:
        base_directory: Directory where the simulation directory is created.
    """
    sim_directory = os.path.join(base_directory, "disc_sim")
    sim = pfem.Simulation(sim_directory, pfem.pic255, "Disc sim")
    sim.create_disc_mesh(0.005, 0.001, 0.00015)
    sim.set_simulation(
        delta_t=1e-8,
        number_of_time_steps=100,
        gamma=0.5,
        beta=0.25,
        simulation_type=pfem.SimulationType.PIEZOELECTRIC,
    )
    sim.set_triangle_pulse_excitation(1)
    sim.set_boundary_conditions()
    sim.save_simulation_settings(
        "An example for a piezo-electric simulation of a disc.")
    sim.simulate()
    sim.create_post_processing_views()
    # gmsh.fltk.run()


def run_thermal_simulation(base_directory, name):
    """Example for a thermo piezoelectric simulation of a disc.

    Parameters:
        base_directory: Directory where the simulation directory is created.
    """
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
    sim.set_triangle_pulse_excitation(1)
    sim.set_boundary_conditions()
    sim.save_simulation_settings(
        "An example for a thermal piezo-electric simulation.")
    sim.simulate()

    post_processing_temp(sim)


def real_model(base_directory):
    """Real thermo piezoelectric simulation of a disc.

    Parameters:
        base_directory: Directory where the simulation directory is created.
    """
    sim_directory = os.path.join(base_directory, "real_model_10k")
    sim = pfem.Simulation(sim_directory, pfem.pic255, "real_model_10k")
    sim.create_disc_mesh(0.005, 0.001, 0.0001)
    sim.set_simulation(
        delta_t=1e-8,
        number_of_time_steps=10000,
        gamma=0.5,
        beta=0.25,
        simulation_type=pfem.SimulationType.THERMOPIEZOELECTRIC,
    )
    sim.set_sinusoidal_excitation(1, 2e6)
    sim.set_boundary_conditions()
    sim.save_simulation_settings(
        "An example for a thermal piezo-electric simulation.")
    sim.simulate()

    post_processing_temp(sim)


def post_processing_temp(sim: pfem.Simulation):
    """Runs post processing for the given simulation object.

    Parameters:
        sim: Simulation object where the simulation is already done.
    """
    if sim.simulation_type is not pfem.SimulationType.THERMOPIEZOELECTRIC:
        raise TypeError("Need thermo-piezoelectric simulation for this post"
                        "processing function")
    sim.create_post_processing_views()
    sim.save_simulation_results()

    # Plot fields
    # gmsh.fltk.run()

    # compare_energies(sim)
    # plot_impedance_with_opencfs(sim, os.path.)


def compare_energies(sim: pfem.Simulation):
    """Prints the input energy and the energy stored in the thermal field
    at the end of the simulation.

    Parameters:
        sim: Simulation object where the simulation is already done.
    """
    input_energy = pfem.calculate_electrical_input_energy(
        sim.excitation, sim.solver.qq, sim.simulation_data.delta_t)
    thermal_energy = sim.solver.temp_field_energy

    print("Input energy:", input_energy)
    print("Energy in thermal field at last time step:", thermal_energy[-1])


def plot_impedance_with_opencfs(sim: pfem.Simulation, open_cfs_hist_file: str):
    """Calculates and plots the impedence curves of the given FEM simulation
    together with OpenCFS results.

    Parameters:
        sim: Simulation object where the simulation was already done.
        open_cfs_hist_file: Hist file cotaining the charge from opencfs
            simulation.
    """
    delta_t = sim.simulation_data.delta_t
    excitation = sim.excitation

    frequencies_fem, impedence_fem = pfem.calculate_impedance(
        sim.solver.q, excitation, delta_t)

    # Get OpenCFS data
    _, charge_cfs = pfem.parse_charge_hist_file(open_cfs_hist_file)
    frequencies_cfs, impedence_cfs = pfem.calculate_impedance(
        charge_cfs, excitation, delta_t)

    # Plot FEM and OpenCfs
    plt.plot(frequencies_fem, np.abs(impedence_fem), label="MyFEM")
    plt.plot(frequencies_cfs, np.abs(impedence_cfs), "+", label="OpenCFS")
    plt.xlabel("Frequency f / Hz")
    plt.ylabel("Impedence |Z| / $\\Omega$")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    #cwd = os.path.join(os.path.abspath(os.path.dirname(__file__)),
    #                   "simulations")
    cwd_scratch = "/upb/users/j/jonasho/scratch/piezo_fem/results/"
    real_model(cwd_scratch)
    # run_disc_simulation(cwd)
    # run_ring_simulation(cwd)
    # run_thermal_simulation(cwd, "real_model")
