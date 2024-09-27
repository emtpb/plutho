"""Implements examples on how to use the simulation class of piezo_fem."""
import os
import piezo_fem as pfem
import gmsh


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


def run_thermal_simulation(base_directory):
    """Example for a thermo piezoelectric simulation of a disc.

    Parameters:
        base_directory: Directory where the simulation directory is created.
    """
    sim_directory = os.path.join(base_directory, "thermal_sim")
    sim = pfem.Simulation(sim_directory, pfem.pic255, "Thermal sim")
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
    sim.create_post_processing_views()
    # gmsh.fltk.run()


if __name__ == "__main__":
    cwd = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                       "simulations")
    # run_disc_simulation(cwd)
    # run_ring_simulation(cwd)
    run_thermal_simulation(cwd)
