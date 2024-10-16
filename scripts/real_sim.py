"""Script for the simulation of a real piezo model."""

# Python standard libraries
import os

# Local libraries
import piezo_fem as pfem


def real_model(base_directory):
    """Real thermo piezoelectric simulation of a disc.

    Parameters:
        base_directory: Directory where the simulation directory is created.
    """
    sim_name = "real_model_15k"
    sim_directory = os.path.join(base_directory, sim_name)
    sim = pfem.Simulation(
        sim_directory,
        pfem.pic255,
        sim_name)
    sim.create_disc_mesh(0.005, 0.001, 0.0001)
    sim.set_simulation(
        delta_t=1e-8,
        number_of_time_steps=15000,
        gamma=0.5,
        beta=0.25,
        simulation_type=pfem.SimulationType.THERMOPIEZOELECTRIC,
    )
    sim.set_sinusoidal_excitation(1, 2e6)
    #sim.set_triangle_pulse_excitation(1)
    sim.set_boundary_conditions()
    sim.save_simulation_settings(
        "Simulation for the thermal field and losses used in a thermal sim.")
    sim.simulate()

    sim.save_simulation_results()
    sim.create_post_processing_views()


if __name__ == "__main__":
    #CWD_SCRATCH = "/home/jonash/uni/Masterarbeit/simulations/"
    CWD_SCRATCH = "/upb/users/j/jonasho/scratch/piezo_fem/results/"
    real_model(CWD_SCRATCH)
