"""Script for the simulation of a real piezo model."""

# Python standard libraries
import os

# Third party libraries
from dotenv import load_dotenv

# Local libraries
import piezo_fem as pfem


def real_model(base_directory):
    """Real thermo piezoelectric simulation of a disc.

    Parameters:
        base_directory: Directory where the simulation directory is created.
    """
    sim_name = "impedance_pic255_thesis"
    sim_directory = os.path.join(base_directory, sim_name)
    sim = pfem.PiezoSimulation(
        sim_directory,
        sim_name)
    sim.create_disc_mesh(0.005, 0.001, 0.0001)
    sim.set_material_data(
        "pic255",
        pfem.pic255,
    )
    sim.material_manager.material_data.alpha_m = 1.267e5
    sim.set_simulation(
        delta_t=1e-8,
        number_of_time_steps=8192,
        gamma=0.5,
        beta=0.25,
        simulation_type=pfem.SimulationType.PIEZOELECTRIC,
    )
    #sim.set_sinusoidal_excitation(20, 1.889e5)
    sim.set_triangle_pulse_excitation(1)
    sim.set_boundary_conditions()
    sim.save_simulation_settings(
        "Simulate for combined sim with anti-resonance frequency.")
    sim.simulate()

    sim.save_simulation_results()
    # sim.create_post_processing_views()


if __name__ == "__main__":
    load_dotenv()

    CWD = os.getenv("piezo_fem_simulation_path")
    if CWD is None:
        print("Couldn't find simulation path.")
        exit(1)

    real_model(CWD)
