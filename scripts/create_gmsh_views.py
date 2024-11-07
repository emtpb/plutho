"""Creates the post processing views of a simulation result."""

# Python standard libraries
import os

# Third party libraries
from dotenv import load_dotenv

# Local libraries
import piezo_fem as pfem

if __name__ == "__main__":
    load_dotenv()

    CWD = os.getenv("piezo_fem_simulation_path")
    if CWD is None:
        print("Couldn't find simulation path.")
        exit(1)

    SIMULATION_NAME = "energy_check_sinusoidal_20k_200kHz"

    sim = pfem.PiezoSimulation.load_simulation_settings(os.path.join(
        CWD,
        SIMULATION_NAME,
        f"{SIMULATION_NAME}.cfg"
    ))
    sim.load_simulation_results()
    sim.create_post_processing_views()
