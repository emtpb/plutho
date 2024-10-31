
# Python standard libraries
import os

# Third party libraries
import numpy as np
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Local libraries
import piezo_fem as pfem

if __name__ == "__main__":
    load_dotenv()

    CWD = os.getenv("piezo_fem_simulation_path")
    if CWD is None:
        print("Couldn't find simulation path.")
        exit(1)

    PIEZO_SIM_NAME = "real_model_30k"
    SIM_FOLDER = os.path.join(CWD, PIEZO_SIM_NAME)
    theta = np.load(os.path.join(SIM_FOLDER, f"{PIEZO_SIM_NAME}_theta.npy"))

    right_boundary_point_indices = [1, 53, 54, 55, 56, 57, 58, 59, 60, 61, 2]
    print(theta[right_boundary_point_indices, -1])
