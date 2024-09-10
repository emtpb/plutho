import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    file = "theta.npy"
    theta = np.load(file)

    TIME_STEP_COUNT = 50
    DELTA_T = 1e-8
    
    for time_index in range(TIME_STEP_COUNT):
        print(time_index, np.where(theta[:, time_index]< 0))