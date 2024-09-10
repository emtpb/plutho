import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    file = "ScS.npy"
    ScS = np.load(file)

    TIME_STEP_COUNT = 50
    DELTA_T = 1e-8

    times = np.arange(0, TIME_STEP_COUNT)*DELTA_T
    dtt_Scs = np.zeros(TIME_STEP_COUNT)
    for time_index, time in enumerate(times):
        if time_index > 2:
            dtt_Scs[time_index] = (2*ScS[10, time_index]-5*ScS[10, time_index-1]+4*ScS[10, time_index-2]-ScS[10, time_index-3])/DELTA_T**2

    power_loss = np.load(file)
    print(power_loss.shape)

    plt.plot(times, ScS[10], "+", label="ScS")
    #plt.plot(times, dtt_Scs, label="dtt manual")
    plt.plot(times, np.gradient(np.gradient(ScS[10])), label="dtt")
    plt.plot(times, power_loss[10], label="Power loss")
    plt.legend()
    plt.grid()
    plt.show()