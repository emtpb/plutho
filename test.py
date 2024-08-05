import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    DELTA_T = 1e-9
    N = 8000
    excitation = np.zeros(N)
    excitation[1:10] = [0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2]

    time_values = np.arange(N)*DELTA_T

    #plt.plot(excitation)
    #plt.grid()
    #plt.show()

    excitation_fft = np.fft.fft(excitation)
    frequency_values = np.arange(N)*1/DELTA_T/N

    plt.plot(frequency_values, np.abs(excitation_fft))
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.show()
