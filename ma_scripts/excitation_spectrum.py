import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    number_of_time_steps = 8192
    delta_t = 1e-8
    excitation = np.zeros(number_of_time_steps)
    excitation[1:10] = np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])

    ex_fft = np.fft.fft(excitation)
    sample_frequency = 1/delta_t
    frequencies = np.arange(number_of_time_steps)*sample_frequency/(number_of_time_steps)

    time = np.arange(number_of_time_steps)*delta_t
    plt.plot(time, excitation)
    plt.xlabel(r"Time $t$ in $\mathrm{s}$")
    plt.grid()
    plt.show()
    plt.plot(frequencies, np.abs(ex_fft))
    plt.xlabel(r"Frequency $f$ in $\mathrm{Hz}$")
    plt.grid()
    plt.show()