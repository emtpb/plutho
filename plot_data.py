import numpy as np
import matplotlib.pyplot as plt

def calculate_impedance(q, excitation, DELTA_T):
    sample_frequency = 1/DELTA_T

    # If size is not power of 2 append zeros
    #for i in range(10):
    #    if len(q) < 2**i:
    #        np.append(q, np.zeros(2**i-len(q))) 
    #        np.append(excitation, np.zeros(2**i-len(q)))
    #        break
        
    excitation_fft = np.fft.fft(excitation)[1:]
    q_fft = np.fft.fft(q)[1:]

    # since f=k*sample_frequency/N
    frequencies = np.arange(1, len(q))*sample_frequency/len(q)
    impedance = excitation_fft/(2*np.pi*1j*frequencies*q_fft)

    plt.plot(frequencies, np.abs(impedance), label="|Z(w)|")
    plt.plot(frequencies, np.abs(excitation_fft), label="|V(w)|")
    plt.plot(frequencies, np.abs(q_fft), label="|Q(w)|")
    plt.xlabel("Frequency f / Hz")
    plt.ylabel("Impedence |Z| / $\Omega$")
    plt.yscale("log")
    #plt.xscale("log")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    charge = np.load("charge.npy")
    #charge = np.load("charge_full_sim.npy")

    #print(charge)

    #NUMBER_TIME_STEPS = int(1/4*8192)
    NUMBER_TIME_STEPS = int(0.5*8192)
    #NUMBER_TIME_STEPS = 1000
    #DELTA_T = 1e-9
    DELTA_T = 1e-8
    time_list = np.arange(0, NUMBER_TIME_STEPS)*DELTA_T

    excitation = np.zeros(NUMBER_TIME_STEPS)
    excitation[1:10] = np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])

    plt.plot(time_list, charge/np.max(charge), "+", label="Charge")
    plt.plot(time_list, excitation/np.max(excitation), label="Excitation")
    plt.grid()
    plt.legend()
    plt.show()

    calculate_impedance(charge, excitation, DELTA_T)