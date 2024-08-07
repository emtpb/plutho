import numpy as np
import matplotlib.pyplot as plt

def calculate_impedance(q, excitation, DELTA_T):

    # If size is not power of 2 append zeros
    #for i in range(10):
    #    if len(q) < 2**i:
    #        np.append(q, np.zeros(2**i-len(q))) 
    #        np.append(excitation, np.zeros(2**i-len(q)))
    #        break
        
    excitation_fft = np.fft.fft(excitation)[1:]
    q_fft = np.fft.fft(q)[1:]

    # since f=k*sample_frequency/N
    sample_frequency = 1/DELTA_T
    frequencies = np.arange(1, len(q))*sample_frequency/len(q)
    impedance = excitation_fft/(2*np.pi*1j*frequencies*q_fft)

    return frequencies, impedance

def read_charge_open_cfs(file_path):
    lines = []
    with open(file_path, "r", encoding="UTF-8") as fd:
        lines = fd.readlines()[3:]

    time = []
    charge = []
    for time_index, line in enumerate(lines):
        current_time, current_charge = line.split()
        time.append(float(current_time))
        charge.append(float(current_charge))

    return time, charge

if __name__ == "__main__":
    # Get MyFEM Charge
    charge_fem = np.load("charge.npy")
    NUMBER_TIME_STEPS = 8192
    DELTA_T = 1e-8
    time_list = np.arange(0, NUMBER_TIME_STEPS)*DELTA_T

    excitation = np.zeros(NUMBER_TIME_STEPS)
    excitation[1:10] = np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])

    frequencie_fem, impedence_fem = calculate_impedance(charge_fem, excitation, DELTA_T)

    # Get CFS Charge
    time_list_cfs, charge_cfs = read_charge_open_cfs("piezo-elecCharge-surfRegion-ground.hist")

    #plt.plot(time_list_cfs[:NUMBER_TIME_STEPS], charge_cfs[:NUMBER_TIME_STEPS], "+", label="OpenCFS")
    #plt.plot(time_list, charge_fem, label="MyFEM")

    frequencies_cfs, impedence_cfs = calculate_impedance(charge_cfs, excitation, DELTA_T)

    plt.plot(frequencie_fem, np.abs(impedence_fem), label="MyFEM")
    plt.plot(frequencies_cfs, np.abs(impedence_cfs), "+", label="OpenCFS")
    plt.xlabel("Frequency f / Hz")
    plt.ylabel("Impedence |Z| / $\\Omega$")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()