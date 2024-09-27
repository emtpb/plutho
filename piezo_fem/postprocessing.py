"""Module for postprocessing functions."""

# Python standard libraries
from typing import Tuple
import numpy as np
import numpy.typing as npt


def calculate_impedance(
        q: npt.NDArray,
        excitation: npt.NDArray,
        delta_t: float) -> Tuple[npt.NDArray, npt.NDArray]:
    """Calculates the impedance given the charge q and the excitation
    per time step.

    Paramters:
        q: Charge per time step.
        excitation: Excitation voltage per time step.
        delta_t: Difference between each time step.
    """
    excitation_fft = np.fft.fft(excitation)[1:]
    q_fft = np.fft.fft(q)[1:]

    # since f=k*sample_frequency/N
    sample_frequency = 1/delta_t
    frequencies = np.arange(1, len(q))*sample_frequency/len(q)
    impedance = excitation_fft/(2*np.pi*1j*frequencies*q_fft)

    return frequencies, impedance


def calculate_electrical_input_energy(
        voltage_excitation: npt.NDArray,
        charge: npt.NDArray,
        delta_t: float) -> float:
    """Calculates the energy of the elctric input.

    Parameters:
        voltage_excitation: Voltage signal used as excitation.
        charge: Charge calculated in the simulation using the same
            excitation.
        delta_t: Time difference also used in the simulation.

    Returns:
        Electric input energy."""
    current = np.gradient(charge)

    return np.trapezoid(current*voltage_excitation, None, delta_t)
