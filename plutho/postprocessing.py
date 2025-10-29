"""Module for postprocessing functions."""

# Python standard libraries
from typing import Tuple, Union
import numpy as np
import numpy.typing as npt

# Local libraries
from .simulations.integrals import energy_integral_theta


__all__ = [
    "calculate_impedance",
    "calculate_electrical_input_energy",
    "calculate_stored_thermal_energy"
]


def calculate_impedance(
    q: npt.NDArray,
    excitation: npt.NDArray,
    delta_t: float
) -> Tuple[npt.NDArray, npt.NDArray]:
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
    delta_t: float
) -> float:
    """Calculates the energy of the elctric input.

    Parameters:
        voltage_excitation: Voltage signal used as excitation.
        charge: Charge calculated in the simulation using the same
            excitation.
        delta_t: Time difference also used in the simulation.

    Returns:
        Electric input energy.
    """
    current = np.gradient(charge, delta_t)

    return np.trapezoid(current*voltage_excitation, None, delta_t)


def calculate_stored_thermal_energy(
    theta: npt.NDArray,
    nodes: npt.NDArray,
    elements: npt.NDArray,
    element_order: int,
    heat_capacity: float,
    density: float
) -> Union[float, npt.NDArray]:
    """Calculates the stored thermal energy from the given field.
    theta can have 2 different formats depending on if its time dependent:
        theta[node_index, time_index] or
        theta[node_index].

    Parameters:
        theta: Temperature field values per node (and per time if necessary).
        nodes: Nodes of the mesh.
        elements: Elements of the mesh.
        heat_capacity: Heat capacity of the model.
        density: Density of the model.

    Returns:
        The stored energy either as float or list of floats.
    """
    points_per_element = int(1/2*(element_order+1)*(element_order+2))

    if len(theta.shape) == 2:
        # Need to calculate for every time step
        stored_energies = np.zeros(theta.shape[1])

        for time_index in range(theta.shape[1]):
            for _, element in enumerate(elements):
                node_points = np.zeros(shape=(2, points_per_element))
                for node_index in range(points_per_element):
                    node_points[:, node_index] = [
                        nodes[element[node_index]][0],
                        nodes[element[node_index]][1]
                    ]

                theta_e = np.zeros(points_per_element)
                for node_index in range(points_per_element):
                    theta_e[node_index] = theta[
                        element[node_index],
                        time_index
                    ]

                stored_energies[time_index] += energy_integral_theta(
                    node_points,
                    theta_e,
                    element_order
                ) * 2 * np.pi * heat_capacity * density

        return stored_energies

    elif len(theta.shape) == 1:
        # Only one time step
        stored_energy = 0

        for _, element in enumerate(elements):
            node_points = np.zeros(shape=(2, points_per_element))
            for node_index in range(points_per_element):
                node_points[:, node_index] = [
                    nodes[element[node_index]][0],
                    nodes[element[node_index]][1]
                ]

            theta_e = np.zeros(points_per_element)
            for node_index in range(points_per_element):
                theta_e[node_index] = theta[element[node_index]]

            stored_energy += energy_integral_theta(
                node_points,
                theta_e,
                element_order
            ) * 2 * np.pi * heat_capacity * density

        return stored_energy

    raise ValueError(
        "Cannot calculate total stored energy for given "
        f"{len(theta.shape)}-dimensional theta"
    )
