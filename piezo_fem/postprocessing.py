"""Module for postprocessing functions."""

# Python standard libraries
from typing import Tuple, Union
import numpy as np
import numpy.typing as npt

# Local libraries
from piezo_fem.simulation.base import energy_integral_theta, \
    gradient_local_shape_functions

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
    current = np.gradient(charge, delta_t)

    return np.trapezoid(current*voltage_excitation, None, delta_t)


def calculate_stored_thermal_energy(
        theta: npt.NDArray,
        nodes: npt.NDArray,
        elements: npt.NDArray,
        heat_capacity: float,
        density: float) -> Union[float, npt.NDArray[float]]:
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
        The stored energy either as float or list of floats."""
    stored_energy = 0
    number_of_nodes = len(nodes)

    if len(theta.shape) == 2:
        # Need to calculate for every time step
        stored_energy = np.zeros(theta.shape[1])

        for time_index in range(theta.shape[1]):
            for element in elements:
                dn = gradient_local_shape_functions()
                node_points = np.array([
                    [nodes[element[0]][0],
                     nodes[element[1]][0],
                     nodes[element[2]][0]],
                    [nodes[element[0]][1],
                     nodes[element[1]][1],
                     nodes[element[2]][1]]
                ])
                jacobian = np.dot(node_points, dn.T)
                jacobian_det = np.linalg.det(jacobian)
                theta_e = np.array([
                    theta[element[0]+3*number_of_nodes, time_index],
                    theta[element[1]+3*number_of_nodes, time_index],
                    theta[element[2]+3*number_of_nodes, time_index]
                ])
                stored_energy[time_index] += energy_integral_theta(
                    node_points,
                    theta_e
                ) * 2 * np.pi * jacobian_det * heat_capacity * density

    elif len(theta.shape) == 1:
        # Only one time step
        stored_energy = 0

        for element in elements:
            dn = gradient_local_shape_functions()
            node_points = np.array([
                [nodes[element[0]][0],
                 nodes[element[1]][0],
                 nodes[element[2]][0]],
                [nodes[element[0]][1],
                 nodes[element[1]][1],
                 nodes[element[2]][1]]
            ])
            jacobian = np.dot(node_points, dn.T)
            jacobian_det = np.linalg.det(jacobian)
            theta_e = np.array([
                theta[element[0]],
                theta[element[1]],
                theta[element[2]]
            ])
            stored_energy+= energy_integral_theta(
                node_points,
                theta_e
            ) * 2 * np.pi * jacobian_det * heat_capacity * density

    return stored_energy
