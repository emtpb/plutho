"""Module for base functionalities needed for the simulations."""

# Python standard libraries
from typing import Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import numpy.typing as npt


class ModelType(Enum):
    """Containts the model type. Since for different model types
    different boundary conditions are needed it is necessary to make a
    distinction.
    Additionaly the Ring model type needs an appropiate mesh set separately
    (using x_offset in the mesh generation).
    """
    DISC = "disc"
    RING = "ring"


class SimulationType(Enum):
    """Contains the simulation type. Currently it is possible to have
    a simulation with or without thermal field."""
    PIEZOELECTRIC = "piezoelectric"
    THERMOPIEZOELECTRIC = "thermo-piezoelectric"


@dataclass
class MaterialData:
    """Contains material data used for the simulation."""
    elasticity_matrix: npt.NDArray
    permittivity_matrix: npt.NDArray
    piezo_matrix: npt.NDArray
    density: float
    thermal_conductivity: float
    heat_capacity: float
    alpha_m: float
    alpha_k: float
    name: str


@dataclass
class SimulationData:
    """Contains data for the simulation itself."""
    delta_t: float
    number_of_time_steps: float
    gamma: float
    beta: float


@dataclass
class MeshData:
    """Contains the mesh data is used in the simulation."""
    nodes: npt.NDArray
    elements: npt.NDArray


class ExcitationType(Enum):
    """Sets the excitation type of the simulation."""
    SINUSOIDAL = "sinusoidal"
    TRIANGULAR_PULSE = "triangular_pulse"


@dataclass
class ExcitationInfo:
    """Contains information about the excitation. Is used to save the
    excitation data in the simulation config file."""
    amplitude: float
    frequency: float
    excitation_type: ExcitationType

    def asdict(self):
        """Returns this object as a dictionary."""
        content = self.__dict__
        if self.frequency is None:
            del content["frequency"]
        content["excitation_type"] = self.excitation_type.value
        return content


def local_shape_functions(s: float, t: float) -> npt.NDArray:
    """Returns the local linear shape functions for the reference triangle.

    Parameters:
        s: Lcoal coordinate.
        t: Local coordinate.

    Returns:
        npt.NDArray: Shape functions depending on s and t for every point
            of the triangle.
    """
    return np.array([1-s-t, s, t])


def gradient_local_shape_functions():
    """Returns the gradient of the local shape functions.

    Returns:
        npt.NDArray: Gradient of each of the shape functions where first list
            is erived by s and second by t.
    """
    return np.array([[-1, 1, 0],
                     [-1, 0, 1]])


def local_to_global_coordinates(
        node_points: npt.NDArray,
        s: float,
        t: float) -> npt.NDArray:
    """Transforms the local coordinates s, t using the node points to
    the global coordinates r, z.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        s: Local coordinate.
        t: Local coordinate.

    Returns:
        npt.NDArray: Global coordinates [r, z]
    """
    return np.dot(node_points, local_shape_functions(s, t))


def b_operator_global(
        node_points: npt.NDArray,
        s: float,
        t: float,
        jacobian_inverted_t: npt.NDArray) -> npt.NDArray:
    """Calculates the B operator for the local coordinantes which is needed
    for voigt-notation.
    The derivates are with respect to the global coordinates r and z.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        s: Local coordinate.
        t: Local coordinate.
        jacobian_inverted_t: Jacobian matrix inverted and transposed, needed
            for calculation of global derivatives.

    Returns:
        B operator 6x4, for a u aligned like [u1_r, u1_z, u2_r, u2_z, ..].
    """
    # Get local shape functions and r (because of theta component)
    n = local_shape_functions(s, t)
    r = local_to_global_coordinates(node_points, s, t)[0]

    # Get gradients of local shape functions (s, t)
    dn = gradient_local_shape_functions()

    # Convert to gradients of (r, z) using jacobi matrix
    global_dn = np.dot(jacobian_inverted_t, dn)

    return np.array([
        [
            global_dn[0][0], 0, global_dn[0][1], 0, global_dn[0][2], 0
        ],
        [
            0, global_dn[1][0], 0, global_dn[1][1], 0, global_dn[1][2]
        ],
        [
            global_dn[1][0], global_dn[0][0], global_dn[1][1],
            global_dn[0][1], global_dn[1][2], global_dn[0][2]
        ],
        [
            n[0]/r, 0, n[1]/r, 0, n[2]/r, 0
        ],
    ])


def integral_m(node_points: npt.NDArray) -> npt.NDArray:
    """Calculates the M integral.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.

    Returns:
        npt.NDArray: 3x3 M matrix for the given element.
    """
    def inner(s, t):
        n = local_shape_functions(s, t)

        # Since the simulation is axisymmetric it is necessary
        # to multiply with the radius in the integral
        # (for the theta component (azimuth))
        r = local_to_global_coordinates(node_points, s, t)[0]

        # Get all combinations of shape function multiplied with each other
        return np.outer(n, n)*r

    return quadratic_quadrature(inner)


def integral_ku(
        node_points: npt.NDArray,
        jacobian_inverted_t: npt.NDArray,
        elasticity_matrix: npt.NDArray) -> npt.NDArray:
    """Calculates the Ku integral

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        jacobian_inverted_t: Jacobian matrix inverted and transposed, needed
            for calculation of global derivatives.
        elasticity_matrix: Elasticity matrix for the current element
            (c matrix).

    Returns:
        npt.NDArray: 6x6 Ku matrix for the given element.
    """
    def inner(s, t):
        b_op = b_operator_global(node_points, s, t, jacobian_inverted_t)
        r = local_to_global_coordinates(node_points, s, t)[0]

        return np.dot(np.dot(b_op.T, elasticity_matrix), b_op)*r

    return quadratic_quadrature(inner)


def integral_kuv(
        node_points: npt.NDArray,
        jacobian_inverted_t: npt.NDArray,
        piezo_matrix: npt.NDArray) -> npt.NDArray:
    """Calculates the KuV integral.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        jacobian_inverted_t: Jacobian matrix inverted and transposed, needed
            for calculation of global derivatives.
        piezo_matrix: Piezo matrix for the current element (e matrix).

    Returns:
        npt.NDArray: 6x3 KuV matrix for the given element.
    """
    def inner(s, t):
        b_op = b_operator_global(node_points, s, t, jacobian_inverted_t)
        dn = gradient_local_shape_functions()
        global_dn = np.dot(jacobian_inverted_t, dn)
        r = local_to_global_coordinates(node_points, s, t)[0]

        return np.dot(np.dot(b_op.T, piezo_matrix.T), global_dn)*r

    return quadratic_quadrature(inner)


def integral_kve(
        node_points: npt.NDArray,
        jacobian_inverted_t: npt.NDArray,
        permittivity_matrix: npt.NDArray) -> npt.NDArray:
    """Calculates the KVe integral.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        jacobian_inverted_t: Jacobian matrix inverted and transposed, needed
            for calculation of global derivatives.
        permittivity_matrix: Permittivity matrix for the
            current element (epsilon matrix).

    Returns:
        npt.NDArray: 3x3 KVe matrix for the given element.
    """
    def inner(s, t):
        dn = gradient_local_shape_functions()
        global_dn = np.dot(jacobian_inverted_t, dn)
        r = local_to_global_coordinates(node_points, s, t)[0]

        return np.dot(np.dot(global_dn.T, permittivity_matrix), global_dn)*r

    return quadratic_quadrature(inner)


def quadratic_quadrature(func: callable) -> npt.NDArray:
    """Integrates the given function of 2 variables using gaussian
    quadrature along 2 variables for a reference triangle.
    This gives exact results for linear shape functions.

    Parameters:
        func: Function which will be integrated.

    Returns:
        Integral of the given function.
    """
    weight_1 = 1/6
    weight_2 = 2/3
    return func(weight_1, weight_1)*weight_1 + \
        func(weight_2, weight_1)*weight_1 + \
        func(weight_1, weight_2)*weight_1


def line_quadrature(func):
    """Integrates the given function of 2 variables along one variable
    for a reference triangle.
    This gives exact results for linear shape functions.

    Parameters:
        func: Function which will be integrated along r-axis

    Returns:
        Integral of the given function along r-axis"""
    return 0.5*(func(-1/2/np.sqrt(3)+1/2, 0) + func(1/2/np.sqrt(3)+1/2, 0))


def apply_dirichlet_bc(
        m: npt.NDArray,
        c: npt.NDArray,
        k: npt.NDArray,
        nodes_u: npt.NDArray,
        nodes_v: npt.NDArray,
        number_of_nodes: int) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Prepares the given matrices m, c and k for the dirichlet boundary
    conditions. This is done by setting the corresponding rows to 0
    excepct for the node which will contain the specific value (this is set
    to 1).

    Parameters:
        m: Mass matrix M.
        c: Damping matrix C.
        k: Stiffness matrix K.
        nodes_u: List of node indices of dirichlet nodes of the displacement
            field (u_r and u_z).
        nodes_v: List of node indices of the dirichlet nodes of the electric
            field (electric potential).
        number_of_nodes: Total number of nodes of the simulation.
    Returns:
        Modified mass, damping and stiffness matrix.
    """
    # Set rows of matrices to 0 and diagonal of K to 1 (at node points)

    # Matrices for u_r component
    for node in nodes_u:
        # Set rows to 0
        m[2*node, :] = 0
        c[2*node, :] = 0
        k[2*node, :] = 0

        # Set diagonal values to 1
        k[2*node, 2*node] = 1

    # Set bc for v
    # Offset because the V values are set in the latter part of the matrix
    offset = 2*number_of_nodes

    for node in nodes_v:
        # Set rows to 0
        m[node+offset, :] = 0
        c[node+offset, :] = 0
        k[node+offset, :] = 0

        # Set diagonal values to 1
        k[node+offset, node+offset] = 1

    # Currently no dirichlet bc for the temperature field -> TODO Instable?

    return m, c, k


def get_dirichlet_boundary_conditions(
        gmsh_handler,
        electrode_excitation: npt.NDArray,
        number_of_time_steps: float,
        set_symmetric_bc: bool = True):
    """Sets the dirichlet boundary condition for the simulation
    given the nodes of electrode, symaxis and ground. The electrode
    nodes are set to the given electrode_excitation.
    The symaxis nodes are the due to the axisymmetric model
    and the ground nodes are set to 0.

    Parameters:
        electrode_nodes: Nodes in the electrode region.
        symaxis_nodes: Nodes on the symmetrical axis (r=0).
        ground_nodes: Nodes in the ground region.
        electrode_excitation: Excitation values for each time step.
        number_of_time_steps: Total number of time steps of the simulation.
    """
    # Get nodes from gmsh handler
    pg_nodes = gmsh_handler.get_nodes_by_physical_groups(
        ["Electrode", "Symaxis", "Ground"])

    electrode_nodes = pg_nodes["Electrode"]
    symaxis_nodes = pg_nodes["Symaxis"]
    ground_nodes = pg_nodes["Ground"]

    # For "Electrode" set excitation function
    # "Symaxis" and "Ground" are set to 0
    # For displacement u set symaxis values to 0.
    # Zeros are set for u_r and u_z but the u_z component is not used.
    if set_symmetric_bc:
        dirichlet_nodes_u = symaxis_nodes
        dirichlet_values_u = np.zeros(
            (number_of_time_steps, len(dirichlet_nodes_u), 2))
    else:
        dirichlet_nodes_u = np.array([])
        dirichlet_values_u = np.array([])

    # For potential v set electrode to excitation and ground to 0
    dirichlet_nodes_v = np.concatenate((electrode_nodes, ground_nodes))
    dirichlet_values_v = np.zeros(
        (number_of_time_steps, len(dirichlet_nodes_v)))

    # Set excitation value for electrode nodes points
    for time_index, excitation_value in enumerate(electrode_excitation):
        dirichlet_values_v[time_index, :len(electrode_nodes)] = \
            excitation_value

    return [dirichlet_nodes_u, dirichlet_nodes_v], \
        [dirichlet_values_u, dirichlet_values_v]
