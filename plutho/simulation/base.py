"""Module for base functionalities needed for the simulations."""

# Python standard libraries
import os
import json
from typing import Tuple, Callable, List, Any, Union
from dataclasses import dataclass, fields
from enum import Enum
import numpy as np
import numpy.typing as npt
from scipy import sparse

# Local libraries
from ..mesh import Mesh

# -------- ENUMS AND DATACLASSES --------


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
    FREQPIEZOELECTRIC = "frequency-piezoelectric"


@dataclass
class SimulationData:
    """Contains data for the simulation itself."""
    delta_t: float
    number_of_time_steps: int
    gamma: float
    beta: float


@dataclass
class MeshData:
    """Contains the mesh data is used in the simulation."""
    nodes: npt.NDArray
    elements: npt.NDArray
    element_order: int


class ExcitationType(Enum):
    """Sets the excitation type of the simulation."""
    SINUSOIDAL = "sinusoidal"
    TRIANGULAR_PULSE = "triangular_pulse"


@dataclass
class ExcitationInfo:
    """Contains information about the excitation. Is used to save the
    excitation data in the simulation config file."""
    amplitude: Union[float, npt.NDArray]
    frequency: Union[float, npt.NDArray]
    excitation_type: ExcitationType

    def asdict(self):
        """Returns this object as a dictionary."""
        content = self.__dict__
        if self.frequency is None:
            del content["frequency"]
        content["excitation_type"] = self.excitation_type.value
        return content


@dataclass
class MaterialData:
    """Contains the plain material data. Some parameters can either be
    a float or an array depending if they are temperature dependent.
    If they are temperature dependent the index in the array corresponds to
    the temperature value from the temperatures array at the same index.
    """
    c11: Union[float, npt.NDArray]
    c12: Union[float, npt.NDArray]
    c13: Union[float, npt.NDArray]
    c33: Union[float, npt.NDArray]
    c44: Union[float, npt.NDArray]
    e15: Union[float, npt.NDArray]
    e31: Union[float, npt.NDArray]
    e33: Union[float, npt.NDArray]
    eps11: Union[float, npt.NDArray]
    eps33: Union[float, npt.NDArray]
    alpha_m: float
    alpha_k: float
    thermal_conductivity: float
    heat_capacity: float
    temperatures: Union[float, npt.NDArray]
    density: float

    def to_dict(self):
        """Convert the dataclass to dict for json serialization."""
        json_dict = {}
        for attribute in fields(self.__class__):
            value = getattr(self, attribute.name)
            if isinstance(value, float) or isinstance(value, int):
                json_dict[attribute.name] = value
            elif isinstance(value, np.ndarray):
                json_dict[attribute.name] = value.tolist()
            else:
                raise ValueError(
                    "Wrong type saved in MaterialData. Value is of type "
                    f"{type(value)}"
                )
        return json_dict

    @staticmethod
    def from_dict(contents):
        """Convert given dict, e.g. from a json deserialization, to a
        MaterialData object."""
        for key, value in contents.items():
            if isinstance(value, List):
                contents[key] = np.array(value)

        return MaterialData(**contents)

    @staticmethod
    def load_from_file(file_path: str):
        """Load the data from given file.

        Parameters:
            file_path: Path to the file
        """
        if not os.path.exists(file_path):
            raise IOError(
                "Given file path {} does not exist. Cannot load "
                "material data."
            )

        with open(file_path, "r", encoding="UTF-8") as fd:
            return MaterialData.from_dict(json.load(fd))


# -------- Local functions and integrals --------


def local_shape_functions_2d(s, t, element_order):
    """Returns the local linear shape functions based on a reference triangle
    with corner points [(0,0), (1,0), (1,1)] for the given coordinates.

    Parameters:
        s: Local s parameter.
        t: Local t parameter.
        element_order: Order of the shape functions (1=linear, 2=quadratic,
            3=cubic).

    Returns:
        npt.NDArray: Shape functions depending on the given parameters for a
            line or a triangle
    """
    L1 = 1-s-t
    L2 = s
    L3 = t

    match element_order:
        case 1:
            return np.array([L1, L2, L3])
        case 2:
            return np.array(
                [
                    L1*(2*L1-1),
                    L2*(2*L2-1),
                    L3*(2*L3-1),
                    4*L1*L2,
                    4*L2*L3,
                    4*L3*L1
                ]
            )
        case 3:
            return np.array(
                [
                    0.5*L1*(3*L1-1)*(3*L1-2),
                    0.5*L2*(3*L2-1)*(3*L2-2),
                    0.5*L3*(3*L3-1)*(3*L3-2),
                    9/2*L1*L2*(3*L1-1),
                    9/2*L2*L1*(3*L2-1),
                    9/2*L2*L3*(3*L2-1),
                    9/2*L3*L2*(3*L3-1),
                    9/2*L3*L1*(3*L3-1),
                    9/2*L1*L3*(3*L1-1),
                    27*L1*L2*L3
                ]
            )
        case _:
            raise ValueError(
                "Shape functions not implemented for element order "
                f"{element_order}"
            )


def gradient_local_shape_functions_2d(s, t, element_order) -> npt.NDArray:
    """Returns the gradient of the local shape functions.

    Parameters:
        s: Local s parameter.
        t: Local t parameter.
        element_order: Order of the shape functions (1=linear, 2=quadratic,
            3=cubic).

    Returns:
        npt.NDArray: Gradient of each of the shape functions. Shape: (2,n),
            where n is the number of shape functions. Since currently shape
            functions are always linear: dim=1 -> n=2 and dim=2 -> n=3
    """
    match element_order:
        case 1:
            return np.array([
                [-1, 1, 0],  # d_s
                [-1, 0, 1]   # d_t
            ])
        case 2:
            return np.array([
                [ # d_s
                    -3+4*t+4*s,
                    4*s-1,
                    0,
                    4-8*s-4*t,
                    4*t,
                    -4*t
                ],
                [ # d_t
                    -3+4*s+4*t,
                    0,
                    4*t-1,
                    -4*s,
                    4*s,
                    4-4*s-8*t
                ]
            ])
        case 3:
            return np.array([
                [  # d_s
                    -13.5*s**2-27*s*t+18*s-13.5*t**2+18*t-5.5,
                    13.5*s**2-9*s+1,
                    0,
                    40.5*s**2+54*s*t-45*s+13.5*t**2-22.5*t+9,
                    -40.5*s**2-27*s*t+36*s+4.5*t-4.5,
                    t*(27*s-4.5),
                    t*(13.5*t-4.5),
                    t*(4.5-13.5*t),
                    t*(27*s+27*t-22.5),
                    27*t*(-2*s-t+1)
                ],
                [  # d_t
                    -13.5*s**2-27*s*t+18*s-13.5*t**2+18*t-5.5,
                    0,
                    13.5*t**2-9*t+1,
                    s*(27*s+27*t-22.5),
                    s*(4.5-13.5*s),
                    s*(13.5*s-4.5),
                    s*(27*t-4.5),
                    -27*s*t+4.5*s-40.5*t**2+36*t-4.5,
                    13.5*s**2+54*s*t-22.5*s+40.5*t**2-45*t+9,
                    27*s*(-s-2*t+1)
                ]
            ])

    raise ValueError(
        "Gradient of shape functions not implemented for element "
        f"order {element_order}"
    )


def local_to_global_coordinates(
    node_points: npt.NDArray,
    s: float,
    t: float,
    element_order: int
) -> Any:
    """Transforms the local coordinates given by dimensions using the node
    points to the global coordinates r, z.

    Parameters:
        node_points: For a line: [[x1, x2], [y1, y2]] and for a triangle:
            [[x1, x2, x3], [y1, y2, y3]] of
        s: Local s coordinate.
        t: Local t coordinate.
        element_order: Order of the shape functions.

    Returns:
        npt.NDArray: Global coordinates [r, z]
    """
    if node_points.shape[1] != int(1/2*(element_order+1)*(element_order+2)):
        raise ValueError(
            "The given node point array does not fit the given number of"
            " dimensions"
        )

    return np.dot(node_points, local_shape_functions_2d(s, t, element_order))


def b_operator_global(
    node_points: npt.NDArray,
    jacobian_inverted_t: npt.NDArray,
    s: float,
    t: float,
    element_order: int
):
    """Calculates the B operator for the local coordinantes which is needed
    for voigt-notation.
    The derivates are with respect to the global coordinates r and z.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        jacobian_inverted_t: Jacobian matrix inverted and transposed, needed
            for calculation of global derivatives.
        s: Local coordinate.
        t: Local coordinate.
        element_order: Order of the shape functions.

    Returns:
        B operator 4x6, for a u aligned like [u1_r, u1_z, u2_r, u2_z, ..].
    """
    nodes_per_element = int(1/2*(element_order+1)*(element_order+2))

    # Get local shape functions and r (because of theta component)
    n = local_shape_functions_2d(s, t, element_order)
    r = local_to_global_coordinates(node_points, s, t, element_order)[0]

    # Get gradients of local shape functions (s, t)
    dn = gradient_local_shape_functions_2d(s, t, element_order)

    # Convert to gradients of (r, z) using jacobi matrix
    global_dn = np.dot(jacobian_inverted_t, dn)

    # Initialize and fill array
    b = np.zeros(shape=(4, 2*nodes_per_element))
    for d in range(nodes_per_element):
        b[:, 2*d:2*d+2] = [
            [
                global_dn[0][d], 0
            ],
            [
                0, global_dn[1][d]
            ],
            [
                global_dn[1][d], global_dn[0][d]
            ],
            [
                n[d]/r, 0
            ],
        ]

    return b


def integral_m(node_points: npt.NDArray, element_order: int):
    """Calculates the M integral.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        element_order: Order of the shape functions.

    Returns:
        npt.NDArray: 3x3 M matrix for the given element.
    """
    def inner(s, t):
        dn = gradient_local_shape_functions_2d(s, t, element_order)
        jacobian = np.dot(node_points, dn.T)
        jacobian_det = np.linalg.det(jacobian)

        n = local_shape_functions_2d(s, t, element_order)

        # Since the simulation is axisymmetric it is necessary
        # to multiply with the radius in the integral
        # (for the theta component (azimuth))
        r = local_to_global_coordinates(node_points, s, t, element_order)[0]

        # Get all combinations of shape function multiplied with each other
        return np.outer(n, n)*r*jacobian_det

    return quadratic_quadrature(inner, element_order)


def integral_ku(
    node_points: npt.NDArray,
    elasticity_matrix: npt.NDArray,
    element_order: int
):
    """Calculates the Ku integral

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        elasticity_matrix: Elasticity matrix for the current element
            (c matrix).
        element_order: Order of the shape functions.

    Returns:
        npt.NDArray: 6x6 Ku matrix for the given element.
    """
    def inner(s, t):
        dn = gradient_local_shape_functions_2d(s, t, element_order)
        jacobian = np.dot(node_points, dn.T)
        jacobian_inverted_t = np.linalg.inv(jacobian).T
        jacobian_det = np.linalg.det(jacobian)

        b_op = b_operator_global(
            node_points,
            jacobian_inverted_t,
            s,
            t,
            element_order
        )
        r = local_to_global_coordinates(node_points, s, t, element_order)[0]

        return np.dot(np.dot(b_op.T, elasticity_matrix), b_op)*r*jacobian_det

    return quadratic_quadrature(inner, element_order)


def integral_kuv(
    node_points: npt.NDArray,
    piezo_matrix: npt.NDArray,
    element_order: int
):
    """Calculates the KuV integral.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        piezo_matrix: Piezo matrix for the current element (e matrix).
        element_order: Order of the shape functions.

    Returns:
        npt.NDArray: 6x3 KuV matrix for the given element.
    """
    def inner(s, t):
        dn = gradient_local_shape_functions_2d(s, t, element_order)
        jacobian = np.dot(node_points, dn.T)
        jacobian_inverted_t = np.linalg.inv(jacobian).T
        jacobian_det = np.linalg.det(jacobian)

        b_op = b_operator_global(
            node_points,
            jacobian_inverted_t,
            s,
            t,
            element_order
        )
        global_dn = np.dot(jacobian_inverted_t, dn)
        r = local_to_global_coordinates(node_points, s, t, element_order)[0]

        return np.dot(np.dot(b_op.T, piezo_matrix.T), global_dn)*r*jacobian_det

    return quadratic_quadrature(inner, element_order)


def integral_kve(
    node_points: npt.NDArray,
    permittivity_matrix: npt.NDArray,
    element_order: int
):
    """Calculates the KVe integral.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        permittivity_matrix: Permittivity matrix for the
            current element (epsilon matrix).
        element_order: Order of the shape functions.

    Returns:
        npt.NDArray: 3x3 KVe matrix for the given element.
    """
    def inner(s, t):
        dn = gradient_local_shape_functions_2d(s, t, element_order)
        jacobian = np.dot(node_points, dn.T)
        jacobian_inverted_t = np.linalg.inv(jacobian).T
        jacobian_det = np.linalg.det(jacobian)

        global_dn = np.dot(jacobian_inverted_t, dn)
        r = local_to_global_coordinates(node_points, s, t, element_order)[0]

        return np.dot(
            np.dot(
                global_dn.T,
                permittivity_matrix
            ),
            global_dn
        )*r*jacobian_det

    return quadratic_quadrature(inner, element_order)


def energy_integral_theta(
    node_points: npt.NDArray,
    theta: npt.NDArray,
    element_order: int
):
    """Integrates the given element over the given theta field.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        theta: List of the temperature field values of the points
            [theta1, theta2, theta3].
        element_order: Order of the shape functions.
    """
    def inner(s, t):
        dn = gradient_local_shape_functions_2d(s, t, element_order)
        jacobian = np.dot(node_points, dn.T)
        jacobian_det = np.linalg.det(jacobian)

        n = local_shape_functions_2d(s, t, element_order)
        r = local_to_global_coordinates(node_points, s, t, element_order)[0]

        return np.dot(n.T, theta)*r*jacobian_det

    return quadratic_quadrature(inner, element_order)


def integral_volume(node_points: npt.NDArray, element_order: int):
    """Calculates the volume of the triangle given by the node points.
    HINT: Must be multiplied with 2*np.pi and the jacobian determinant in order
    to give the correct volume of any rotationsymmetric triangle.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        element_order: Order of the shape functions.

    Returns:
        Float. Volume of the triangle.
    """
    def inner(s, t):
        dn = gradient_local_shape_functions_2d(s, t, element_order)
        jacobian = np.dot(node_points, dn.T)
        jacobian_det = np.linalg.det(jacobian)

        r = local_to_global_coordinates(node_points, s, t, element_order)[0]

        return r*jacobian_det

    return quadratic_quadrature(inner, element_order)


def quadratic_quadrature(func: Callable, element_order: int):
    """Integrates the given function of 2 variables using gaussian
    quadrature along 2 variables for a reference triangle.
    This gives exact results for linear shape functions.

    Parameters:
        func: Function which will be integrated.
        element_order: Order of the shape functions.

    Returns:
        Integral of the given function.
    """
    weights = []
    points = []

    match element_order:
        case 1 | 2:
            w1 = 1/6
            w2 = 2/3
            weights = np.array([w1, w1, w1])
            points = np.array([
                [w1, w1],
                [w2, w1],
                [w1, w2]
            ])
        case 3 | 4 | 5:
            w1 = 155-np.sqrt(15)
            w2 = 155+np.sqrt(15)
            w3 = 2400
            weights = np.array([
                w1/w3,
                w1/w3,
                w1/w3,
                w2/w3,
                w2/w3,
                w2/w3,
                9/80
            ])
            p1 = 6-np.sqrt(15)
            p2 = 9+2*np.sqrt(15)
            p3 = 6+np.sqrt(15)
            p4 = 9-2*np.sqrt(15)
            p5 = 21
            points = np.array([
                [p1/p5, p1/p5],
                [p2/p5, p1/p5],
                [p1/p5, p2/p5],
                [p3/p5, p3/p5],
                [p4/p5, p3/p5],
                [p3/p5, p4/p5],
                [1/3, 1/3]
            ])
        case _:
            raise NotImplementedError(
                "No quadratic quadrature for element order "
                f"{element_order} implemented"
            )

    # TODO Can this made faster using numpy?
    sum = 0
    for i in range(len(weights)):
        sum += weights[i]*func(points[i][0], points[i][1])

    return sum


def line_quadrature(func: Callable, element_order: int):
    """Integrates the given function of 2 variables along one variable
    for a reference triangle.
    This gives exact results for linear shape functions.

    Parameters:
        func: Function which will be integrated along r-axis.
        element_order: Order of the shape functions.

    Returns:
        Integral of the given function along r-axis"""
    weights = []
    points = []

    match element_order:
        case 1 | 2:
            weights = np.array([1/2, 1/2])
            points = np.array([
                (3-np.sqrt(3))/6,
                (3+np.sqrt(3))/6
            ])
        case 3 | 4 | 5:
            weights = np.array([5/18, 8/18, 5/18])
            points = np.array([
                (5-np.sqrt(15))/10,
                1/2,
                (5+np.sqrt(15))/5
            ])
        case _:
            raise NotImplementedError(
                "No linear quadrature for element order "
                f"{element_order} implemented"
            )

    # TODO Make use of numpy?
    sum = 0
    for i in range(len(weights)):
        sum += weights[i]*func(points[i])

    return sum


# -------- Boundary condition functions --------


def apply_dirichlet_bc(
    m: sparse.lil_array,
    c: sparse.lil_array,
    k: sparse.lil_array,
    nodes: npt.NDArray
) -> Tuple[sparse.lil_array, sparse.lil_array, sparse.lil_array]:
    """Prepares the given matrices m, c and k for the dirichlet boundary
    conditions. This is done by setting the corresponding rows to 0
    excepct for the node which will contain the specific value (this is set
    to 1). Right now only boundary conditions for v and u_r can be set, not
    for u_z (not neede yet).

    Parameters:
        m: Mass matrix M.
        c: Damping matrix C.
        k: Stiffness matrix K.
        nodes: List of nodes at which a dirichlet boundary condition
            shall be applied.

    Returns:
        Modified mass, damping and stiffness matrix.
    """
    # Set rows of matrices to 0 and diagonal of K to 1 (at node points)

    # Matrices for u_r component
    for node in nodes:
        # Set rows to 0
        m[node, :] = 0
        c[node, :] = 0
        k[node, :] = 0

        # Set diagonal values to 1
        k[node, node] = 1

    return m, c, k


def create_dirichlet_bc_nodes_freq(
    mesh: Mesh,
    amplitudes: npt.NDArray,
    number_of_frequencies: int,
    set_symmetric_bc: bool = True
) -> Tuple:
    """Create the dirichlet boundary condition nodes for a simulation in the
    frequency domain.

    Parameters:
        mesh: Mesh class for accessing nodes.
        amplitudes: List of amplitude per frequency step.
        number_of_frequencies: Total number of frequency steps.
        set_symmetric_bc: Set to True of the left border of the model is on
            the ordinate axis.

    Returns:
        The dirichlet nodes and values for u and v respectively.
    """

    # Get nodes from mesh
    pg_nodes = mesh.get_nodes_by_physical_groups(
        ["Electrode", "Symaxis", "Ground"])

    electrode_nodes = pg_nodes["Electrode"]
    symaxis_nodes = pg_nodes["Symaxis"]
    ground_nodes = pg_nodes["Ground"]

    if set_symmetric_bc:
        dirichlet_nodes_u = symaxis_nodes
        dirichlet_values_u = np.zeros(
            (number_of_frequencies, len(dirichlet_nodes_u), 2))
    else:
        dirichlet_nodes_u = np.array([])
        dirichlet_values_u = np.array([])

    # For potential v set electrode to excitation and ground to 0
    dirichlet_nodes_v = np.concatenate((electrode_nodes, ground_nodes))
    dirichlet_values_v = np.zeros(
        (number_of_frequencies, len(dirichlet_nodes_v)))

    # Set excitation value for electrode nodes points
    for freq_index, amplitude in enumerate(amplitudes):
        dirichlet_values_v[freq_index, :len(electrode_nodes)] = \
            amplitude

    return [dirichlet_nodes_u, dirichlet_nodes_v], \
        [dirichlet_values_u, dirichlet_values_v]


def create_dirichlet_bc_nodes(
    mesh: Mesh,
    electrode_excitation: npt.NDArray,
    number_of_time_steps: int,
    set_symmetric_bc: bool = True
):
    """Creates lists of nodes and values used in the simulation to
    apply the boundary conditions.
    There are 4 lists returning: nodes_u, values_u, nodes_v, values_v.
    The values are corresponding to the nodes with the same index.
    The lists are set using the given electrode excitation as well
    as the information if a symmetric bc is needed
        Disc -> Symmetric bc
        Ring -> No symmtric bc
    The symmetric bc sets the u_r values at the symmetric axis nodes to 0.
    The ground electrode nodes are implicitly set to 0.

    Parameters:
        electrode_nodes: Nodes in the electrode region.
        symaxis_nodes: Nodes on the symmetrical axis (r=0).
        ground_nodes: Nodes in the ground region.
        electrode_excitation: Excitation values for each time step.
        number_of_time_steps: Total number of time steps of the simulation.

    Returns:
        Tuple of 2 tuples. The first inner tuple is a tuple containing
        the nodes and values for u and the second inner tuple contains
        the nodes and values for v.
    """
    # Get nodes from gmsh handler
    pg_nodes = mesh.get_nodes_by_physical_groups(
        ["Electrode", "Symaxis", "Ground"])

    electrode_nodes = pg_nodes["Electrode"]
    symaxis_nodes = pg_nodes["Symaxis"]
    ground_nodes = pg_nodes["Ground"]

    # For Potential set "Electrode" to excitation function
    # "Ground" is set to 0
    # For displacement set symaxis values to 0.
    # Zeros are set for u_r and u_z but the u_z component is not used.
    # Others are implicit natrual bcs (Neumann B.C. -> 0).
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


def create_node_points(
    nodes: npt.NDArray,
    elements: npt.NDArray,
    element_order: int
) -> npt.NDArray:
    """Create the local node data and the corresponding matrices
    for every element which are needed in many parts of the simulations.

    Parameters:
        nodes: Nodes of the mesh
        elements: Elements of the mesh

    Returns:
        List of LocalElementData objects.
    """
    points_per_element = int(1/2*(element_order+1)*(element_order+2))

    node_points = np.zeros(
        shape=(len(elements), 2, points_per_element)
    )

    for element_index, element in enumerate(elements):
        # Get node points of element in format
        # [x1 x2 x3 ... xn]
        # [y1 y2 y3 ... yn] where (xi, yi) are the coordinates for Node i
        for node_index in range(points_per_element):
            node_points[element_index, :, node_index] = [
                nodes[element[node_index]][0],
                nodes[element[node_index]][1]
            ]

    return node_points


def calculate_volumes(node_points: npt.NDArray, element_order):
    """Calculates the volume of each element. The element information
    is given by the local_element_data

    Parameters:
        node_points: Node points of all elements for which the volume shall be
            calculated.

    Returns:
        List of volumes of the elements.
    """
    volumes = []

    number_of_elements = node_points.shape[0]

    for element_index in range(number_of_elements):
        volumes.append(
            integral_volume(node_points[element_index], element_order)
            *2*np.pi
        )

    return volumes


def get_avg_temp_field_per_element(
    theta: npt.NDArray,
    elements: npt.NDArray
):
    """Returns the average temperature for each element.

    Parameters:
        theta: Temperatures for each node.
        elements: List of elements.

    Returns:
        Mean temperature for each element.
    """
    theta_elements = np.zeros(len(elements))

    for element_index, element in enumerate(elements):
        theta_elements[element_index] = np.mean(theta[element])

    return np.array(theta_elements)
