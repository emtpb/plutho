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
class LocalElementData():
    """Conatins the data of a single element"""
    node_points: npt.NDArray
    jacobian: npt.NDArray
    jacobian_inverted_t: npt.NDArray
    jacobian_det: float


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


def local_shape_functions(*dimensions: float):
    """Returns the local linear shape functions for the given coordinates for a
    line or a triangle.
    The dimension of the returned shape functions depends on the number of
    coordinates given.

    Parameters:
        dimensions: Each parameter corresponds to one coordinate. If only one
            parameter is given it resembles 2 shape functions along a line. If
            two coordinates are given it resembles 3 shape functions for a
            triangle.

    Returns:
        npt.NDArray: Shape functions depending on the given parameters for a
            line or a triangle
    """
    dim = len(dimensions)
    if dim == 1:
        return np.array([1-dimensions[0], dimensions[0]])
    elif dim == 2:
        return np.array([
            1-dimensions[0]-dimensions[1],
            dimensions[0],
            dimensions[1]
        ])
    else:
        raise NotImplementedError(
            "Function not implemented for dimensions of 3 and above"
        )

def gradient_local_shape_functions(dim: int = 2):
    """Returns the gradient of the local shape functions. The dimension can be
    1 for a line or 2 for a triangle.

    Returns:
        npt.NDArray: Gradient of each of the shape functions. Shape: (2,n),
            where n is the number of shape functions. Since currently shape
            functions are always linear: dim=1 -> n=2 and dim=2 -> n=3
    """
    if dim == 1:
        print("dim=1")
        return np.array([
            [-1, 1],
            [0, 0]
        ])
    elif dim == 2:
        return np.array([
            [-1, 1, 0],
            [-1, 0, 1]
        ])
    else:
        raise NotImplementedError(
            "Function not implemented for dimensions of 3 and above"
        )


def local_to_global_coordinates(
        node_points: npt.NDArray,
        *dimensions: float
) -> Any:
    """Transforms the local coordinates given by dimensions using the node
    points to the global coordinates r, z.
    Parameters:
        node_points: For a line: [[x1, x2], [y1, y2]] and for a triangle:
            [[x1, x2, x3], [y1, y2, y3]] of
        dimensions: Local coordinates

    Returns:
        npt.NDArray: Global coordinates [r, z]
    """
    dim = len(dimensions)

    if node_points.shape[1] != dim + 1:
        raise ValueError(
            "The given node point array does not fit the given number of"
            " dimensions"
        )

    return np.dot(node_points, local_shape_functions(*dimensions))


def b_operator_global(
    node_points: npt.NDArray,
    jacobian_inverted_t: npt.NDArray,
    *dimensions: float
):
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
        B operator 4x6, for a u aligned like [u1_r, u1_z, u2_r, u2_z, ..].
    """
    dim = len(dimensions)

    # Get local shape functions and r (because of theta component)
    n = local_shape_functions(*dimensions)
    r = local_to_global_coordinates(node_points, *dimensions)[0]

    # Get gradients of local shape functions (s, t)
    dn = gradient_local_shape_functions(dim)

    # Convert to gradients of (r, z) using jacobi matrix
    global_dn = np.dot(jacobian_inverted_t, dn)

    # Initialize and fill array
    b = np.zeros(shape=(4, 2*(dim+1)))
    for d in range(dim+1):
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


def integral_m(node_points: npt.NDArray):
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
    elasticity_matrix: npt.NDArray
):
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
        b_op = b_operator_global(
            node_points,
            jacobian_inverted_t,
            s,
            t
        )
        r = local_to_global_coordinates(node_points, s, t)[0]

        return np.dot(np.dot(b_op.T, elasticity_matrix), b_op)*r

    return quadratic_quadrature(inner)


def integral_kuv(
    node_points: npt.NDArray,
    jacobian_inverted_t: npt.NDArray,
    piezo_matrix: npt.NDArray
):
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
        b_op = b_operator_global(
            node_points,
            jacobian_inverted_t,
            s,
            t
        )
        dn = gradient_local_shape_functions()
        global_dn = np.dot(jacobian_inverted_t, dn)
        r = local_to_global_coordinates(node_points, s, t)[0]

        return np.dot(np.dot(b_op.T, piezo_matrix.T), global_dn)*r

    return quadratic_quadrature(inner)


def integral_kve(
    node_points: npt.NDArray,
    jacobian_inverted_t: npt.NDArray,
    permittivity_matrix: npt.NDArray
):
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


def energy_integral_theta(
    node_points: npt.NDArray,
    theta: npt.NDArray
):
    """Integrates the given element over the given theta field.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        theta: List of the temperature field values of the points
            [theta1, theta2, theta3].
    """
    def inner(s, t):
        n = local_shape_functions(s, t)
        r = local_to_global_coordinates(node_points, s, t)[0]

        return np.dot(n.T, theta)*r

    return quadratic_quadrature(inner)


def integral_volume(node_points: npt.NDArray):
    """Calculates the volume of the triangle given by the node points.
    HINT: Must be multiplied with 2*np.pi and the jacobian determinant in order
    to give the correct volume of any rotationsymmetric triangle.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.

    Returns:
        Float. Volume of the triangle.
    """
    def inner(s, t):
        r = local_to_global_coordinates(node_points, s, t)[0]
        return r

    return quadratic_quadrature(inner)


def quadratic_quadrature(func: Callable):
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
    return 0.5*(func(-1/2/np.sqrt(3)+1/2) + func(1/2/np.sqrt(3)+1/2))


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


def create_local_element_data(
    nodes: npt.NDArray,
    elements: npt.NDArray
) -> List[LocalElementData]:
    """Create the local node data and the corresponding matrices
    for every element which are needed in many parts of the simulations.

    Parameters:
        nodes: Nodes of the mesh
        elements: Elements of the mesh

    Returns:
        List of LocalElementData objects.
    """
    local_element_data = []
    dn = gradient_local_shape_functions(2)
    for element in elements:
        # Get node points of element in format
        # [x1 x2 x3]
        # [y1 y2 y3] where (xi, yi) are the coordinates for Node i
        node_points = np.array([
            [nodes[element[0]][0],
             nodes[element[1]][0],
             nodes[element[2]][0]],
            [nodes[element[0]][1],
             nodes[element[1]][1],
             nodes[element[2]][1]]
        ])
        jacobian = np.dot(node_points, dn.T)
        jacobian_inverted_t = np.linalg.inv(jacobian).T
        jacobian_det = np.linalg.det(jacobian)
        local_element_data.append(LocalElementData(
            node_points,
            jacobian,
            jacobian_inverted_t,
            jacobian_det
        ))

    return local_element_data


def calculate_volumes(local_element_data: List[LocalElementData]):
    """Calculates the volume of each element. The element information
    is given by the local_element_data

    Parameters:
        local_element_data: List of LocalElementData objects.

    Returns:
        List of volumes of the elements.
    """
    volumes = []

    for local_element in local_element_data:
        node_points = local_element.node_points
        jacobian_det = local_element.jacobian_det
        volumes.append(integral_volume(node_points) * 2 * np.pi * jacobian_det)

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
