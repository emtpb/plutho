"""This module contains various helper functions needed in some simulations.
"""

# Python standard libraries
from typing import Union, Tuple

# Third party libraries
import numpy as np
import numpy.typing as npt
from scipy import sparse

# Local libraries
from .integrals import integral_charge_u, integral_charge_v, integral_volume
from ..materials import MaterialManager


__all__ = [
    "create_node_points",
    "calculate_volumes",
    "get_avg_temp_field_per_element",
    "calculate_charge",
]


def create_node_points(
    nodes: npt.NDArray,
    elements: npt.NDArray,
    element_order: int
) -> npt.NDArray:
    """Create the local node data for every given element.

    Parameters:
        nodes: Nodes of the mesh.
        elements: Elements of the mesh.
        element_order: Order of the elements.

    Returns:
        List of nodes.
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


def calculate_volumes(
    node_points: npt.NDArray,
    element_order: int
) -> npt.NDArray:
    """Calculates the volume of each element. The element information
    is given by the local_element_data

    Parameters:
        node_points: Node points of all elements for which the volume shall be
            calculated.
        element_order: Order of the elements.

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

    return np.array(volumes)


def get_avg_temp_field_per_element(
    theta: npt.NDArray,
    elements: npt.NDArray
) -> npt.NDArray:
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


def calculate_charge(
    u: npt.NDArray,
    material_manager: MaterialManager,
    elements: npt.NDArray,
    element_normals: npt.NDArray,
    nodes: npt.NDArray,
    element_order: int,
    is_complex: bool = False
) -> float:
    """Calculates the charge of the given elements.

    Parameters:
        u: List of u values for every node.
        material_manager: MaterialManager object.
        elements: Elements for which the charge shall be calculated.
        element_normals: List of element normal vectors (index corresponding to
            the elements list).
        nodes: All nodes used in the simulation.
        element_order: Order of the shape functions.
        is_complex: Set to true of u can be complex typed.
    """
    points_per_element = int(1/2*(element_order+1)*(element_order+2))
    number_of_nodes = len(nodes)
    q = 0

    for element_index, element in enumerate(elements):
        node_points = np.zeros(shape=(2, points_per_element))
        for node_index in range(points_per_element):
            node_points[:, node_index] = [
                nodes[element[node_index]][0],
                nodes[element[node_index]][1]
            ]

        # The integration is always done along the first two points of the
        # triangle.
        # TODO Not true for element_order > 1
        jacobian_det = np.sqrt(
            np.square(nodes[element[1]][0]-nodes[element[0]][0])
            + np.square(nodes[element[1]][1]-nodes[element[0]][1])
        )

        # Altough it is a line integral, it is necessary to take the field of
        # all element points even if they are not on the line.
        # This is due to the charge calculation needing the derivatives of the
        # shape function (dN_i/dr and dN_i/dz) which are (+/-) 1 along the
        # whole triangle for all 3 points.
        # TODO Is this true?
        if is_complex:
            u_e = np.zeros(2*points_per_element, dtype=np.complex128)
            ve_e = np.zeros(points_per_element, dtype=np.complex128)
        else:
            u_e = np.zeros(2*points_per_element)
            ve_e = np.zeros(points_per_element)

        for i in range(points_per_element):
            u_e[2*i] = u[2*element[i]]
            u_e[2*i+1] = u[2*element[i]+1]
            ve_e[i] = u[element[i]+2*number_of_nodes]

        q_u = integral_charge_u(
            node_points,
            u_e,
            material_manager.get_piezo_matrix(element_index),
            element_order
        ) * 2 * np.pi * jacobian_det
        q_v = integral_charge_v(
            node_points,
            ve_e,
            material_manager.get_permittivity_matrix(element_index),
            element_order
        ) * 2 * np.pi * jacobian_det

        # Now take the component normal to the line (outer direction)
        q_u_normal = np.dot(q_u, element_normals[element_index])
        q_v_normal = np.dot(q_v, element_normals[element_index])

        q += q_u_normal - q_v_normal

    return q

def mat_apply_dbcs(
    m: Union[sparse.lil_array, None],
    c: Union[sparse.lil_array, None],
    k: sparse.lil_array,
    nodes: npt.NDArray
) -> Tuple[
    Union[sparse.lil_array, None],
    Union[sparse.lil_array, None],
    sparse.lil_array
]:
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
    if m is not None:
        m[nodes, :] = 0

    if c is not None:
        c[nodes, :] = 0

    k[nodes, :] = 0
    k[nodes, nodes] = 1

    return m, c, k
