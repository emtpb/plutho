"""Module for base functions used in nonlinear simulations"""

# Python standard libraries
from enum import Enum
from typing import Tuple, Callable, Union

# Third party libraries
import numpy as np
import numpy.typing as npt
from scipy import sparse

# Local libraries
from ..base import MeshData, local_to_global_coordinates, b_operator_global, \
    integral_m, integral_ku, integral_kuv, integral_kve, \
    create_node_points, quadratic_quadrature, gradient_local_shape_functions_2d
from plutho.materials import MaterialManager


class NonlinearType(Enum):
    Rayleigh = "Rayleigh"
    Custom = "Custom"


def assemble(
    mesh_data: MeshData,
    material_manager: MaterialManager
) -> Tuple[
    sparse.lil_array,
    sparse.lil_array,
    sparse.lil_array
]:
    # TODO Assembly takes to long rework this algorithm
    # Maybe the 2x2 matrix slicing is not very fast
    # TODO Change to lil_array
    nodes = mesh_data.nodes
    elements = mesh_data.elements
    element_order = mesh_data.element_order
    node_points = create_node_points(nodes, elements, element_order)

    number_of_nodes = len(nodes)
    mu = sparse.lil_matrix(
        (2*number_of_nodes, 2*number_of_nodes),
        dtype=np.float64
    )
    ku = sparse.lil_matrix(
        (2*number_of_nodes, 2*number_of_nodes),
        dtype=np.float64
    )
    kuv = sparse.lil_matrix(
        (2*number_of_nodes, number_of_nodes),
        dtype=np.float64
    )
    kv = sparse.lil_matrix(
        (number_of_nodes, number_of_nodes),
        dtype=np.float64
    )

    for element_index, element in enumerate(elements):
        current_node_points = node_points[element_index]

        # TODO Check if its necessary to calculate all integrals
        # --> Dirichlet nodes could be leaved out?
        # Multiply with jac_det because its integrated with respect to
        # local coordinates.
        # Mutiply with 2*pi because theta is integrated from 0 to 2*pi
        mu_e = (
            material_manager.get_density(element_index)
            * integral_m(current_node_points, element_order)
            * 2 * np.pi
        )
        ku_e = (
            integral_ku(
                current_node_points,
                material_manager.get_elasticity_matrix(element_index),
                element_order
            ) * 2 * np.pi
        )
        kuv_e = (
            integral_kuv(
                current_node_points,
                material_manager.get_piezo_matrix(element_index),
                element_order
            ) * 2 * np.pi
        )
        kve_e = (
            integral_kve(
                current_node_points,
                material_manager.get_permittivity_matrix(
                    element_index
                ),
                element_order
            ) * 2 * np.pi
        )

        # Now assemble all element matrices
        for local_p, global_p in enumerate(element):
            for local_q, global_q in enumerate(element):
                # Mu_e is returned as a 3x3 matrix (should be 6x6)
                # because the values are the same.
                # Only the diagonal elements have values.
                mu[2*global_p, 2*global_q] += mu_e[local_p][local_q]
                mu[2*global_p+1, 2*global_q+1] += mu_e[local_p][local_q]

                # Ku_e is a 6x6 matrix and 2x2 matrices are sliced out
                # of it.
                ku[2*global_p:2*global_p+2, 2*global_q:2*global_q+2] += \
                    ku_e[2*local_p:2*local_p+2, 2*local_q:2*local_q+2]

                # KuV_e is a 6x3 matrix and 2x1 vectors are sliced out.
                # [:, None] converts to a column vector.
                kuv[2*global_p:2*global_p+2, global_q] += \
                    kuv_e[2*local_p:2*local_p+2, local_q][:, None]

                # KVe_e is a 3x3 matrix and therefore the values can
                # be set directly.
                kv[global_p, global_q] += kve_e[local_p, local_q]

    # Calculate damping matrix
    # Currently in the simulations only one material is used
    # TODO Add algorithm to calculate cu for every element on its own
    # in order to use multiple materials
    cu = (
        material_manager.get_alpha_m(0) * mu
        + material_manager.get_alpha_k(0) * ku
    )

    # Calculate block matrices
    zeros1x1 = np.zeros((number_of_nodes, number_of_nodes))
    zeros2x1 = np.zeros((2*number_of_nodes, number_of_nodes))
    zeros1x2 = np.zeros((number_of_nodes, 2*number_of_nodes))

    m = sparse.block_array([
        [mu, zeros2x1],
        [zeros1x2, zeros1x1],
    ]).tolil()
    c = sparse.block_array([
        [cu, zeros2x1],
        [zeros1x2, zeros1x1],
    ]).tolil()
    k = sparse.block_array([
        [ku, kuv],
        [kuv.T, -1*kv]
    ]).tolil()

    print("m:", m.count_nonzero())
    print("c:", c.count_nonzero())
    print("k:", k.count_nonzero())

    print("mu:", mu.count_nonzero())
    print("ku:", ku.count_nonzero())
    print("kuv:", kuv.count_nonzero())
    print("kv:", kv.count_nonzero())

    return m, c, k


def integral_u_nonlinear_quadratic(
    node_points,
    nonlinear_elasticity_tensor,
    element_order
):
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

        B_TC = np.einsum("im,mjk->ijk", b_op.T, nonlinear_elasticity_tensor)
        B_TCB = np.einsum("ink,nj->ijk", B_TC, b_op)
        B_TCBB = np.einsum("ijp,pk->ijk", B_TCB, b_op)

        return B_TCBB * r * jacobian_det

    return quadratic_quadrature(inner, element_order)


def integral_u_nonlinear_cubic(
    node_points,
    nonlinear_elasticity_tensor,
    element_order
):
    raise NotImplementedError("Cubic nonlinearity not implemented yet")


def second_gradient_local_shape_functions_2d(
    s,
    t,
    element_order
) -> npt.NDArray:
    # Since the second gradient has to be done for s and t as well, 3 indices
    # are needed [i,j,k]. The first two represent if its derived after s [i=0]
    # or t [i=1] and the last index [k] represents the node index of the
    # element.
    nodes_per_element = int(1/2*(element_order+1)*(element_order+2))
    match element_order:
        case 1:
            return np.zeros(shape=(2, 2, nodes_per_element))
        case 2:
            return np.array([
                [  # (outer) d_s
                    [4, 4, 0, -8, 0, 0],  # (inner) d_s
                    [4, 0, 0, -4, 4, -4]  # (inner) d_t
                ],
                [  # (outer) d_t
                    [4, 0, 0, -4, 4, -4],  # (inner) d_s
                    [4, 0, 4, 0, 0, -8]  # (inner) d_t
                ],
            ])
        case 3:
            raise NotImplementedError("Not yet implemented for order 3")

    raise ValueError(
        "Second gradient of shape functions not implemented for element "
        f"order {element_order}"
    )
