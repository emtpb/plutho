"""Module for base functions used in nonlinear simulations"""

# Python standard libraries
from enum import Enum
from typing import Tuple

# Third party libraries
import numpy as np
import numpy.typing as npt
from scipy import sparse

# Local libraries
from ..base import MeshData, local_to_global_coordinates, b_operator_global, \
    integral_m, integral_ku, integral_kuv, integral_kve, \
    create_local_element_data, quadratic_quadrature
from plutho.materials import MaterialManager


class NonlinearType(Enum):
    Rayleigh = "Rayleigh"
    Custom = "Custom"


def assemble(
        mesh_data: MeshData,
        material_manager: MaterialManager,
        nonlinear_type: NonlinearType,
        **kwargs
) -> Tuple[
    sparse.lil_array,
    sparse.lil_array,
    sparse.lil_array,
    sparse.lil_array
]:
    """Assembles the FEM matrices based on the set mesh_data and
    material_data. Resulting FEM matrices are saved in global variables.

    Parameters:
        ntype: NonlinearType object.
        kwargs: Can contain different parameters based on the nonlinear
            type:
            - If ntype=NonlinearType.Rayleigh:
                "zeta" (float): Scalar nonlinearity parameter.
            - If ntype=NonlinearType.Custom:
                "nonlinear_matrix" (npt.NDArray): 6x6 custom nonlinear
                    material matrix.

    Returns:
        Tuple containing the assembled matrices m, c, k, ln
    """

    # Check for nonlinear types
    if nonlinear_type is NonlinearType.Rayleigh:
        if "zeta" not in kwargs:
            raise ValueError(
                "Missing 'zeta' parameter for nonlinear type: Rayleigh"
            )
        zeta = kwargs["zeta"]
    elif nonlinear_type is NonlinearType.Custom:
        if "nonlinear_matrix" not in kwargs:
            raise ValueError(
                "Missing 'nonlinear_matrix' parameter for nonlinaer type:"
                " Rayleigh"
            )
        nm = kwargs["nonlinear_matrix"]
        nonlinear_matrix = np.array([
            [nm[0, 0], nm[0, 2], 0, nm[0, 1]],
            [nm[0, 2], nm[2, 2], 0, nm[0, 2]],
            [0, 0, nm[3, 3], 0],
            [nm[0, 1], nm[0, 2], 0, nm[0, 0]]
        ])
    else:
        raise NotImplementedError(
            f"Nonlinearity type {NonlinearType.value} is not implemented"
        )

    # TODO Assembly takes to long rework this algorithm
    # Maybe the 2x2 matrix slicing is not very fast
    nodes = mesh_data.nodes
    elements = mesh_data.elements
    local_elements = create_local_element_data(nodes, elements)

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
    lu = sparse.lil_matrix(
        (2*number_of_nodes, 2*number_of_nodes),
        dtype=np.float64
    )

    for element_index, element in enumerate(elements):
        # Get local element nodes and matrices
        local_element = local_elements[element_index]
        node_points = local_element.node_points
        jacobian_inverted_t = local_element.jacobian_inverted_t
        jacobian_det = local_element.jacobian_det

        # TODO Check if its necessary to calculate all integrals
        # --> Dirichlet nodes could be leaved out?
        # Multiply with jac_det because its integrated with respect to
        # local coordinates.
        # Mutiply with 2*pi because theta is integrated from 0 to 2*pi
        mu_e = (
            material_manager.get_density(element_index)
            * integral_m(node_points)
            * jacobian_det * 2 * np.pi
        )
        ku_e = (
            integral_ku(
                node_points,
                jacobian_inverted_t,
                material_manager.get_elasticity_matrix(element_index)
            )
            * jacobian_det * 2 * np.pi
        )
        kuv_e = (
            integral_kuv(
                node_points,
                jacobian_inverted_t,
                material_manager.get_piezo_matrix(element_index)
            )
            * jacobian_det * 2 * np.pi
        )
        kve_e = (
            integral_kve(
                node_points,
                jacobian_inverted_t,
                material_manager.get_permittivity_matrix(
                    element_index
                )
            )
            * jacobian_det * 2 * np.pi
        )
        if nonlinear_type is NonlinearType.Custom:
            lu_e = (
                integral_u_nonlinear(
                    node_points,
                    jacobian_inverted_t,
                    nonlinear_matrix
                ) * jacobian_det * 2 * np.pi
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

                if nonlinear_type is NonlinearType.Custom:
                    # L_e is similar to Ku_e
                    lu[
                        2*global_p:2*global_p+2,
                        2*global_q:2*global_q+2
                    ] += lu_e[
                        2*local_p:2*local_p+2,
                        2*local_q:2*local_q+2
                    ]

    # Calculate damping matrix
    # Currently in the simulations only one material is used
    # TODO Add algorithm to calculate cu for every element on its own
    # in order to use multiple materials
    cu = (
        material_manager.get_alpha_m(0) * mu
        + material_manager.get_alpha_k(0) * ku
    )

    if nonlinear_type is NonlinearType.Rayleigh:
        # Set Rayleigh type nonlinear matrix
        lu = zeta*ku

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
    ln = sparse.block_array([
        [lu, zeros2x1],
        [zeros1x2, zeros1x1]
    ]).tolil()

    return m, c, k, ln


def integral_u_nonlinear(
    node_points: npt.NDArray,
    jacobian_inverted_t: npt.NDArray,
    nonlinear_elasticity_matrix: npt.NDArray
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
            node_points, jacobian_inverted_t,
            s,
            t,
        )
        r = local_to_global_coordinates(node_points, s, t)[0]

        return 1/2*np.dot(np.dot(b_op.T, nonlinear_elasticity_matrix), b_op)*r

    return quadratic_quadrature(inner)
