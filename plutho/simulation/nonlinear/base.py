"""Module for base functions used in nonlinear simulations"""

# Python standard libraries
from enum import Enum
from typing import Tuple, Callable

# Third party libraries
import numpy as np
import numpy.typing as npt
from scipy import sparse

# Local libraries
from ..base import MeshData, local_shape_functions_2d, local_to_global_coordinates, b_operator_global, \
    integral_m, integral_ku, integral_kuv, integral_kve, \
    create_node_points, quadratic_quadrature, gradient_local_shape_functions_2d
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
                "Missing 'nonlinear_matrix' parameter for nonlinear type:"
                " Rayleigh"
            )
        # Reduce to axisymmetric matrix
        nm = kwargs["nonlinear_matrix"]
        voigt_map = [0, 1, 3, 2]
        nm_reduced = np.zeros(shape=(4, 4, 4))
        for i_new, i_old in enumerate(voigt_map):
            for j_new, j_old in enumerate(voigt_map):
                for k_new, k_old in enumerate(voigt_map):
                    nm_reduced[i_new, j_new, k_new] = nm[i_old, j_old, k_old]

        nonlinear_matrix = nm_reduced
    else:
        raise NotImplementedError(
            f"Nonlinearity type {NonlinearType.value} is not implemented"
        )

    # TODO Assembly takes to long rework this algorithm
    # Maybe the 2x2 matrix slicing is not very fast
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
    lu = sparse.lil_matrix(
        (2*number_of_nodes, 2*number_of_nodes),
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
        if nonlinear_type is NonlinearType.Custom:
            lu_e = (
                integral_u_nonlinear(
                    current_node_points,
                    nonlinear_matrix,
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
    nonlinear_elasticity_matrix: npt.NDArray,
    element_order: int
):
    """Calculates the nonlinear stiffness integral

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        jacobian_inverted_t: Jacobian matrix inverted and transposed, needed
            for calculation of global derivatives.
        elasticity_matrix: Elasticity matrix for the current element
            (c matrix).

    Returns:
        npt.NDArray: 6x6 matrix for the given element.
    """
    def inner(s, t):
        dn = gradient_local_shape_functions_2d(s, t, element_order)
        jacobian = np.dot(node_points, dn.T)
        jacobian_inverted_t = np.linalg.inv(jacobian).T
        jacobian_det = np.linalg.det(jacobian)

        def t_i(s, t):
            b_op = b_operator_global(
                node_points,
                jacobian_inverted_t,
                s,
                t,
                element_order
            )

            return np.einsum(
                "ijk,jl,kl->il",
                nonlinear_elasticity_matrix,
                b_op,
                b_op
            )

        t_derived = b_opt_global_t_numerical(
            node_points,
            t_i,
            s,
            t,
            jacobian_inverted_t,
            element_order
        )

        r = local_to_global_coordinates(node_points, s, t, element_order)[0]

        N = np.diag(
            local_shape_functions_2d(s, t, element_order)
        )

        return np.outer(
            t_derived,
            N
        ) * 1/2 * r * jacobian_det

    return quadratic_quadrature(inner, element_order)


def integral_u_nonlinear_analytic(
    node_points: npt.NDArray,
    nonlinear_elasticity_matrix: npt.NDArray,
    element_order: int
):
    """Calculates the nonlinear stiffness integral

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        jacobian_inverted_t: Jacobian matrix inverted and transposed, needed
            for calculation of global derivatives.
        elasticity_matrix: Elasticity matrix for the current element
            (c matrix).

    Returns:
        npt.NDArray: 6x6 matrix for the given element.
    """
    def inner(s, t):
        dn = gradient_local_shape_functions_2d(s, t, element_order)
        jacobian = np.dot(node_points, dn.T)
        jacobian_inverted_t = np.linalg.inv(jacobian).T
        jacobian_det = np.linalg.det(jacobian)

        r = local_to_global_coordinates(node_points, s, t, element_order)[0]

        b_op = b_operator_global(
            node_points,
            jacobian_inverted_t,
            s,
            t,
            element_order
        )

        bb_op = double_b_operator_global(
            node_points,
            jacobian_inverted_t,
            s,
            t,
            element_order
        )

        left = np.einsum(
            "ijk,jl,kl->il",
            nonlinear_elasticity_matrix,
            bb_op,
            b_op
        )

        right = np.einsum(
            "ijk,jl,kl->il",
            nonlinear_elasticity_matrix,
            b_op,
            bb_op
        )

        return np.outer(
            left+right,
            local_shape_functions_2d(s, t, element_order)
        ) * 1/2 * r * jacobian_det

    return quadratic_quadrature(inner, element_order)


def b_opt_global_t_numerical(
    node_points,
    func: Callable,
    s,
    t,
    jacobian_inverted_t,
    element_order
):
    # Idea:
    # f contains the function values at the nodes
    # first calculate f at the given positions s and t
    # then derive f at that point after s and t
    # then calculate the derivatives after r and z

    # f should have shape 4 x nodes_per_element

    # TODO Check if this value is suitable
    h = 1/20

    nodes_per_element = int(1/2*(element_order+1)*(element_order+2))

    # Points at which the function is evaluated
    p1 = [s+h,t]
    p2 = [s-h,t]
    p3 = [s,t+h]
    p4 = [s,t-h]

    r = local_to_global_coordinates(node_points, s, t, element_order)[0]

    # Local derivatives
    f = func(s, t)
    ds_f = 1/(2*h)*(func(*p1)-func(*p2))
    dt_f = 1/(2*h)*(func(*p3)-func(*p4))

    dr_f = np.zeros(shape=ds_f.shape)
    dz_f = np.zeros(shape=dt_f.shape)
    b_num = np.zeros(shape=(2*nodes_per_element, 2*nodes_per_element))

    for i in range(ds_f.shape[0]):
        for j in range(ds_f.shape[1]):
            # TODO Not all derivatives are needed
            # Derivatives with respect to local coordinates s and t
            local_der = np.array([ds_f[i][j], dt_f[i][j]])

            # Derivatives with respect to global coordinates r and z
            global_der = np.dot(jacobian_inverted_t, local_der)
            dr_f[i, j] = global_der[0]
            dz_f[i, j] = global_der[1]

    for node_index in range(nodes_per_element):
        current_f = f[:, 2*node_index:2*(node_index+1)]
        dr = dr_f[:, 2*node_index:2*(node_index+1)]
        dz = dz_f[:, 2*node_index:2*(node_index+1)]

        # Get global derivatives
        b_num[2*node_index:2*(node_index+1), 2*node_index:2*(node_index+1)] = \
            [
                [
                    dr[0][0]+dz[2][1]+current_f[3][0]/r,
                    dr[0][1]+dz[2][1]+current_f[3][1]
                ],
                [
                    dz[1][0]+dr[2][0],
                    dz[1][1]+dr[2][1]
                ]
            ]

    return b_num


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

def double_b_operator_global(
    node_points: npt.NDArray,
    jacobian_inverted_t: npt.NDArray,
    s: float,
    t: float,
    element_order: int
):
    nodes_per_element = int(1/2*(element_order+1)*(element_order+2))

    dn = gradient_local_shape_functions_2d(s, t, element_order)
    global_dn = np.dot(jacobian_inverted_t, dn)
    ddn = second_gradient_local_shape_functions_2d(s, t, element_order)

    # Calculate derivatives with respect to global coordinates
    global_ddn = np.zeros(shape=ddn.shape)

    for node_index in range(nodes_per_element):
        # Get hessian with shape
        # [ds*ds, ds*dt]
        # [dt*ds, dt*dt]
        # Local hessian since the derivatives are with respect to s and t
        local_hessian = np.array([
            [global_ddn[0][0][node_index], global_ddn[0][1][node_index]],
            [global_ddn[1][0][node_index], global_ddn[1][1][node_index]]
        ])

        # Global hessian since the derivatives are with respect to r and z
        # TODO Verify
        global_hessian = np.dot(
            np.dot(
                jacobian_inverted_t,
                local_hessian
            ),
            jacobian_inverted_t.T
        )

        # TODO Make using slices
        global_ddn[0][0][node_index] = global_hessian[0][0]
        global_ddn[0][1][node_index] = global_hessian[0][1]
        global_ddn[1][0][node_index] = global_hessian[1][0]
        global_ddn[1][1][node_index] = global_hessian[1][1]

    r = local_to_global_coordinates(node_points, s, t, element_order)[0]

    bb = np.zeros(shape=(2*nodes_per_element, 2*nodes_per_element))
    for i in range(nodes_per_element):
        for j in range(nodes_per_element):
            if i == j:
                # Diagonal elements have second order derivatives
                bb[2*i:2*(i+1), 2*j:2*(j+1)] = [
                    [
                        global_ddn[0][0][i] + global_ddn[1][1][i]+1/(r**2),
                        global_ddn[1][0][i]
                    ],
                    [
                        global_ddn[0][1][i],
                        global_ddn[0][0][i] + global_ddn[1][1][i]
                    ]
                ]
            else:
                # Offdiagional elements have 2 first order derivatives
                bb[2*i:2*(i+1), 2*j:2*(j+1)] = [
                    [
                        (
                            global_dn[0][i]*global_dn[0][j]
                            + global_dn[1][i]*global_dn[1][j]
                            + 1/(r**2)
                        ),
                        global_dn[1][i]*global_dn[1][j]
                    ],
                    [
                        global_dn[0][i]*global_dn[1][j],
                        (
                            global_dn[0][i]*global_dn[0][j]
                            + global_dn[1][i]*global_dn[1][j]
                        )
                    ]
                ]

    return bb
