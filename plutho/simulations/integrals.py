"""General module for the implementation of various functions on local and
global shape functions as well as integration techniques"""

# Python standard libraries
from typing import Callable

# Third party libraries
import numpy as np
import numpy.typing as npt


#
# -------- Local shape functions --------
#
def local_shape_functions_2d(
    s: float,
    t: float,
    element_order: int
) -> npt.NDArray:
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


def gradient_local_shape_functions_2d(
    s: float,
    t: float,
    element_order: int
) -> npt.NDArray:
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
                [  # d_s
                    -3+4*t+4*s,
                    4*s-1,
                    0,
                    4-8*s-4*t,
                    4*t,
                    -4*t
                ],
                [  # d_t
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
) -> npt.NDArray:
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
) -> npt.NDArray:
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

#
# -------- Integrals --------
#
def integral_m(node_points: npt.NDArray, element_order: int) -> npt.NDArray:
    """Calculates the M integral.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        element_order: Order of the shape functions.

    Returns:
        npt.NDArray: 3x3 M matrix for the given element.
    """
    def inner(s: float, t: float) -> npt.NDArray:
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
) -> npt.NDArray:
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
    def inner(s: float, t: float) -> npt.NDArray:
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

        return np.dot(
            np.dot(b_op.T, elasticity_matrix),
            b_op
        ) * r * jacobian_det

    return quadratic_quadrature(inner, element_order)


def integral_kuv(
    node_points: npt.NDArray,
    piezo_matrix: npt.NDArray,
    element_order: int
) -> npt.NDArray:
    """Calculates the KuV integral.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        piezo_matrix: Piezo matrix for the current element (e matrix).
        element_order: Order of the shape functions.

    Returns:
        npt.NDArray: 6x3 KuV matrix for the given element.
    """
    def inner(s: float, t: float) -> npt.NDArray:
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

        return np.dot(
            np.dot(b_op.T, piezo_matrix.T),
            global_dn
        ) * r * jacobian_det

    return quadratic_quadrature(inner, element_order)


def integral_kve(
    node_points: npt.NDArray,
    permittivity_matrix: npt.NDArray,
    element_order: int
) -> npt.NDArray:
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
    def inner(s: float, t: float) -> npt.NDArray:
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
        ) * r * jacobian_det

    return quadratic_quadrature(inner, element_order)


def energy_integral_theta(
    node_points: npt.NDArray,
    theta: npt.NDArray,
    element_order: int
) -> npt.NDArray:
    """Integrates the given element over the given theta field.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        theta: List of the temperature field values of the points
            [theta1, theta2, theta3].
        element_order: Order of the shape functions.
    """
    def inner(s: float, t: float) -> npt.NDArray:
        dn = gradient_local_shape_functions_2d(s, t, element_order)
        jacobian = np.dot(node_points, dn.T)
        jacobian_det = np.linalg.det(jacobian)

        n = local_shape_functions_2d(s, t, element_order)
        r = local_to_global_coordinates(node_points, s, t, element_order)[0]

        return np.dot(n.T, theta) * r * jacobian_det

    return quadratic_quadrature(inner, element_order)


def integral_volume(
    node_points: npt.NDArray,
    element_order: int
) -> npt.NDArray:
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
    def inner(s: float, t: float) -> npt.NDArray:
        dn = gradient_local_shape_functions_2d(s, t, element_order)
        jacobian = np.dot(node_points, dn.T)
        jacobian_det = np.linalg.det(jacobian)

        r = local_to_global_coordinates(node_points, s, t, element_order)[0]

        return r * jacobian_det

    return quadratic_quadrature(inner, element_order)


def integral_loss_scs(
    node_points: npt.NDArray,
    u_e: npt.NDArray,
    elasticity_matrix: npt.NDArray,
    element_order: int
):
    """Calculate sthe integral of dS/dt*c*dS/dt over one triangle in the
    frequency domain for the given frequency.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of one
            triangle.
        u_e: Displacement at this element [u1_r, u1_z, u_2r, u_2z, u_3r, u_3z].
        angular_frequency: Angular frequency at which the loss shall be
            calculated.
        elasticity_matrix: Elasticity matrix for the current element
            (c matrix).
        element_order: Order of the shape functions.
    """
    def inner(s, t):
        dn = gradient_local_shape_functions_2d(s, t, element_order)
        jacobian = np.dot(node_points, dn.T)
        jacobian_inverted_t = np.linalg.inv(jacobian).T
        jacobian_det = np.linalg.det(jacobian)

        b_opt = b_operator_global(
            node_points,
            jacobian_inverted_t,
            s,
            t,
            element_order
        )
        r = local_to_global_coordinates(node_points, s, t, element_order)[0]

        s_e = np.dot(b_opt, u_e)

        # return np.dot(dt_s.T, np.dot(elasticity_matrix.T, dt_s))*r
        return np.dot(
            np.conjugate(s_e).T,
            np.dot(
                elasticity_matrix.T,
                s_e
            )
        ) * r * jacobian_det

    return quadratic_quadrature(inner, element_order)


def integral_charge_u(
    node_points: npt.NDArray,
    u_e: npt.NDArray,
    piezo_matrix: npt.NDArray,
    element_order: int
):
    """Calculates the integral of eBu of the given element.

    Parameters:
        node_points: List of node points [[x1, x2], [y1, y2]] of
            one line.
        u_e: List of u values at the nodes of the triangle
            [u1_r, u1_z, u2_r, u2_z].
        piezo_matrix: Piezo matrix for the current element (e matrix).
        element_order: Order of the shape functions.

    Returns:
        Float: Integral of eBu of the current triangle.
    """
    def inner(s):
        dn = gradient_local_shape_functions_2d(s, 0, element_order)
        jacobian = np.dot(node_points, dn.T)
        jacobian_inverted_t = np.linalg.inv(jacobian).T

        b_opt_global = b_operator_global(
            node_points,
            jacobian_inverted_t,
            s,
            0,
            element_order
        )
        r = local_to_global_coordinates(node_points, s, 0, element_order)[0]

        return -np.dot(np.dot(piezo_matrix, b_opt_global), u_e)*r

    return line_quadrature(inner, element_order)


def integral_charge_v(
    node_points: npt.NDArray,
    v_e: npt.NDArray,
    permittivity_matrix: npt.NDArray,
    element_order: int
):
    """Calculates the integral of epsilonBVe of the given element.

    Parameters:
        node_points: List of node points [[x1, x2], [y1, y2]] of
            one line.
        v_e: List of u values at the nodes of the line
            [v1, v2].
        permittivity_matrix: Permittivity matrix for the current
            element (e matrix).
        element_order: Order of the shape functions.

    Returns:
        Float: Integral of epsilonBVe of the current triangle.
    """
    def inner(s):
        dn = gradient_local_shape_functions_2d(s, 0, element_order)
        jacobian = np.dot(node_points, dn.T)
        jacobian_inverted_t = np.linalg.inv(jacobian).T

        global_dn = np.dot(jacobian_inverted_t, dn)
        r = local_to_global_coordinates(node_points, s, 0, element_order)[0]

        return -np.dot(np.dot(permittivity_matrix, global_dn), v_e)*r

    return line_quadrature(inner, element_order)


def integral_heat_flux(
    node_points: npt.NDArray,
    heat_flux: npt.NDArray,
    element_order: int
) -> npt.NDArray:
    """Integrates the heat flux using the shape functions.

    Parameters:
        node_points: List of node points [[x1, x2, ..], [y1, y2, ..]].
        heat_flux: Heat flux at the points.
        element_order: Order of the shape functions.

    Returns:
        npt.NDArray heat flux integral on each point.
    """
    def inner(s: float) -> npt.NDArray:
        n = local_shape_functions_2d(s, 0, element_order)
        r = local_to_global_coordinates(node_points, s, 0, element_order)[0]

        return n*heat_flux*r

    return line_quadrature(inner, element_order)


def integral_ktheta(
    node_points: npt.NDArray,
    element_order: int
) -> npt.NDArray:
    """Calculates the Ktheta integral.

    Parameters:
        node_points: List of node points [[x1, x2, ..], [y1, y2, ..]].
        element_order: Order of the shape functions.

    Returns:
        npt.NDArray: 3x3 Ktheta matrix for the given element.
    """
    def inner(s: float, t: float) -> npt.NDArray:
        dn = gradient_local_shape_functions_2d(s, t, element_order)
        jacobian = np.dot(node_points, dn.T)
        jacobian_inverted_t = np.linalg.inv(jacobian).T
        jacobian_det = np.linalg.det(jacobian)

        global_dn = np.dot(jacobian_inverted_t, dn)
        r = local_to_global_coordinates(node_points, s, t, element_order)[0]

        return np.dot(global_dn.T, global_dn)*r*jacobian_det

    return quadratic_quadrature(inner, element_order)


def integral_theta_load(
    node_points: npt.NDArray,
    mech_loss: float,
    element_order: int
) -> npt.NDArray:
    """Returns the load value for the temperature field (f) for the specific
    element.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        point_loss: Loss power on each node (heat source).
        element_order: Order of the shape functions.

    Returns:
        npt.NDArray: f vector value at the specific ndoe
    """
    def inner(s: float, t: float) -> npt.NDArray:
        dn = gradient_local_shape_functions_2d(s, t, element_order)
        jacobian = np.dot(node_points, dn.T)
        jacobian_det = np.linalg.det(jacobian)

        n = local_shape_functions_2d(s, t, element_order)
        r = local_to_global_coordinates(node_points, s, t, element_order)[0]

        return n*mech_loss*r*jacobian_det

    return quadratic_quadrature(inner, element_order)


def integral_loss_scs_time(
    node_points: npt.NDArray,
    u_e_t: npt.NDArray,
    u_e_t_minus_1: npt.NDArray,
    u_e_t_minus_2: npt.NDArray,
    delta_t: float,
    elasticity_matrix: npt.NDArray,
    element_order: int
) -> npt.NDArray:
    """Calculates the integral of dS/dt*c*dS/dt over one triangle. Since foward
    difference quotient of second oder is used the last 2 values of e_u are
    needed.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        u_e_t: Values of u_e at the current time point.
            Format: [u1_r, u1_z, u2_r, u2_z, u3_r, u3_z].
        u_e_t_minus_1: Values of u_e at one time point earlier. Same format
            as u_e_t.
        u_e_t_minus_2: Values of u_e at two time points earlier. Same format
            as u_e_t.
        delta_t: Difference between the time steps.
        jacobian_inverted_t: Jacobian matrix inverted and transposed, needed
            for calculation of global derivatives.
        elasticity_matrix: Elasticity matrix for the current element
            (c matrix).
        element_order: Order of the shape functions.
    """
    def inner(s: float, t: float) -> npt.NDArray:
        dn = gradient_local_shape_functions_2d(s, t, element_order)
        jacobian = np.dot(node_points, dn.T)
        jacobian_inverted_t = np.linalg.inv(jacobian).T
        jacobian_det = np.linalg.det(jacobian)

        r = local_to_global_coordinates(node_points, s, t, element_order)[0]
        b_opt = b_operator_global(
            node_points,
            jacobian_inverted_t,
            s,
            t,
            element_order
        )

        s_e = np.dot(b_opt, u_e_t)
        s_e_t_minus_1 = np.dot(b_opt, u_e_t_minus_1)
        s_e_t_minus_2 = np.dot(b_opt, u_e_t_minus_2)
        dt_s = (3*s_e-4*s_e_t_minus_1+s_e_t_minus_2)/(2*delta_t)

        return np.dot(
            dt_s.T,
            np.dot(
                elasticity_matrix.T,
                dt_s
            )
        ) * r * jacobian_det

    return quadratic_quadrature(inner, element_order)


#
# -------- Numerical calculations --------
#
def quadratic_quadrature(func: Callable, element_order: int) -> npt.NDArray:
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


def line_quadrature(func: Callable, element_order: int) -> npt.NDArray:
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

