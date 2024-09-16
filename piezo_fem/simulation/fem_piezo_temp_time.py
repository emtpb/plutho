"""Main module for the simulation of piezoelectric systems with
thermal field."""

# Python standard libraries
from typing import Tuple
import numpy as np
import numpy.typing as npt
import scipy.sparse as sparse
import scipy.sparse.linalg as slin

# Local libraries
from .base import MaterialData, SimulationData, MeshData, \
    local_shape_functions, gradient_local_shape_functions, \
    local_to_global_coordinates, b_operator_global, integral_m, \
    integral_ku, integral_kuv, integral_kve, apply_dirichlet_bc, \
    quadratic_quadrature, line_quadrature


def integral_ktheta(
        node_points: npt.NDArray,
        jacobian_inverted_t: npt.NDArray) -> npt.NDArray:
    """Calculates the Ktheta integral.

    Parameters:
        node_points: List of node points [[x1, x2, ..], [y1, y2, ..]]
        jacobian_inverted_t: Jacobian matrix inverted and transposed, needed
            for calculation of global derivatives

    Returns:
        npt.NDArray: 3x3 Ktheta matrix for the given element.
    """
    def inner(s, t):
        dn = gradient_local_shape_functions()
        global_dn = np.dot(jacobian_inverted_t, dn)
        r = local_to_global_coordinates(node_points, s, t)[0]

        return np.dot(global_dn.T, global_dn)*r

    return quadratic_quadrature(inner)


def integral_theta_load(
        node_points: npt.NDArray,
        point_loss: npt.NDArray) -> npt.NDArray:
    """Returns the load value for the temperature field (f) for the specific
    element.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        point_loss: Loss power on each node (heat source)
    Returns:
        npt.NDArray: f vector value at the specific ndoe
    """
    def inner(s, t):
        n = local_shape_functions(s, t)
        r = local_to_global_coordinates(node_points, s, t)[0]

        return n*point_loss*r

    return quadratic_quadrature(inner)


def charge_integral_u(
        node_points: npt.NDArray,
        u_e: npt.NDArray,
        piezo_matrix: npt.NDArray,
        jacobian_inverted_t: npt.NDArray) -> float:
    """Calculates the integral of eBu of the given element.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        u_e: List of u values at the nodes of the triangle
            [u1_r, u1_z, u2_r, u2_z, u3_r, u3_z].
        piezo_matrix: Piezo matrix for the current element (e matrix).
        jacobian_inverted_t: Jacobian matrix inverted and transposed, needed
            for calculation of global derivatives.

    Returns:
        Integral of eBu of the current triangle.
    """
    def inner(s, t):
        r = local_to_global_coordinates(node_points, s, t)[0]
        b_opt_global = b_operator_global(
            node_points,
            s,
            t,
            jacobian_inverted_t)
        # [1] commponent is taken because the normal of the
        # boundary line is in z-direction
        return np.dot(np.dot(piezo_matrix, b_opt_global), u_e)[1]*r

    return line_quadrature(inner)


def charge_integral_v(
        node_points: npt.NDArray,
        v_e: npt.NDArray,
        permittivity_matrix: npt.NDArray,
        jacobian_inverted_t: npt.NDArray) -> float:
    """Calculates the integral of epsilonBVe of the given element.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        v_e: List of u values at the nodes of the triangle
            [v1, v2, v3].
        permittivity_matrix: Permittivity matrix for the current
            element (e matrix).
        jacobian_inverted_t: Jacobian matrix inverted and transposed, needed
            for calculation of global derivatives.

    Returns:
        Integral of epsilonBVe of the current triangle.
    """
    def inner(s, t):
        r = local_to_global_coordinates(node_points, s, t)[0]
        dn = gradient_local_shape_functions()
        global_dn = np.dot(jacobian_inverted_t, dn)

        return np.dot(np.dot(permittivity_matrix, global_dn), v_e)[1]*r

    return line_quadrature(inner)


def loss_integral_scs(
        node_points: npt.NDArray,
        u_e_t: npt.NDArray,
        u_e_t_minus_1: npt.NDArray,
        u_e_t_minus_2: npt.NDArray,
        delta_t: float,
        jacobian_inverted_t: npt.NDArray,
        elasticity_matrix: npt.NDArray):
    """Calculates the dS/dt*c*dS/dt integral. Since difference foward
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
    """
    def inner(s, t):
        b_opt = b_operator_global(node_points, s, t, jacobian_inverted_t)
        dt_s = np.dot(
            b_opt,
            (3*u_e_t-4*u_e_t_minus_1+u_e_t_minus_2)/(2*delta_t)
        )
        return np.dot(dt_s.T, np.dot(elasticity_matrix.T, dt_s))

    return quadratic_quadrature(inner)


def calculate_charge(
        u: npt.NDArray,
        permittivity_matrix: npt.NDArray,
        piezo_matrix: npt.NDArray,
        elements: npt.NDArray,
        nodes) -> float:
    """Calculates the charge of the given elements.

    Parameters:
        u: List of u values for every node.
        permittivity_matrix: Permittivity matrix for the
            current element (epsilon matrix).
        piezo_matrix: Piezo matrix for the current element (e matrix).
        elements: Elements for which the charge shall be calculated.
        nodes: All nodes used in the simulation.
    """
    number_of_nodes = len(nodes)
    q = 0

    for element in elements:
        dn = gradient_local_shape_functions()
        node_points = np.array([
            [nodes[element[0]][0], nodes[element[1]][0], nodes[element[2]][0]],
            [nodes[element[0]][1], nodes[element[1]][1], nodes[element[2]][1]]
        ])

        # TODO Check why the jacobian_det did not work
        # It should work with line elements instead of triangle elements
        jacobian = np.dot(node_points, dn.T)
        jacobian_inverted_t = np.linalg.inv(jacobian).T
        # jacobian_det = np.linalg.det(jacobian)

        # Since its a line integral along the top edge of the model where the
        # triangles are aligned in a line.
        # it is necessarry to use a different determinant. In this case the
        # determinant is the difference between the.
        # first 2 points of the triangle (which are always on the boundary
        # line in gmsh).
        jacobian_det = nodes[element[0]][0]-nodes[element[1]][0]

        u_e = np.array([u[2*element[0]],
                        u[2*element[0]+1],
                        u[2*element[1]],
                        u[2*element[1]+1],
                        u[2*element[2]],
                        u[2*element[2]+1]])
        ve_e = np.array([u[element[0]+2*number_of_nodes],
                         u[element[1]+2*number_of_nodes],
                         u[element[2]+2*number_of_nodes]])

        q_u = charge_integral_u(
            node_points,
            u_e,
            piezo_matrix,
            jacobian_inverted_t)*2*np.pi*jacobian_det
        q_v = charge_integral_v(
            node_points,
            ve_e,
            permittivity_matrix,
            jacobian_inverted_t)*2*np.pi*jacobian_det

        q += q_u - q_v

    return q


class PiezoThermoSim:
    """Class for the coupled simulation of mechanical-electric and
    thermal field.

    Attributes:
        mesh_data: Contains the mesh information.
        material_data: Contains the information about the materials.
        simulation_data: Contains the information about the simulation.
        dirichlet_nodes: List of nodes for which dirichlet values are set.
        dirichlet_values: List of values of the corresponding dirichlet nodes
            per time step.
    """

    mesh_data: MeshData
    material_data: MaterialData
    simulation_data: SimulationData

    dirichlet_nodes: npt.NDArray
    dirichlet_values: npt.NDArray

    def __init__(
            self,
            mesh_data: MeshData,
            material_data: MaterialData,
            simulation_data: SimulationData):
        """Constructor of PiezoThermo.

        Parameters:
            mesh_data: MeshData format.
            material_data: MaterialData format.
            simulation_data: SimulationData format.
        """
        self.mesh_data = mesh_data
        self.material_data = material_data
        self.simulation_data = simulation_data

    def assemble(self) \
            -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Assembles the FEM matrices based on the data classes set in this
        class (mesh_data, material_data).
        """
        # TODO Assembly takes to long rework this algorithm
        # Maybe the 2x2 matrix slicing is not very fast
        nodes = self.mesh_data.nodes

        number_of_nodes = len(nodes)
        mu = sparse.lil_matrix(
            (2*number_of_nodes, 2*number_of_nodes),
            dtype=np.float64)
        ku = sparse.lil_matrix(
            (2*number_of_nodes, 2*number_of_nodes),
            dtype=np.float64)
        kuv = sparse.lil_matrix(
            (2*number_of_nodes, number_of_nodes),
            dtype=np.float64)
        kve = sparse.lil_matrix(
            (number_of_nodes, number_of_nodes),
            dtype=np.float64)
        ctheta = sparse.lil_matrix(
            (number_of_nodes, number_of_nodes),
            dtype=np.float64)
        ktheta = sparse.lil_matrix(
            (number_of_nodes, number_of_nodes),
            dtype=np.float64)

        for element in self.mesh_data.elements:
            dn = gradient_local_shape_functions()
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

            # TODO Check if its necessary to calculate all integrals
            # --> Dirichlet nodes could be leaved out?
            # Multiply with jac_det because its integrated with respect to
            # local coordinates.
            # Mutiply with 2*pi because theta is integrated from 0 to 2*pi
            mu_e = (
                self.material_data.density
                * integral_m(node_points)
                * jacobian_det * 2 * np.pi
            )
            ku_e = (
                integral_ku(
                    node_points,
                    jacobian_inverted_t,
                    self.material_data.elasticity_matrix)
                * jacobian_det * 2 * np.pi
            )
            kuv_e = (
                integral_kuv(
                    node_points,
                    jacobian_inverted_t,
                    self.material_data.piezo_matrix)
                * jacobian_det * 2 * np.pi
            )
            kve_e = (
                integral_kve(
                    node_points,
                    jacobian_inverted_t,
                    self.material_data.permittivity_matrix)
                * jacobian_det * 2 * np.pi
            )
            ctheta_e = (
                integral_m(node_points)
                * self.material_data.density
                * self.material_data.heat_capacity
                * jacobian_det * 2 * np.pi
            )
            ktheta_e = (
                integral_ktheta(
                    node_points,
                    jacobian_inverted_t)
                * self.material_data.thermal_conductivity
                * jacobian_det*2*np.pi
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
                    kve[global_p, global_q] += kve_e[local_p, local_q]

                    # Mtheta_e and Ktheta_e are 3x3 matrices too.
                    ctheta[global_p, global_q] += ctheta_e[local_p, local_q]
                    ktheta[global_p, global_q] += ktheta_e[local_p, local_q]

        # Calculate damping matrix
        cu = self.material_data.alpha_m * mu + self.material_data.alpha_k * ku

        # Calculate block matrices
        zeros1x1 = np.zeros((number_of_nodes, number_of_nodes))
        zeros2x1 = np.zeros((2*number_of_nodes, number_of_nodes))
        zeros1x2 = np.zeros((number_of_nodes, 2*number_of_nodes))

        m = sparse.bmat([
            [mu, zeros2x1, zeros2x1],
            [zeros1x2, zeros1x1, zeros1x1],
            [zeros1x2, zeros1x1, zeros1x1]
        ])
        c = sparse.bmat([
            [cu, zeros2x1, zeros2x1],
            [zeros1x2, zeros1x1, zeros1x1],
            [zeros1x2, zeros1x1, ctheta]
        ])
        k = sparse.bmat([
            [ku, kuv, zeros2x1],
            [kuv.T, -1*kve, zeros1x1],
            [zeros1x2, zeros1x1, ktheta]
        ])

        return m.tolil(), c.tolil(), k.tolil()

    def set_dirichlet_boundary_conditions(
            self,
            electrode_nodes: npt.NDArray,
            symaxis_nodes: npt.NDArray,
            ground_nodes: npt.NDArray,
            electrode_excitation: npt.NDArray):
        """Sets the dirichelt boundary condition for the simulation
        given the nodes of electrode, symaxis and ground. The electrode
        nodes are set to the given electrode_excitation.
        The symaxis nodes are the due to the axisymmetric model 
        and the ground nodes are set to 0."""
        number_of_time_steps = self.simulation_data.number_of_time_steps
        # For "Electrode" set excitation function
        # "Symaxis" and "Ground" are set to 0

        # For displacement u set symaxis values to 0.
        # Zeros are set for u_r and u_z but the u_z component is not used.
        dirichlet_nodes_u = symaxis_nodes
        dirichlet_values_u = np.zeros(
            (number_of_time_steps, len(dirichlet_nodes_u), 2))

        # For potential v set electrode to excitation and ground to 0
        dirichlet_nodes_v = np.concatenate((electrode_nodes, ground_nodes))
        dirichlet_values_v = np.zeros(
            (number_of_time_steps, len(dirichlet_nodes_v)))

        # Set excitation value for electrode nodes points
        for time_index, excitation_value in enumerate(electrode_excitation):
            dirichlet_values_v[time_index, :len(electrode_nodes)] = \
                excitation_value

        self.dirichlet_nodes = [dirichlet_nodes_u, dirichlet_nodes_v]
        self.dirichlet_values = [dirichlet_values_u, dirichlet_values_v]

    def solve_time(
            self,
            dirichlet_nodes: npt.NDArray,
            dirichlet_values: npt.NDArray,
            electrode_elements: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        """Runs the simulation using the given m, c and k matrices as well as the
        excitation. Calculates the displacement field, potential field and
        the thermal field of the given model.

        Parameters:
            m: Mass matrix
            c: Damping matrix
            k: Stiffness matrix
            mesh_data: Used for"""
        """Effective stiffness implementation"""
        number_of_time_steps = simulation_data.number_of_time_steps
        beta = simulation_data.beta
        gamma = simulation_data.gamma
        delta_t = simulation_data.delta_t
        elements = mesh_data.elements
        nodes = mesh_data.nodes
        number_of_elements = len(elements)
        number_of_nodes = len(nodes)

        # Init arrays
        u = np.zeros((m.shape[0], number_of_time_steps), dtype=np.float64) # Displacement u which is calculated
        v = np.zeros((m.shape[0], number_of_time_steps), dtype=np.float64) # Displacement u derived after t (du/dt)
        a = np.zeros((m.shape[0], number_of_time_steps), dtype=np.float64) # v derived after u (d^2u/dt^2)

        q = np.zeros(number_of_time_steps, dtype=np.float64) # Charge calculated during simulation (for impedance)

        m, c, k = apply_dirichlet_bc(m, c, k, dirichlet_nodes[0], dirichlet_nodes[1], number_of_nodes)

        K_star = (k+gamma/(beta*delta_t)*c+1/(beta*delta_t**2)*m).tocsr()

        power_loss = np.zeros((number_of_elements, number_of_time_steps), dtype=np.float64) # Power loss calculated during simulation (for thermal field)
        
        print("Starting simulation")
        for time_index in range(number_of_time_steps-1):
            # Calculate load vector and add dirichlet boundary conditions
            f = get_load_vector(dirichlet_nodes[0], 
                                dirichlet_values[0][time_index+1], 
                                dirichlet_nodes[1],
                                dirichlet_values[1][time_index+1],
                                mesh_data,
                                power_loss[:, time_index]
            )

            # Perform Newmark method
            # Predictor step
            u_tilde = u[:, time_index] + delta_t*v[:, time_index] + delta_t**2/2*(1-2*beta)*a[:, time_index]
            v_tilde = v[:, time_index] + (1-gamma)*delta_t*a[:, time_index]

            # Solve for next time step
            u[:, time_index+1] = slin.spsolve(K_star, f-c*v_tilde+(1/(beta*delta_t**2)*m+gamma/(beta*delta_t)*c)*u_tilde)

            # Perform corrector step
            a[:, time_index+1] = (u[:, time_index+1]-u_tilde)/(beta*delta_t**2)
            v[:, time_index+1] = v_tilde + gamma*delta_t*a[:, time_index+1]

            # Calculate charge
            q[time_index+1] = calculate_charge(
                u[:, time_index+1],
                material_data.permittivity_matrix,
                material_data.piezo_matrix,
                electrode_elements,
                mesh_data.nodes
            )

            # Calculate power_loss
            # TODO Calculate charge and power loss together (one loop)
            if time_index > 1:
                for element_index, element in enumerate(elements):
                    node_points = np.array([
                        [nodes[element[0]][0], nodes[element[1]][0], nodes[element[2]][0]],
                        [nodes[element[0]][1], nodes[element[1]][1], nodes[element[2]][1]]
                    ])

                    dN = gradient_local_shape_functions()
                    node_points = np.array([
                        [nodes[element[0]][0], nodes[element[1]][0], nodes[element[2]][0]],
                        [nodes[element[0]][1], nodes[element[1]][1], nodes[element[2]][1]]
                    ])
                    jacobian = np.dot(node_points, dN.T)
                    jacobian_inverted_T = np.linalg.inv(jacobian).T
                    jacobian_det = np.linalg.det(jacobian)
                    dN = gradient_local_shape_functions()
                    u_e = np.array([u[2*element[0], time_index],
                                    u[2*element[0]+1, time_index],
                                    u[2*element[1], time_index],
                                    u[2*element[1]+1, time_index],
                                    u[2*element[2], time_index],
                                    u[2*element[2]+1, time_index]])
                    u_e_t_minus_1 = np.array([
                                    u[2*element[0], time_index-1],
                                    u[2*element[0]+1, time_index-1],
                                    u[2*element[1], time_index-1],
                                    u[2*element[1]+1, time_index-1],
                                    u[2*element[2], time_index-1],
                                    u[2*element[2]+1, time_index-1]])
                    u_e_t_minus_2 = np.array([
                                    u[2*element[0], time_index-2],
                                    u[2*element[0]+1, time_index-2],
                                    u[2*element[1], time_index-2],
                                    u[2*element[1]+1, time_index-2],
                                    u[2*element[2], time_index-2],
                                    u[2*element[2]+1, time_index-2]])

                    loss_value = loss_integral_scs(node_points, u_e, u_e_t_minus_1, u_e_t_minus_2, delta_t, jacobian_inverted_T, material_data.elasticity_matrix)*2*np.pi*jacobian_det
                    power_loss[element_index, time_index+1] = material_data.tau*loss_value

            if (time_index + 1) % 100 == 0:
                print(f"Finished time step {time_index+1}")

        return u, q, power_loss

    def get_load_vector(nodes_u, values_u, nodes_v, values_v, mesh_data, power_loss):
        # For u and v the load vector is set to the corresponding dirichlet values given by values_u and values_v
        # at the nodes nodes_u and nodes_v since there is no inner charge and no forces given.
        # For the temperature field the load vector represents the temperature sources given by the mechanical losses.
        nodes = mesh_data.nodes
        number_of_nodes = len(nodes)

        # Can be initialized to 0 because external load and volume charge density is 0.
        f = np.zeros(4*number_of_nodes, dtype=np.float64)
        
        # Set dirichlet values for u_r only
        for node, value in zip(nodes_u, values_u):
            f[2*node] = value[0]

        # Set dirichlet values for v
        # Use offset because v nodes have higher index than u nodes
        offset = 2*number_of_nodes

        for node, value in zip(nodes_v, values_v):
            f[node+offset] = value

        # Calculate for theta load
        # It needs to be assembled every step and every element since the power is time- and position-dependent
        f_theta = np.zeros(number_of_nodes)

        for element_index, element in enumerate(mesh_data.elements):
            dN = gradient_local_shape_functions()
            node_points = np.array([
                [nodes[element[0]][0], nodes[element[1]][0], nodes[element[2]][0]],
                [nodes[element[0]][1], nodes[element[1]][1], nodes[element[2]][1]]
            ])
            jacobian = np.dot(node_points, dN.T)
            jacobian_det = np.linalg.det(jacobian)

            point_loss = np.ones(3) * power_loss[element_index]/3

            f_theta_e = integral_theta_load(
                node_points,
                point_loss)*2*np.pi*jacobian_det

            for local_p, global_p in enumerate(element):
                    f_theta[global_p] += f_theta_e[local_p]

        f[3*number_of_nodes:] = f_theta

        return f