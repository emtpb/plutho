"""Main module for the simulation of piezoelectric systems with thermal field."""

# Python standard libraries
import numpy as np
import numpy.typing as npt
import scipy.integrate as integrate
import scipy.sparse as sparse
import scipy.sparse.linalg as slin
from dataclasses import dataclass

import piezo_fem
from piezo_fem import mesh

@dataclass
class MaterialData:
    elasticity_matrix: npt.NDArray
    permittivity_matrix: npt.NDArray
    piezo_matrix: npt.NDArray
    density: float
    thermal_diffusivity: float
    heat_capacity: float
    alpha_m: float
    alpha_k: float

@dataclass
class SimulationData:
    delta_t: float
    number_of_time_steps: float
    gamma: float
    beta: float

@dataclass
class MeshData:
    nodes: npt.NDArray
    elements: npt.NDArray

def local_shape_functions(s, t):
    return np.array([1-s-t, s, t])

def gradient_local_shape_functions():
    return np.array([[-1, 1, 0],
                     [-1, 0, 1]])

def local_to_global_coordinates(node_points, s, t):
    # Transforms local s, t coordinates to global r, z coordinates
    return np.dot(node_points, local_shape_functions(s, t))

def b_operator_global(node_points, s, t, jacobian_inverted_T):
    # Get local shape functions and r (because of theta component)
    N = local_shape_functions(s, t)
    r = local_to_global_coordinates(node_points, s, t)[0]

    # Get gradients of local shape functions (s, t)
    dN = gradient_local_shape_functions()

    # Convert to gradients of (r, z) using jacobi matrix
    global_dN = np.dot(jacobian_inverted_T, dN)

    return np.array([
        [global_dN[0][0],               0, global_dN[0][1],               0, global_dN[0][2],               0],
        [              0, global_dN[1][0],               0, global_dN[1][1],               0, global_dN[1][2]],
        [global_dN[1][0], global_dN[0][0], global_dN[1][1], global_dN[0][1], global_dN[1][2], global_dN[0][2]],
        [         N[0]/r,               0,          N[1]/r,               0,          N[2]/r,               0],
    ])

def integral_M(node_points):
    def inner(s, t):
        N = local_shape_functions(s, t)
        
        # Since the simulation is axisymmetric it is necessary
        # to multiply with the radius in the integral (for the theta component (azimuth))
        r = local_to_global_coordinates(node_points, s, t)[0]

        # Get all combinations of shape function multiplied with each other
        return np.outer(N, N)*r

    return quadratic_quadrature(inner)

def integral_Ku(node_points, jacobian_inverted_T, elasticity_matrix):
    def inner(s, t):
        b_op = b_operator_global(node_points, s, t, jacobian_inverted_T)
        r = local_to_global_coordinates(node_points, s, t)[0]

        return np.dot(np.dot(b_op.T, elasticity_matrix), b_op)*r

    return quadratic_quadrature(inner)

def integral_KuV(node_points, jacobian_inverted_T, piezo_matrix):
    def inner(s, t):
        b_op = b_operator_global(node_points, s, t, jacobian_inverted_T)
        dN = gradient_local_shape_functions()
        global_dN = np.dot(jacobian_inverted_T, dN)
        r = local_to_global_coordinates(node_points, s, t)[0]

        return np.dot(np.dot(b_op.T, piezo_matrix.T), global_dN)*r
    
    return quadratic_quadrature(inner)

def integral_KVe(node_points, jacobian_inverted_T, permittivity_matrix):
    def inner(s, t):
        dN = gradient_local_shape_functions()
        global_dN = np.dot(jacobian_inverted_T, dN)
        r = local_to_global_coordinates(node_points, s, t)[0]
        
        return np.dot(np.dot(global_dN.T, permittivity_matrix), global_dN)*r
    
    return quadratic_quadrature(inner)

def integral_Ktheta(node_points, jacobian_inverted_T):
    def inner(s, t):
        dN = gradient_local_shape_functions()
        global_dN = np.dot(jacobian_inverted_T, dN)
        r = local_to_global_coordinates(node_points, s, t)[0]

        return np.dot(global_dN.T, global_dN)*r

    return quadratic_quadrature(inner)

def integral_theta_load(node_points, point_loss, thermal_diffusivity, density, heat_capacity):
    def inner(s, t):
        N = local_shape_functions(s, t)
        r = local_to_global_coordinates(node_points, s, t)[0]

        return np.array([
            N[0]*point_loss/(density*heat_capacity)/3,
            N[1]*point_loss/(density*heat_capacity)/3,
            N[2]*point_loss/(density*heat_capacity)/3
        ])

    return quadratic_quadrature(inner)


def quadratic_quadrature(func):
    weight_1 = 1/6
    weight_2 = 2/3
    return func(weight_1, weight_1)*weight_1 + \
        func(weight_2, weight_1)*weight_1 + \
        func(weight_1, weight_2)*weight_1

def line_quadrature(func):
    # TODO Analyze difference
    #return func(1/np.sqrt(3), 0) + func(-1/np.sqrt(3), 0)
    return integrate.quad(lambda x: func(x, 0), 0, 1)[0]

def assemble(mesh_data, material_data):
    # TODO Assembly takes to long rework this algorithm
    # Maybe the 2x2 matrix slicing is not very fast
    nodes = mesh_data.nodes

    number_of_nodes = len(nodes)
    Mu = sparse.lil_matrix((2*number_of_nodes, 2*number_of_nodes), dtype=np.float64)
    Ku = sparse.lil_matrix((2*number_of_nodes, 2*number_of_nodes), dtype=np.float64)
    KuV = sparse.lil_matrix((2*number_of_nodes, number_of_nodes), dtype=np.float64)
    KVe = sparse.lil_matrix((number_of_nodes, number_of_nodes), dtype=np.float64)
    Mtheta = sparse.lil_matrix((number_of_nodes, number_of_nodes), dtype=np.float64)
    Ktheta = sparse.lil_matrix((number_of_nodes, number_of_nodes), dtype=np.float64)

    for element in mesh_data.elements:
        dN = gradient_local_shape_functions()
        # Get node points of element in format
        # [x1 x2 x3]
        # [y1 y2 y3] where (xi, yi) are the coordinates for Node i
        node_points = np.array([
            [nodes[element[0]][0], nodes[element[1]][0], nodes[element[2]][0]],
            [nodes[element[0]][1], nodes[element[1]][1], nodes[element[2]][1]]
        ])
        jacobian = np.dot(node_points, dN.T)
        jacobian_inverted_T = np.linalg.inv(jacobian).T
        jacobian_det = np.linalg.det(jacobian)
        
        # TODO Check if its necessary to calculate all integrals
        # --> Dirichlet nodes could be leaved out?
        # Multiply with jac_det because its integrated with respect to local coordinates
        # Mutiply with 2*pi because theta is integrated from 0 to 2*pi
        Mu_e = material_data.density*integral_M(node_points)*jacobian_det*2*np.pi
        Ku_e = integral_Ku(node_points, jacobian_inverted_T, material_data.elasticity_matrix)*jacobian_det*2*np.pi
        KuV_e = integral_KuV(node_points, jacobian_inverted_T, material_data.piezo_matrix)*jacobian_det*2*np.pi
        KVe_e = integral_KVe(node_points, jacobian_inverted_T, material_data.permittivity_matrix)*jacobian_det*2*np.pi
        Mtheta_e = integral_M(node_points)*jacobian_det*2*np.pi
        Ktheta_e = material_data.thermal_diffusivity*integral_Ktheta(node_points, jacobian_inverted_T)*jacobian_det*2*np.pi

        # Now assemble all element matrices
        for local_p, global_p in enumerate(element):
            for local_q, global_q in enumerate(element):
                # M is returned as a 3x3 matrix (should be 6x6) because the values are the same
                # Only the diagonal elements have values
                Mu[2*global_p, 2*global_q] += Mu_e[local_p][local_q]
                Mu[2*global_p+1, 2*global_q+1] += Mu_e[local_p][local_q]

                # Ku is a 6x6 matrix and 2x2 matrices are sliced out of it
                Ku[2*global_p:2*global_p+2, 2*global_q:2*global_q+2] += Ku_e[2*local_p:2*local_p+2, 2*local_q:2*local_q+2]

                # KuV is a 6x3 matrix and 2x1 vectors are sliced out
                # [:, None] converts to a column vector
                KuV[2*global_p:2*global_p+2, global_q] += KuV_e[2*local_p:2*local_p+2, local_q][:, None] 

                # KVe is a 3x3 matrix and therefore the values can be set directly
                KVe[global_p, global_q] += KVe_e[local_p, local_q]

                # Mtheta and Ktheta are 3x3 matrices too
                Mtheta[global_p, global_q] += Mtheta_e[local_p, local_q]
                Ktheta[global_p, global_q] += Ktheta_e[local_p, local_q]

    # Calculate damping matrix
    Cu = material_data.alpha_m*Mu+material_data.alpha_k*Ku
    #Cu = np.zeros((2*number_of_nodes, 2*number_of_nodes))

    zeros1x1 = np.zeros((number_of_nodes, number_of_nodes))
    zeros2x1 = np.zeros((2*number_of_nodes, number_of_nodes))
    zeros1x2 = np.zeros((number_of_nodes, 2*number_of_nodes))

    # Calculate block matrices
    M = sparse.bmat([
        [Mu, zeros2x1, zeros2x1],
        [zeros1x2, zeros1x1, zeros1x1],
        [zeros1x2, zeros1x1, Mtheta]
    ])
    C = sparse.bmat([
        [Cu, zeros2x1, zeros2x1],
        [zeros1x2, zeros1x1, zeros1x1],
        [zeros1x2, zeros1x1, zeros1x1]
    ])
    K = sparse.bmat([
        [Ku, KuV, zeros2x1],
        [KuV.T, -1*KVe, zeros1x1],
        [zeros1x2, zeros1x1, Ktheta]
    ])

    return M.tolil(), C.tolil(), K.tolil()

def charge_integral_u(node_points, u_e, piezo_matrix, jacobian_inverted_T):
    def inner(s, t):
        r = local_to_global_coordinates(node_points, s, t)[0]
        b_opt_global = b_operator_global(node_points, s, t, jacobian_inverted_T)
        # [1] commponent is taken because the normal of the boundary line is in z-direction
        return np.dot(np.dot(piezo_matrix, b_opt_global), u_e)[1]*r

    return line_quadrature(inner)

def charge_integral_v(node_points, v_e, permittivity_matrix, jacobian_inverted_T):
    def inner(s, t):
        r = local_to_global_coordinates(node_points, s, t)[0]
        dN = gradient_local_shape_functions()
        global_dN = np.dot(jacobian_inverted_T, dN)

        return np.dot(np.dot(permittivity_matrix, global_dN), v_e)[1]*r

    return line_quadrature(inner)

def calculate_charge(u, permittivity_matrix, piezo_matrix, elements, nodes):
    number_of_nodes = len(nodes)
    q = 0

    for element in elements:
        dN = gradient_local_shape_functions()
        node_points = np.array([
            [nodes[element[0]][0], nodes[element[1]][0], nodes[element[2]][0]],
            [nodes[element[0]][1], nodes[element[1]][1], nodes[element[2]][1]]
        ])

        # TODO Check why the jacobian_det did not work
        # It should work with line elements instead of triangle elements
        jacobian = np.dot(node_points, dN.T)
        jacobian_inverted_T = np.linalg.inv(jacobian).T
        #jacobian_det = np.linalg.det(jacobian)

        # Since its a line integral along the top edge of the model where the triangles are aligned in a line
        # it is necessarry to use a different determinant. In this case the determinant is the difference between the
        # first 2 points of the triangle (which are always on the boundary line in gmsh)
        jacobian_det = nodes[element[0]][0]-nodes[element[1]][0]

        u_e = np.array([u[2*element[0]],
                        u[2*element[0]+1],
                        u[2*element[1]],
                        u[2*element[1]+1],
                        u[2*element[2]],
                        u[2*element[2]+1]])
        Ve_e = np.array([u[element[0]+2*number_of_nodes],
                         u[element[1]+2*number_of_nodes],
                         u[element[2]+2*number_of_nodes]])

        q_u = charge_integral_u(node_points, u_e, piezo_matrix, jacobian_inverted_T)*2*np.pi*jacobian_det
        q_v = charge_integral_v(node_points, Ve_e, permittivity_matrix, jacobian_inverted_T)*2*np.pi*jacobian_det
        
        q += q_u - q_v

    return q

def solve_time(M, C, K, mesh_data, material_data, simulation_data, dirichlet_nodes, dirichlet_values, electrode_elements):
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
    u = np.zeros((M.shape[0], number_of_time_steps), dtype=np.float64) # Displacement u which is calculated
    v = np.zeros((M.shape[0], number_of_time_steps), dtype=np.float64) # Displacement u derived after t (du/dt)
    a = np.zeros((M.shape[0], number_of_time_steps), dtype=np.float64) # v derived after u (d^2u/dt^2)

    q = np.zeros(number_of_time_steps, dtype=np.float64) # Charge calculated during simulation
    power_loss = np.zeros((len(mesh_data.elements), number_of_time_steps), dtype=np.float64)
    e = np.zeros((len(mesh_data.elements), number_of_time_steps, 3), dtype=np.float64)

    M, C, K = apply_dirichlet_bc(M, C, K, dirichlet_nodes[0], dirichlet_nodes[1], number_of_nodes)

    K_star = (K+gamma/(beta*delta_t)*C+1/(beta*delta_t**2)*M).tocsr()

    I = np.zeros((number_of_elements, number_of_time_steps), dtype=np.float64)
    J = np.zeros((number_of_elements, number_of_time_steps), dtype=np.float64)

    print("Starting simulation")
    for time_index in range(number_of_time_steps-1):
        # Calculate load vector and add dirichlet boundary conditions
        f = get_load_vector(dirichlet_nodes[0], 
                            dirichlet_values[0][time_index+1], 
                            dirichlet_nodes[1],
                            dirichlet_values[1][time_index+1],
                            mesh_data,
                            material_data,
                            power_loss[:, time_index]
        )

        # Perform Newmark method
        # Predictor step
        u_tilde = u[:, time_index] + delta_t*v[:, time_index] + delta_t**2/2*(1-2*beta)*a[:, time_index]
        v_tilde = v[:, time_index] + (1-gamma)*delta_t*a[:, time_index]

        # Solve for next time step
        u[:, time_index+1] = slin.spsolve(K_star, f-C*v_tilde+(1/(beta*delta_t**2)*M+gamma/(beta*delta_t)*C)*u_tilde)

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
        for index, element in enumerate(elements):
            if time_index < 2:
                break
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
            b_opt = b_operator_global(node_points, 1/3, 1/3, jacobian_inverted_T)
            dN = gradient_local_shape_functions()
            global_dN = np.dot(jacobian_inverted_T, dN)
            u_e = np.array([u[2*element[0], time_index],
                            u[2*element[0]+1, time_index],
                            u[2*element[1], time_index],
                            u[2*element[1]+1, time_index],
                            u[2*element[2], time_index],
                            u[2*element[2]+1, time_index]])
            Ve_e = np.array([u[element[0]+2*number_of_nodes, time_index],
                            u[element[1]+2*number_of_nodes, time_index],
                            u[element[2]+2*number_of_nodes, time_index]])

            
            S_e = np.dot(b_opt, u_e)
            E_e = -np.dot(global_dN, Ve_e)
            tau = 1.99e-11

            I[index, time_index+1] = np.dot(S_e.T, np.dot(material_data.elasticity_matrix.T, S_e))
            J[index, time_index+1] = np.dot(E_e.T, np.dot(material_data.piezo_matrix, S_e))

            #power_loss[index, time_index+1] = tau*(I[index, time_index]-2*I[index, time_index-1]+I[index, time_index-2])/delta_t**2 + \
            #    (I[index, time_index]-I[index, time_index-1])/delta_t - \
            #        (J[index, time_index]-J[index, time_index-1])/delta_t
            power_loss[index, time_index+1] = tau*(I[index, time_index]-2*I[index, time_index-1]+I[index, time_index-2])/delta_t**2

        if (time_index + 1) % 100 == 0:
            print(f"Finished time step {time_index+1}")

    return u, q, power_loss

def get_load_vector(nodes_u, values_u, nodes_v, values_v, mesh_data, materia_data, power_loss):
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
    # It needs to be assembled every step since the power is position-dependend
    f_theta = np.zeros(number_of_nodes)

    for index, element in enumerate(mesh_data.elements):
        dN = gradient_local_shape_functions()
        node_points = np.array([
            [nodes[element[0]][0], nodes[element[1]][0], nodes[element[2]][0]],
            [nodes[element[0]][1], nodes[element[1]][1], nodes[element[2]][1]]
        ])
        jacobian = np.dot(node_points, dN.T)
        jacobian_det = np.linalg.det(jacobian)

        #local_power_loss = np.array([
        #    power_loss[element[0]], power_loss[element[1]], power_loss[element[2]]
        #])

        f_theta_e = integral_theta_load(
            node_points,
            power_loss[index],
            materia_data.density,
            materia_data.heat_capacity,
            materia_data.thermal_diffusivity)*2*np.pi*jacobian_det

        for local_p, global_p in enumerate(element):
                f_theta[global_p] += f_theta_e[local_p]

    f[3*number_of_nodes:] = f_theta

    return f

def apply_dirichlet_bc(M, C, K, nodes_u, nodes_v, number_of_nodes):
    # Set rows of matrices to 0 and diagonal of K to 1 (at node points)

    # Matrices for u_r component
    for node in nodes_u:
        # Set rows to 0
        M[2*node, :] = 0
        C[2*node, :] = 0
        K[2*node, :] = 0

        # Set diagonal values to 1
        K[2*node, 2*node] = 1
        
    # Set bc for v
    # Offset because the V values are set in the latter part of the matrix
    offset = 2*number_of_nodes

    for node in nodes_v:
        # Set rows to 0
        M[node+offset, :] = 0
        C[node+offset, :] = 0
        K[node+offset, :] = 0
        
        # Set diagonal values to 1
        K[node+offset, node+offset] = 1

    # Currently no dirichlet bc for the temperature field -> TODO Instable?

    return M, C, K

def create_node_excitation(electrode_nodes, symaxis_nodes, ground_nodes, excitation, number_of_time_steps):
    # For "Electrode" set excitation function
    # "Symaxis" and "Ground" are set to 0

    # For displacement u set symaxis values to 0
    dirichlet_nodes_u = symaxis_nodes
    dirichlet_values_u = np.zeros((number_of_time_steps, len(dirichlet_nodes_u), 2))

    # For potential v set electrode to excitation and ground to 0
    dirichlet_nodes_v = np.concatenate((electrode_nodes, ground_nodes))
    dirichlet_values_v = np.zeros((number_of_time_steps, len(dirichlet_nodes_v)))

    # Set excitation value for electrode nodes points
    for time_index, excitation_value in enumerate(excitation):
        dirichlet_values_v[time_index, :len(electrode_nodes)] = excitation_value

    return [dirichlet_nodes_u, dirichlet_nodes_v], [dirichlet_values_u, dirichlet_values_v]