import numpy as np
import os
import scipy.integrate as integrate
import scipy.sparse as sparse
import scipy.sparse.linalg as slin
import matplotlib.pyplot as plt
from parse_gmsh import MeshParser

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

def integral_M(node_points, density):
    def inner(s, t):
        N = local_shape_functions(s, t)
        
        # Since the simulation is axisymmetric it is necessary
        # to multiply with the radius in the integral (for the theta component (azimuth))
        r = local_to_global_coordinates(node_points, s, t)[0]

        # Get all combinations of shape function multiplied with each other
        return density*np.outer(N, N)*r

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

def assemble(nodes, elements, density, elasticity_matrix, piezo_matrix, permittivity_matrix, alpha_m, alpha_k):
    # TODO Assembly takes to long rework this algorithm
    # Maybe the 2x2 matrix slicing is not very fast
    
    number_of_nodes = len(nodes)
    Mu = sparse.lil_matrix((2*number_of_nodes, 2*number_of_nodes), dtype=np.float64)
    Ku = sparse.lil_matrix((2*number_of_nodes, 2*number_of_nodes), dtype=np.float64)
    KuV = sparse.lil_matrix((2*number_of_nodes, number_of_nodes), dtype=np.float64)
    KVe = sparse.lil_matrix((number_of_nodes, number_of_nodes), dtype=np.float64)

    for element in elements:
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
        Mu_e = integral_M(node_points, density)*jacobian_det*2*np.pi
        Ku_e = integral_Ku(node_points, jacobian_inverted_T, elasticity_matrix)*jacobian_det*2*np.pi
        KuV_e = integral_KuV(node_points, jacobian_inverted_T, piezo_matrix)*jacobian_det*2*np.pi
        KVe_e = integral_KVe(node_points, jacobian_inverted_T, permittivity_matrix)*jacobian_det*2*np.pi

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

    # Calculate damping matrix
    Cu = alpha_m*Mu+alpha_k*Ku
    #Cu = np.zeros((2*number_of_nodes, 2*number_of_nodes))

    # Calculate block matrices
    M = sparse.bmat([
        [Mu, np.zeros((2*number_of_nodes, number_of_nodes))],
        [np.zeros((1*number_of_nodes, 2*number_of_nodes)), np.zeros((number_of_nodes, number_of_nodes))]
    ])
    C = sparse.bmat([
        [Cu, np.zeros((2*number_of_nodes, number_of_nodes))],
        [np.zeros((1*number_of_nodes, 2*number_of_nodes)), np.zeros((number_of_nodes, number_of_nodes))]
    ])
    K = sparse.bmat([
        [Ku, KuV],
        [KuV.T, -1*KVe]
    ])

    return M.tolil(), C.tolil(), K.tolil()

def charge_integral_u(node_points, u_e, piezo_matrix, jacobian_inverted_T):
    def inner(s, t):
        r = local_to_global_coordinates(node_points, s, t)[0]
        b_opt_global = b_operator_global(node_points, s, t, jacobian_inverted_T)
        # [1] commponent is taken because the normal of the boundary line is in z-direction
        return np.dot(np.dot(piezo_matrix, b_opt_global), u_e)[1]*r

        # TODO Check why this variant is different
        #b = np.array([
        #    [global_dN[0][0],               0],
        #    [              0, global_dN[1][0]],
        #    [global_dN[1][0], global_dN[0][0]],
        #    [         N[0]/r,               0],
        #])

        #return np.dot(np.dot(piezo_matrix, b), np.dot(u_e, N))[1]*r

    return line_quadrature(inner)

def charge_integral_v(node_points, v_e, permittivity_matrix, jacobian_inverted_T):
    def inner(s, t):
        r = local_to_global_coordinates(node_points, s, t)[0]
        dN = gradient_local_shape_functions()
        global_dN = np.dot(jacobian_inverted_T, dN)

        return np.dot(np.dot(permittivity_matrix, global_dN), v_e)[1]*r

    return line_quadrature(inner)

def calculate_charge(u, permittivity_matrix, piezo_matrix, elements, nodes, number_of_nodes):
    q = 0

    for element in elements:
        node_points = np.array([
            [nodes[element[0]][0], nodes[element[1]][0], nodes[element[2]][0]],
            [nodes[element[0]][1], nodes[element[1]][1], nodes[element[2]][1]]
        ])

        #dN = gradient_local_shape_functions()
        #jacobian = np.dot(node_points, dN.T)
        #jacobian_inverted_T = np.linalg.inv(jacobian).T
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

        q_u = charge_integral_u(node_points, u_e, piezo_matrix, jacobian_inverted_T)*jacobian_det*2*np.pi
        q_v = charge_integral_v(node_points, Ve_e, permittivity_matrix, jacobian_inverted_T)*2*np.pi*jacobian_det
        
        q += q_u - q_v

    return q

def get_electrode_triangles(electrode_elements, all_elements):
    triangle_elements = []
    for element in electrode_elements:
        for check_element in all_elements:
            if element[0] in check_element and element[1] in check_element:
                triangle_elements.append(check_element)
                break

    return triangle_elements

def solve_time(M, C, K, delta_t, number_of_time_steps, dirichlet_nodes, dirichlet_values, gamma, beta, number_of_nodes, electrode_elements):
    # Init arrays
    u = np.zeros((M.shape[0], number_of_time_steps), dtype=np.float64) # Displacement u which is calculated
    v = np.zeros((M.shape[0], number_of_time_steps), dtype=np.float64) # Displacement u derived after t (du/dt)
    a = np.zeros((M.shape[0], number_of_time_steps), dtype=np.float64) # v derived after u (d^2u/dt^2)

    q = np.zeros(number_of_time_steps, dtype=np.float64)


    M, C, K = apply_dirichlet_effective_stiffness(M, C, K, dirichlet_nodes[0], dirichlet_nodes[1], number_of_nodes)
    #np.savetxt("M.txt", M.todense())
    #np.savetxt("C.txt", C.todense())
    #np.savetxt("K.txt", K.todense())

    M_star = (M + gamma*delta_t*C + beta*delta_t**2*K).tocsr()
    #M_star = apply_dirichlet_m_star(M_star.tolil(), dirichlet_nodes[0], dirichlet_nodes[1], number_of_nodes, beta, delta_t)
    #dense = M_star.todense()
    #print("Condition number", np.linalg.cond(dense))
    #print("Determinant", np.linalg.det(dense))
    #print("Rang of matrix", np.linalg.matrix_rank(dense))
    #print("Size of the matrix", dense.shape)

    print("Starting simulation")
    for time_index in range(number_of_time_steps-1):
        # Calculate load vector and add dirichlet boundary conditions
        f = get_load_vector(dirichlet_nodes[0], dirichlet_values[0][time_index+1], dirichlet_nodes[1], dirichlet_values[1][time_index+1], number_of_nodes)

        # Perform Newmark method
        # Predictor step
        u_tilde = u[:, time_index] + delta_t*v[:, time_index] + delta_t**2/2*(1-2*beta)*a[:, time_index]
        v_tilde = v[:, time_index] + (1-gamma)*delta_t*a[:, time_index]


        # Solve for next time step
        a[:, time_index+1] = slin.spsolve(M_star, f-1*K*u_tilde-1*C*v_tilde)

        # Perform corrector step
        u[:, time_index+1] = u_tilde + beta*delta_t**2*a[:, time_index+1]
        v[:, time_index+1] = v_tilde + gamma*delta_t*a[:, time_index+1]

        # Calculate charge
        q[time_index] = calculate_charge(u[:, time_index], permittivity_matrix, piezo_matrix, electrode_elements, nodes, number_of_nodes)

        print(f"Finished time step {time_index+1}")

    return u, q

def solve_time_effective_stiffness(M, C, K, delta_t, number_of_time_steps, dirichlet_nodes, dirichlet_values, gamma, beta, number_of_nodes, electrode_elements, nodes):
    # Init arrays
    u = np.zeros((M.shape[0], number_of_time_steps), dtype=np.float64) # Displacement u which is calculated
    v = np.zeros((M.shape[0], number_of_time_steps), dtype=np.float64) # Displacement u derived after t (du/dt)
    a = np.zeros((M.shape[0], number_of_time_steps), dtype=np.float64) # v derived after u (d^2u/dt^2)

    q = np.zeros(number_of_time_steps, dtype=np.float64)
    q_u = np.zeros(number_of_time_steps, dtype=np.float64)
    q_v = np.zeros(number_of_time_steps, dtype=np.float64)


    M, C, K = apply_dirichlet_effective_stiffness(M, C, K, dirichlet_nodes[0], dirichlet_nodes[1], number_of_nodes)
    #np.savetxt("M.txt", M.todense())
    #np.savetxt("C.txt", C.todense())
    #np.savetxt("K.txt", K.todense())

    K_star = (K+gamma/(beta*delta_t)*C+1/(beta*delta_t**2)*M).tocsr()
    #dense = K_star.todense()
    #print("Condition number", np.linalg.cond(dense))
    #print("Determinant", np.linalg.det(dense))
    #print("Rang of matrix", np.linalg.matrix_rank(dense))
    #print("Size of the matrix", dense.shape)

    print("Starting simulation")
    for time_index in range(number_of_time_steps-1):
        # Calculate load vector and add dirichlet boundary conditions
        f = get_load_vector(dirichlet_nodes[0], dirichlet_values[0][time_index+1], dirichlet_nodes[1], dirichlet_values[1][time_index+1], number_of_nodes)

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
        q[time_index+1], q_u[time_index+1], q_v[time_index+1] = calculate_charge(u[:, time_index+1], permittivity_matrix, piezo_matrix, electrode_elements, nodes, number_of_nodes)

        print(f"Finished time step {time_index+1}")

    np.savetxt("q_u.txt", q_u)
    np.savetxt("q_v.txt", q_v)
    np.savetxt("q", q)

    return u, q

def get_load_vector(nodes_u, values_u, nodes_v, values_v, number_of_nodes):
    # Can be initialized to 0 because external load and volume charge density is 0.
    f = np.zeros(3*number_of_nodes, dtype=np.float64)
    
    # Set dirichlet values for u_r only
    for node, value in zip(nodes_u, values_u):
        f[2*node] = value[0]

    # Set dirichlet values for v
    # Use offset because v nodes have higher index than u nodes
    offset = 2*number_of_nodes

    for node, value in zip(nodes_v, values_v):
        f[node+offset] = value

    return f

def apply_dirichlet_effective_stiffness(M, C, K, nodes_u, nodes_v, number_of_nodes):
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

    return M, C, K

def create_node_excitation(parser, excitation, number_of_time_steps):
    # For "Electrode" set excitation function
    # "Symaxis" and "Ground" are set to 0
    electrode_nodes = parser.getNodesInPhysicalGroup("Electrode")
    symaxis_nodes = parser.getNodesInPhysicalGroup("Symaxis")
    ground_nodes = parser.getNodesInPhysicalGroup("Ground")

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

def create_vector_field_as_csv(u, nodes, folder_path):
    # TODO Add check if folder exists
    number_of_nodes = len(nodes)
    number_of_time_steps = u.shape[1]

    vector_field = np.zeros((number_of_time_steps, number_of_nodes, 5))
    for time_step in range(number_of_time_steps):
        for node_index, node in enumerate(nodes):
            current_u_r = u[2*node_index, time_step]
            current_u_z = u[2*node_index+1, time_step]
            current_v = u[2*number_of_nodes+node_index, time_step]
            vector_field[time_step, node_index, 0] = node[0]
            vector_field[time_step, node_index, 1] = node[1]
            vector_field[time_step, node_index, 2] = current_u_r
            vector_field[time_step, node_index, 3] = current_u_z
            vector_field[time_step, node_index, 4] = current_v
        
    for time_step in range(number_of_time_steps):
        current_file_path = os.path.join(folder_path, f"u_{time_step}.csv")
        field = vector_field[time_step]

        text = f"r,z,u_r,u_z,v\n"
        for node_index in range(number_of_nodes):
            r = field[node_index][0]
            z = field[node_index][1]
            u_r = field[node_index][2]
            u_z = field[node_index][3]
            v = field[node_index][4]
            text += f"{r},{z},{u_r},{u_z},{v}\n"
            
        with open(current_file_path, "w", encoding="UTF-8") as fd:
            fd.write(text)  

def calculate_impedance(q, excitation, DELTA_T):
    sample_frequency = 1/DELTA_T

    # If size is not power of 2 append zeros
    #for i in range(10):
    #    if len(q) < 2**i:
    #        np.append(q, np.zeros(2**i-len(q))) 
    #        np.append(excitation, np.zeros(2**i-len(q)))
    #        break
        
    excitation_fft = np.fft.fft(excitation)
    q_fft = np.fft.fft(q)

    # since f=k*sample_frequency/N
    frequencies = np.arange(len(q))*sample_frequency/len(q)
    impedance = excitation_fft/(2*np.pi*1j*frequencies*q_fft)

    plt.plot(frequencies, np.abs(impedance), label="|Z(w)|")
    plt.plot(frequencies, np.abs(excitation_fft), label="|V(w)|")
    plt.plot(frequencies, np.abs(q_fft), label="|Q(w)|")
    plt.xlabel("Frequency f / Hz")
    plt.ylabel("Impedence |Z| / $\Omega$")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Time parameters
    #NUMBER_TIME_STEPS = 100
    #NUMBER_TIME_STEPS = int(0.5*8192)
    NUMBER_TIME_STEPS = 8192
    DELTA_T = 1e-8
    time_list = np.arange(0, NUMBER_TIME_STEPS)*DELTA_T

    GAMMA = 0.5
    BETA = 0.25

    # Material parameters
    rho = 7800
    alpha_M = 1.267e5
    alpha_K = 6.259e-10
    elasticity_matrix = np.array([
        [1.19e11, 0.83e11,       0, 0.84e11],
        [0.83e11, 1.17e11,       0, 0.83e11],
        [      0,       0, 0.21e11,       0],
        [0.84e11, 0.83e11,       0, 1.19e11]
    ])
    permittivity_matrix = np.diag([8.15e-9, 6.58e-9])
    piezo_matrix = np.array([
        [0, 0, 12.09, 0],
        [-6.03, 15.49, 0, -6.03]
    ])

    excitation = np.zeros(NUMBER_TIME_STEPS)
    excitation[1:10] = np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])

    # Load mesh
    mesh_file = "piezo.msh"
    parser = MeshParser(mesh_file)

    nodes = parser.nodes.copy()
    elements = parser.getTriangleElements()
    number_of_nodes = len(nodes)

    print("Anzahl Nodes:", number_of_nodes)
    print("Anzahl Elemente:", len(elements))

    dirichlet_nodes, dirichlet_values = create_node_excitation(parser, excitation, NUMBER_TIME_STEPS)

    electrode_elements = parser.getElementsInPhysicalGroup("Electrode")
    electrode_triangles = get_electrode_triangles(electrode_elements, elements)

    M, C, K = assemble(nodes, elements, rho, elasticity_matrix, piezo_matrix, permittivity_matrix, alpha_M, alpha_K)
    u, q = solve_time_effective_stiffness(M, C, K, DELTA_T, NUMBER_TIME_STEPS, dirichlet_nodes, dirichlet_values, GAMMA, BETA, number_of_nodes, electrode_triangles, nodes)

    create_vector_field_as_csv(u, nodes, "fields")

    np.save("charge", q)