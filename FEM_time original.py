import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as slin
import time
import os

# MATERIAL PARAMETER
rho = 7800
alpha_M = 1.267e5
alpha_K = 6.259e-10
c = np.array([[1.19e11, 8.4e10, 8.3e10, 0],
              [8.4e10, 1.19e11, 8.3e10, 0],
              [8.3e10, 8.3e10, 1.17e11, 0],
              [0, 0, 0, 2.1e10]])
e = np.array([[0, 0, 0, 12.09],
              [-6.03, -6.03, 15.49, 0]])
eps = np.diag([8.15e-9, 6.58e-9])

# TIME INTEGRATION PARAMETERS
gamma = 0.5
beta = 0.25

# READING MESH
file = open('piezoMat_ring.msh', 'r')
mesh = file.readlines()
file.close()

idx_names = [i for i in range(np.size(mesh)) if mesh[i] == '$PhysicalNames\n'][0]
idx_nodes = [i for i in range(np.size(mesh)) if mesh[i] == '$Nodes\n'][0]
idx_elements = [i for i in range(np.size(mesh)) if mesh[i] == '$Elements\n'][0]

number_names = int(mesh[idx_names+1])
number_nodes = int(mesh[idx_nodes+1])
number_elements = int(mesh[idx_elements+1])

names = [None]*number_names
for name in range(idx_names+2, idx_names+2+number_names):
    idx = int(mesh[name].split()[1])-1
    names[idx] = mesh[name].split()[-1]

points = []
for node in range(idx_nodes+2, idx_nodes+2+number_nodes):
    floats = [float(x) for x in mesh[node].split()]
    points = points+[floats[1:3]]

boundaries = {new: [] for new in names}
elements = []
for node in range(idx_elements+2, idx_elements+2+number_elements):
    ints = [int(x) for x in mesh[node].split()]
    number_tags = ints[2]
    if ints[1] == 1:  # line elements
        entity_number = ints[3]
        boundaries[names[entity_number-1]] += [[i-1 for i in ints[3+number_tags::]]]
    elif ints[1] == 2:  # area elements
        elements = elements+[[i-1 for i in ints[-3::]]]
    elif ints[1] == 15:  # point elements
        entity_number = ints[3]
        boundaries[names[entity_number-1]] += [ints[-1]]

# sampling
delta_t = 2e-8
time_steps = 20
t = np.arange(0, time_steps)*delta_t
q = np.ones(np.size(t))
u = np.zeros((3*number_nodes, time_steps))
z = np.zeros((3*number_nodes, time_steps))
a = np.zeros((3*number_nodes, time_steps))

tic = time.time()


# shape functions
def shape(x):
    vec = np.array([1 - x[0] - x[1], x[0], x[1]])
    return vec[None, :]


def der_shape(x):
    return np.array([[-1, -1],
                     [1, 0],
                     [0, 1]]).T


def transform_coordinates(points, x):
    shape_val = shape(x)
    return np.squeeze(np.dot(points, shape_val.T))


# operators
def b_op_global(x, jac_inv, points):
    shape_val = shape(x)
    der = np.dot(jac_inv, der_shape(x))
    r = transform_coordinates(points, x)[0]
    val = np.array([[der[0, 0], der[0, 1], der[0, 2], 0, 0, 0],
                    [shape_val[0, 0]/r, shape_val[0, 1]/r, shape_val[0, 2]/r, 0, 0, 0],
                    [0, 0, 0, der[1, 0], der[1, 1], der[1, 2]],
                    [der[1, 0], der[1, 1], der[1, 2],
                    der[0, 0], der[0, 1], der[0, 2]]])
    return val


def gradient(x, jac_inv, points):
    der = np.dot(jac_inv, der_shape(x))
    return np.array([[der[0, 0], der[0, 1], der[0, 2]],
                    [der[1, 0], der[1, 1], der[1, 2]]])


# quadrature rules
def quadratic_quadrature(function):
    return (-9/32*function([1/3, 1/3]) + 25/96*function([1/5, 1/5]) +
            25/96*function([3/5, 1/5]) + 25/96*function([1/5, 3/5]))


def line_quadrature(function):
    return 0.5*(function([-1/2/np.sqrt(3)+1/2, 0]) + function([1/2/np.sqrt(3)+1/2, 0]))


# integrals
def integral_M(points):
    def func(x):
        return np.dot(shape(x).T, shape(x))*transform_coordinates(points, x)[0]
    integral = quadratic_quadrature(func)
    return integral


def integral_KGcG(jac_inv, points):
    def func(x):
        return (np.dot(np.dot(b_op_global(x, jac_inv, points).T, c),
                       b_op_global(x, jac_inv, points)) *
                transform_coordinates(points, x)[0])
    integral = quadratic_quadrature(func)
    return integral


def integral_KDeG(jac_inv, points):
    def func(x):
        return (np.dot(np.dot(gradient(x, jac_inv, points).T, e),
                       b_op_global(x, jac_inv, points)) *
                transform_coordinates(points, x)[0])
    integral = quadratic_quadrature(func)
    return integral


def integral_KDepsD(jac_inv, points):
    def func(x):
        return (np.dot(np.dot(gradient(x, jac_inv, points).T, eps), gradient(x, jac_inv, points)) *
                transform_coordinates(points, x)[0])
    integral = quadratic_quadrature(func)
    return integral


# matrix assembly
M = spa.lil_matrix((number_nodes, number_nodes))
K = spa.lil_matrix((number_nodes*3, number_nodes*3))

for element in elements:
    jacobian = np.array([[(points[element[1]][0] - points[element[0]][0]),
                          (points[element[2]][0] - points[element[0]][0])],
                        [(points[element[1]][1] - points[element[0]][1]),
                         (points[element[2]][1] - points[element[0]][1])]])
    det = np.linalg.det(jacobian)
    jac_inv = np.linalg.inv(jacobian).T
    point_elem = np.array([[points[element[0]][0], points[element[1]][0], points[element[2]][0]],
                           [points[element[0]][1], points[element[1]][1], points[element[2]][1]]])

    M_elem = integral_M(point_elem)*det*2*np.pi
    K_GcG = integral_KGcG(jac_inv, point_elem)*det*2*np.pi
    K_DeG = integral_KDeG(jac_inv, point_elem)*det*2*np.pi
    K_DepsD = integral_KDepsD(jac_inv, point_elem)*det*2*np.pi
    K_elem = np.block([[K_GcG, K_DeG.T],
                       [K_DeG, -1*K_DepsD]])
    for i in range(3):
        for j in range(3):
            M[element[i], element[j]] += M_elem[i, j]
            for shift_i in range(3):
                for shift_j in range(3):
                    K[element[i]+shift_i*number_nodes, element[j]+shift_j*number_nodes] +=\
                        K_elem[i+shift_i*3, j+shift_j*3]


# boundary conditions
def dirichlet_vector(idx):
    diri = np.zeros(2*number_nodes)
    diri[idx] = 1
    return diri


def Dirichlet(matrix, excitation, idx_list, value_list):
    for n, idx in enumerate(idx_list):
        matrix[idx, :] = 0
        matrix[idx, idx] = 1
        excitation[idx] = value_list[n]
    return matrix, excitation


# boundaries and loads
f_excite = 0*np.ones(number_nodes)
q_excite = np.zeros(8000)
excitation = np.hstack((f_excite, f_excite, q_excite))
Dirichlet_idx = np.hstack((np.unique(np.array(sum(boundaries['"axis"'], []))),

                           np.unique(np.array(sum(boundaries['"electrodes12"'], []))) + 2 * number_nodes,
                           np.unique(np.array(sum(boundaries['"electrode3"'], []))) + 2 * number_nodes))
integral_idx = boundaries['"electrodes12"']

# extending matrices
zeros_vec = np.zeros((number_nodes, number_nodes))
zeros_vec2 = np.zeros((2*number_nodes, number_nodes))
matrix_M = spa.bmat([[rho*M, zeros_vec, zeros_vec],
                     [zeros_vec, rho*M, zeros_vec],
                     [zeros_vec, zeros_vec, zeros_vec]], format='lil')

# get Dirichlet boundaries in matrices
#matrix_M, f_ex = Dirichlet(matrix_M, excitation, Dirichlet_idx, Dirichlet_value)
#K, f_ex = Dirichlet(K, excitation, Dirichlet_idx, Dirichlet_value)
K_mech = spa.bmat([[K[0:2*number_nodes, 0:2*number_nodes], zeros_vec2],
                   [zeros_vec2.T, zeros_vec]])

C = alpha_M*matrix_M + alpha_K*K_mech


matrix_lhs = (matrix_M + gamma*delta_t*C + beta*delta_t**2*K).tolil()
pot_excite = np.zeros(len(t))
n_beg = int(1e-7/delta_t+1)
pot_excite[n_beg:2*n_beg-1] = np.arange(1, n_beg)/(n_beg-1)
pot_excite[n_beg*2-1:3*n_beg-3] = np.arange(1, n_beg-1)[::-1]/(n_beg-1)

np.savetxt("original_lhs.txt", matrix_lhs.todense())
# sparsity conversions
K = K.tocsr()
C = C.tocsr()

for t_idx, t in enumerate(t[:-1]):
    print(t_idx)
    q_idx = 0
    Dirichlet_value = np.hstack((np.zeros(np.size(np.unique(np.array(sum(boundaries['"axis"'],
                                                                         []))))),
                                0,
                                pot_excite[t_idx]*np.ones(np.size(np.unique(np.array(sum(
                                        boundaries['"electrodes12"'], []))))),
                                0*np.ones(np.size(np.unique(np.array(sum(
                                        boundaries['"electrode3"'], [])))))))
    # predictor
    u_tilde = u[:, t_idx] + delta_t*z[:, t_idx] + (1-beta)*delta_t**2/2*a[:, t_idx]
    z_tilde = z[:, t_idx] + delta_t*(1-gamma)*a[:, t_idx]
    # solve
    matrix_dir, f_ex = Dirichlet(matrix_lhs, excitation, Dirichlet_idx, Dirichlet_value)
    matrix_dir = matrix_dir.tocsr()
    rhs = -1*K*u_tilde - C*z_tilde
    print(np.linalg.cond(matrix_dir.todense()))
    a[:, t_idx+1] = slin.spsolve(matrix_dir, rhs)
    u[:, t_idx+1] = u_tilde + beta*delta_t**2 * a[:, t_idx+1]
    z[:, t_idx+1] = z_tilde + gamma*delta_t*a[:, t_idx+1]

    for idx in (integral_idx):  # over all boundary elements in electrode
        element = [element for element in elements
                   if idx[0] in element
                   and idx[1] in element][0]
        spare = [e for e in element if e != idx[0] and e != idx[1]][0]
        order = [element.index(idx[0]), element.index(idx[1]), element.index(spare)]
        element = [element[i] for i in order]
        jacobian = np.array([[(points[element[1]][0] - points[element[0]][0]),
                              (points[element[2]][0] - points[element[0]][0])],
                            [(points[element[1]][1] - points[element[0]][1]),
                             (points[element[2]][1] - points[element[0]][1])]])
        det = np.linalg.det(jacobian)
        jac_inv = np.linalg.inv(jacobian).T
        point_elem = np.array([[points[element[0]][0], points[element[1]][0],
                                points[element[2]][0]],
                               [points[element[0]][1], points[element[1]][1],
                                points[element[2]][1]]])

        ur_K = [u[i, t_idx+1] for i in element]
        uz_K = [u[i+number_nodes, t_idx+1] for i in element]
        u_K = ur_K + uz_K
        phi_K = [u[i+2*number_nodes, t_idx+1] for i in element]
        normal = np.array([points[idx[1]][1]-points[idx[0]][1],
                           points[idx[0]][0]-points[idx[1]][0]])
        normalize = np.linalg.norm(normal)
        normal = normal/normalize

        def function_pot(x):
            return ((np.dot(np.dot(np.dot(eps, gradient(x, jac_inv, point_elem)), phi_K),
                    normal))*transform_coordinates(point_elem, x)[0])

        def function_u(x):
            return ((np.dot(np.dot(np.dot(e, b_op_global(x, jac_inv, point_elem)), u_K),
                    normal))*transform_coordinates(point_elem, x)[0])
        int_eps = line_quadrature(function_pot)*2*np.pi*normalize
        int_e = line_quadrature(function_u)*2*np.pi*normalize
        q_idx += np.sum(-1*int_eps + int_e)
    q[t_idx+1] = q_idx
print(time.time()-tic)

plt.plot(q)
plt.show()

# plt.figure(3)
# plt.semilogy(freq/1e6, imp)
#
#points_x = [i[0] for i in points]
#points_y = [i[1] for i in points]
#plt.figure()
#plt.scatter(points_x, points_y, c=abs(u[0*number_nodes:1*number_nodes]))