from sympy import *

r, N_1, N_2, N_3, N_1_r, N_2_r, N_3_r, N_1_z, N_2_z, N_3_z = symbols("r N_1 N_2 N_3 N_1_r N_2_r N_3_r N_1_z N_2_z N_3_z")

B_opt = Matrix([
    [N_1_r, 0, N_2_r, 0, N_3_r, 0],
    [0, N_1_z, 0, N_2_z, 0, N_3_z],
    [N_1/r, 0, N_2/r, 0, N_3/r, 0],
    [N_1_z, N_1_r, N_2_z, N_2_r, N_3_z, N_3_r]
])

c11, c12, c13, c14, c21, c22, c23, c24, c31, c32, c33, c34, c41, c42, c43, c44 = symbols("c_11 c_12 c_13 c_14 c_21 c_22 c_23 c_24 c_31 c_32 c_33 c_34 c_41 c_42 c_43 c_44")
c = Matrix([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]
])

print(simplify(B_opt.T @ c @ B_opt))