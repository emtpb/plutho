import numpy as np

import piezo_fem as pfem

pic255 = pfem.MaterialData(
    elasticity_matrix=np.array([
        [1.19e11, 0.83e11, 0, 0.84e11],
        [0.83e11, 1.17e11, 0, 0.83e11],
        [0, 0, 0.21e11, 0],
        [0.84e11, 0.83e11, 0, 1.19e11]
    ]),
    permittivity_matrix=np.diag([8.15e-9, 6.58e-9]),
    piezo_matrix=np.array([
        [0, 0, 12.09, 0],
        [-6.03, 15.49, 0, -6.03]
    ]),
    density=7800,
    thermal_conductivity=1.1,
    heat_capacity=350,
    alpha_k=6.259e-10,
    alpha_m=1.267e5
)
