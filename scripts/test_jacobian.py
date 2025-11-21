"""Module to test the analytical jacobians with ones calculated through
finite differences.
"""

import os
import plutho


# Make random?
pic181 = plutho.MaterialData(
    **{
      "c11": 141218192791.12772,
      "c12": 82292914124.0279,
      "c13": 80362329625.75212,
      "c33": 137188538228.6311,
      "c44": 29848846049.816402,
      "e15": 13.216444117013664,
      "e31": -4.979419636068149,
      "e33": 14.149818966822629,
      "eps11": 1.3327347064648263e-08,
      "eps33": 5.380490373139249e-09,
      "alpha_m": 0.0,
      "alpha_k": 1.289813815258054e-10,
      "thermal_conductivity": 1.1,
      "heat_capacity": 350,
      "temperatures": 25,
      "density": 7850
    }
)


if __name__ == "__main__":
    CWD = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "simulations"
    )

    if not os.path.isdir(CWD):
        os.makedirs(CWD)

    # Load/create ring mesh
    mesh_file = os.path.join(CWD, "ring_mesh.msh")
    if not os.path.exists(mesh_file):
        plutho.Mesh.generate_rectangular_mesh(
            mesh_file,
            width=0.00635,
            height=0.001,
            x_offset=0.0026,
            mesh_size=0.0001
        )

    ZETA = 1e12
    HB_ORDER = 3

    mesh = plutho.Mesh(mesh_file, element_order=1)

    # Create a base nonlinearitz and simulation class
    nonlinearity = plutho.Nonlinearity()
    nonlinearity.set_cubic_rayleigh(ZETA)

    sim = plutho.NLPiezoHB(
        "HB_Test",
        mesh,
        nonlinearity,
        HB_ORDER
    )

    # Set materials
    sim.add_material(
        material_name="pic181",
        material_data=pic181,
        physical_group_name=""
    )

    sim.assemble()

    for i in range(10):
        sim.test_jacobian()
