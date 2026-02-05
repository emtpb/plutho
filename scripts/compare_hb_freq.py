
# Standard libraries
import os

# Third party libraries
import numpy as np

# Local libraries
import plutho


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


def sim_hb(mesh, frequencies):
    HB_ORDER = 1

    nonlinearity = plutho.Nonlinearity()
    nonlinearity.set_cubic_rayleigh(0)

    sim = plutho.NLPiezoHB(
        "HB_Test",
        mesh,
        nonlinearity,
        1
    )

    # Set materials
    sim.add_material(
        material_name="pic181",
        material_data=pic181,
        physical_group_name=""  # Means all elements
    )

    sim.add_dirichlet_bc(
        plutho.FieldType.PHI,
        "Electrode",
        np.ones(len(frequencies))
    )
    sim.add_dirichlet_bc(
        plutho.FieldType.PHI,
        "Ground",
        np.zeros(len(frequencies))
    )

    # Run simulation
    sim.assemble()
    u = np.zeros(shape=(len(frequencies), 6*len(sim.mesh_data.nodes)))
    for index, frequency in enumerate(frequencies):
        u[index, :] = sim.simulate_linear(frequency)

    print(u[1, :])


def sim_piezo_freq(mesh, frequencies):
    sim = plutho.PiezoFreq(
        simulation_name="pincic_test",
        mesh=mesh
    )

    sim.add_material(
        material_name="pic181",
            material_data=pic181,
        physical_group_name=""
    )

    sim.add_dirichlet_bc(
        field_type=plutho.FieldType.PHI,
        physical_group_name="Electrode",
        values=np.ones(len(frequencies))
    )
    sim.add_dirichlet_bc(
        field_type=plutho.FieldType.PHI,
        physical_group_name="Ground",
        values=np.zeros(len(frequencies))
    )

    sim.assemble()
    sim.simulate(
        frequencies,
        False
    )

    u = sim.u
    print(np.real(u[1, :]))
    print(np.imag(u[1, :]))


if __name__ == "__main__":
    CWD = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "simulations"
    )

    if not os.path.isdir(CWD):
        os.makedirs(CWD)

    # Settings
    element_order = 1
    frequencies = np.linspace(0, 1e7, 10)[1:]

    # Load/create ring mesh
    mesh_file = os.path.join(CWD, "mesh.msh")
    plutho.Mesh.generate_rectangular_mesh(
        mesh_file,
        width=0.00635,
        height=0.001,
        x_offset=0.0026,
        mesh_size=0.01,
        element_order=element_order
    )
    mesh = plutho.Mesh(mesh_file, element_order)

    sim_hb(mesh, frequencies)
    sim_piezo_freq(mesh, frequencies)

