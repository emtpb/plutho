
# Standard libraries
import os

# Third party libraries
import numpy as np
import matplotlib.pyplot as plt

# Local libraries
import plutho


# Material data needed for the simulations
pic255 = plutho.MaterialData(
    **{
        "c11": 1.19e11,
        "c12": 0.84e11,
        "c13": 0.83e11,
        "c33": 1.17e11,
        "c44": 0.21e11,
        "e15": 12.09,
        "e31": -6.03,
        "e33": 15.49,
        "eps11": 8.15e-9,
        "eps33": 6.58e-9,
        "alpha_m": 0,
        "alpha_k": 6.259e-10,
        "density": 7800,
        "heat_capacity": 350,
        "thermal_conductivity": 1.1,
        "temperatures": 20
    }
)


def impedance_freq(mesh):
    frequencies = np.linspace(0, 1e7, 1000)[1:]

    sim = plutho.PiezoFreq(
        simulation_name="pic255_impedance_example",
        mesh=mesh
    )

    sim.add_material(
        material_name="pic255",
        material_data=pic255,
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

    sim.add_dirichlet_bc(
        field_type=plutho.FieldType.U_R,
        physical_group_name="Symaxis",
        values=np.zeros(len(frequencies))
    )

    sim.assemble()
    sim.simulate(frequencies)
    sim.calculate_charge("Electrode", is_complex=True)

    return frequencies, 1/(1j*2*np.pi*frequencies*sim.q)


def impedance_time(mesh):
    sim = plutho.PiezoTime(
        simulation_name="pic255_impedance_example_time",
        mesh=mesh
    )

    # Triangular excitation
    NUMBER_OF_TIME_STEPS = 8192
    DELTA_T = 1e-8
    excitation = np.zeros(NUMBER_OF_TIME_STEPS)
    excitation[1:10] = (
        1 * np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    )

    sim.add_material(
        material_name="pic255",
        material_data=pic255,
        physical_group_name=""
    )

    sim.add_dirichlet_bc(
        field_type=plutho.FieldType.PHI,
        physical_group_name="Electrode",
        values=np.ones(NUMBER_OF_TIME_STEPS)
    )

    sim.add_dirichlet_bc(
        field_type=plutho.FieldType.PHI,
        physical_group_name="Ground",
        values=np.zeros(NUMBER_OF_TIME_STEPS)
    )

    sim.add_dirichlet_bc(
        field_type=plutho.FieldType.U_R,
        physical_group_name="Symaxis",
        values=np.zeros(NUMBER_OF_TIME_STEPS)
    )

    sim.assemble()
    sim.simulate(
        DELTA_T,
        NUMBER_OF_TIME_STEPS,
        0.5,
        0.25
    )
    sim.calculate_charge("Electrode", is_complex=False)

    return plutho.calculate_impedance(sim.q, excitation, DELTA_T)


if __name__ == "__main__":
    CWD = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "simulations"
    )

    if not os.path.isdir(CWD):
        os.makedirs(CWD)

    # Load/create ring mesh
    mesh_file = os.path.join(CWD, "mesh.msh")
    plutho.Mesh.generate_rectangular_mesh(
        mesh_file,
        width=0.00635,
        height=0.001,
        x_offset=0.0026,
        mesh_size=0.0001,
        element_order=1
    )
    mesh = plutho.Mesh(mesh_file, 1)
    freq_time, imp_time = impedance_time(mesh)
    freq_freq, imp_freq = impedance_freq(mesh)

    plt.plot(freq_time, np.angle(imp_time), label="TD")
    plt.plot(freq_freq, np.angle(imp_freq), label="FD")
    plt.grid()
    plt.show()
