
# Python standard libraries
import os
import time

# Third party libraries
import numpy as np
import matplotlib.pyplot as plt

# Local libraries
import plutho


pic181_small_1 = plutho.MaterialData(
    **{
        "c11": 145235418881.3833,
        "c12": 92239167225.64952,
        "c13": 81099198743.00365,
        "c33": 129637003240.59453,
        "c44": 22909547556.381878,
        "e15": 11.117240926254714,
        "e31": -5.043538535682103,
        "e33": 13.828571335580783,
        "eps11": 1.7215323179892597e-09,
        "eps33": 5.5430630756095885e-09,
        "alpha_m": 11.382565204240533,
        "alpha_k": 4.789248360238781e-10,
        "thermal_conductivity": 0,
        "heat_capacity": 0,
        "density": 7850,
        "temperatures": []
    }
)


def simulate_hb(mesh, frequencies, hb_order):
    nonlinearity = plutho.Nonlinearity()
    nonlinearity.set_cubic_rayleigh(0)

    # Create simulation
    sim = plutho.NLPiezoHB(
        "HB_Test_Impedance",
        mesh,
        nonlinearity,
        hb_order
    )

    # Set materials
    sim.add_material(
        material_name="pic181",
        material_data=pic181_small_1,
        physical_group_name=""
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
    sim.simulate(frequencies)
    sim.calculate_charge("Electrode", is_complex=True)

    return sim.q


def simulate_freq(mesh, frequencies):
    sim = plutho.PiezoFreq(
        simulation_name="pincic_test",
        mesh=mesh
    )

    sim.add_material(
        material_name="pic181",
        material_data=pic181_small_1,
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

    sim.calculate_charge("Electrode", is_complex=True)

    return sim.q


def plot_impedances(charge_freq, charge_hb, frequencies):
    impedance_freq = np.abs(1/(1j*2*np.pi*frequencies*charge_freq))
    impedance_hb = np.abs(1/(1j*2*np.pi*frequencies*charge_hb[0, :]))
    plt.plot(frequencies/1e6, impedance_freq, label="plutho freq")
    plt.plot(frequencies/1e6, impedance_hb, "--", label="plutho hb")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    CWD = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..",
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
    mesh = plutho.Mesh(mesh_file, element_order=1)

    # Simulation parameters
    frequencies = np.linspace(0, 1e7, 1000)[1:]
    HB_ORDER = 3

    # Simulate impedance
    start_time = time.time()
    charge_hb = simulate_hb(mesh, frequencies, HB_ORDER)
    plutho_hb_time = time.time() - start_time

    start_time = time.time()
    charge_freq = simulate_freq(mesh, frequencies)
    plutho_freq_time = time.time() - start_time

    print(f"plutho hb took {plutho_hb_time}s to simulate.")
    print(f"plutho freq pincic took {plutho_freq_time}s to simulate.")

    plot_impedances(charge_freq, charge_hb, frequencies)
