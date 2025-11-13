"""Implements an example on how to run a staionary nonlinear simulation."""

# Python standard libraries
import os

# Third party libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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


def simulate_hb(CWD, mesh, frequencies):
    # Simulation parameters
    ZETA = 10
    AMPLITUDE = 1
    HB_ORDER = 3

    nonlinearity = plutho.Nonlinearity()
    nonlinearity.set_cubic_rayleigh(ZETA)

    # Create simulation
    sim = plutho.NLPiezoHB(
        sim_name,
        mesh,
        nonlinearity,
        HB_ORDER
    )

    # Set materials
    sim.add_material(
        material_name="pic181",
        material_data=pic181_small_1,
        physical_group_name=""  # Means all elements
    )

    sim.add_dirichlet_bc(
        plutho.FieldType.PHI,
        "Electrode",
        AMPLITUDE*np.ones(len(frequencies))
    )
    sim.add_dirichlet_bc(
        plutho.FieldType.PHI,
        "Ground",
        np.zeros(len(frequencies))
    )

    # Run simulation
    sim.assemble()
    sim.simulate(frequencies, tolerance=1e-5)
    sim.calculate_charge("Electrode", is_complex=True)

    # Save results
    if not os.path.isdir(CWD):
        os.makedirs(CWD)

    u_file = os.path.join(CWD, "u.npy")
    q_file = os.path.join(CWD, "q.npy")
    np.save(u_file, sim.u)
    np.save(q_file, sim.q)

    return sim.q


def simulate_piezo_freq(mesh, frequencies):
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

    sim.calculate_charge("Electrode", is_complex=True)

    return sim.q


def plot_impedances(charge_pincic, charge_plutho, frequencies):
    impedance_pincic = np.abs(1/(1j*2*np.pi*frequencies*charge_pincic))
    impedance_plutho = np.abs(1/(1j*2*np.pi*frequencies*charge_plutho))
    plt.plot(frequencies/1e6, impedance_pincic, label="Pincic")
    plt.plot(frequencies/1e6, impedance_plutho, "--", label="Plutho")
    plt.legend()
    plt.grid()
    plt.show()

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
    mesh = plutho.Mesh(mesh_file, element_order=1)

    sim_name = "hb_test"
    hb_wd = os.path.join(CWD, sim_name)

    frequencies = np.linspace(122e3, 128e3, num=500)

    ## Simulate
    if True:
        # simulate_nonlinear_stationary(CWD)
        q_hb = simulate_hb(hb_wd, mesh, frequencies)
        q_piezo = simulate_piezo_freq(mesh, frequencies)
        plot_impedances(q_hb, q_piezo, frequencies)
