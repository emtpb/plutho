"""Implements an example script for the finite element method harmonic
balacing simulation.
"""

# Python standard libraries
import os

# Third party libraries
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

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


def simulate_hb(CWD, mesh, frequencies, amplitude, zeta, hb_order):
    nonlinearity = plutho.Nonlinearity()
    nonlinearity.set_cubic_rayleigh(zeta)

    # Create simulation
    sim = plutho.NLPiezoHB(
        "HB_Test",
        mesh,
        nonlinearity,
        hb_order
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
        amplitude*np.ones(len(frequencies))
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

    # Save results
    if not os.path.isdir(CWD):
        os.makedirs(CWD)

    u_file = os.path.join(CWD, "u.npy")
    q_file = os.path.join(CWD, "q.npy")
    np.save(u_file, sim.u)
    if sim.q is not None:
        np.save(q_file, sim.q)


def plot_impedances(charges, amplitudes, frequencies):
    for charge, amplitude in zip(charges, amplitudes):
        impedance = np.abs(amplitude/(1j*2*np.pi*frequencies*charge))
        plt.plot(frequencies/1e6, impedance, label=f"{amplitude} V")

    plt.xlabel("Frequency / kHz")
    plt.ylabel("Impedance / $\\Omega$")
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
    HB_ORDER = 3
    frequencies = np.linspace(90e3, 120e3, num=1000)
    zetas = [0, 1, 10, 50, 100, 500, 1000, 5000]
    amplitudes = [0.7, 3.6, 6.6]

    # Simulate
    if True:
        args = []
        for zeta in zetas:
            for amplitude in amplitudes:
                args.append((
                    f"hb_test_{amplitude}_{zeta}",
                    mesh,
                    frequencies,
                    amplitude,
                    zeta,
                    HB_ORDER
                ))

        with Pool() as p:
            p.starmap(simulate_hb, args)

    # Plot
    if False:
        zeta = 5000

        charges = []
        for amplitude in amplitudes:
            hb_wd = os.path.join(CWD, f"hb_test_{amplitude}_{zeta}")
            charge = np.load(os.path.join(hb_wd, "q.npy"))
            charges.append(charge)
        plot_impedances(charges, amplitudes, frequencies)
