# Standard libraries
import os

# Third party libraries
import numpy as np
import matplotlib.pyplot as plt

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


def sim_hb(mesh, frequency, amplitude, hb_order, zeta):
    nonlinearity = plutho.Nonlinearity()
    nonlinearity.set_cubic_rayleigh(zeta)

    sim = plutho.NLPiezoHB(
        "HB_Test",
        mesh,
        nonlinearity,
        hb_order
    )

    # Set materials
    sim.add_material(
        material_name="pic181",
        material_data=pic181,
        physical_group_name=""
    )

    sim.add_dirichlet_bc(
        plutho.FieldType.PHI,
        "Electrode",
        amplitude*np.ones(1)
    )
    sim.add_dirichlet_bc(
        plutho.FieldType.PHI,
        "Ground",
        np.zeros(1)
    )

    # Simulation settings
    tolerance = 1e-7
    max_iter = 100
    newton_damping = 1

    # Run simulation
    sim.assemble()
    sim.simulate(
        [frequency],
        tolerance,
        max_iter,
        newton_damping
    )

    return sim


def plot_displacements(sims, labels, base_frequency):
    for sim, label in zip(sims, labels):
        dof = 3*len(sim.mesh_data.nodes)
        hb_order = sim.hb_order
        node_index = 128

        frequencies = [base_frequency*(i+1) for i in range(hb_order)]
        u = []
        for i in range(hb_order):
            u_cos = sim.u[0, 2*i*dof:(2*i+1)*dof]
            u_sin = sim.u[0, (2*i+1)*dof:(2*i+2)*dof]
            u.append(u_cos[2*node_index] + 1j*u_sin[2*node_index])

        plt.scatter(frequencies, np.abs(np.array(u)), label=label)

    plt.grid()
    plt.legend()
    plt.show()


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

    # Run simulations
    frequency = 1e6
    nl_factors = np.logspace(14, 15, 10)
    sims = []
    labels = []
    for factor in nl_factors:
        print(factor)
        labels.append(f"{factor}")
        sims.append(sim_hb(mesh, frequency, 1, 3, factor))

    plot_displacements(sims, labels, frequency)
