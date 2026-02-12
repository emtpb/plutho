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


def sim_hb(mesh, frequencies, amplitude, hb_order, zeta):
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

    # NOTE: To set the amplitude, the np.ones - array can just be multiplied
    # with a scalar
    # NOTE: Right now the excitation is only set for the cosine part
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

    # Simulation settings
    tolerance = 1e-7
    max_iter = 100
    newton_damping = 1

    # Run simulation
    sim.assemble()
    sim.simulate(
        frequencies,
        tolerance,
        max_iter,
        newton_damping
    )

    return sim


def plot_displacement(sim, base_frequency):
    dof = 3*len(sim.mesh_data.nodes)
    hb_order = sim.hb_order
    node_index = 128

    frequencies = [base_frequency*(i+1) for i in range(hb_order)]
    u = []
    for i in range(hb_order):
        u_cos = sim.u[0, 2*i*dof:(2*i+1)*dof]
        u_sin = sim.u[0, (2*i+1)*dof:(2*i+2)*dof]
        u.append(u_cos[2*node_index] + 1j*u_sin[2*node_index])

    print(np.max(np.abs(np.array(u))))
    plt.scatter(frequencies, np.abs(np.array(u)))
    plt.show()


def plot_impedance(sim, frequencies):
    sim.calculate_charge("Electrode")

    # NOTE: Amplitude must be set depending the excitation -> boundary condition
    excitation_amplitude = 1
    imp = excitation_amplitude/(1j*2*np.pi*frequencies*sim.q[0, :])

    plt.plot(frequencies, np.abs(imp))


if __name__ == "__main__":
    CWD = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "simulations"
    )

    if not os.path.isdir(CWD):
        os.makedirs(CWD)

    # Settings
    element_order = 1
    frequencies = np.linspace(98.7e3, 100e3, 100)

    # Load/create ring mesh
    mesh_file = os.path.join(CWD, "mesh.msh")
    plutho.Mesh.generate_rectangular_mesh(
        mesh_file,
        width=0.00635,
        height=0.001,
        x_offset=0.0026,
        mesh_size=0.0001,
        element_order=element_order
    )
    mesh = plutho.Mesh(mesh_file, element_order)

    # Run simulations
    sim_linear = sim_hb(mesh, frequencies, 1, 1, 0)
    sim_nonlinear_0_7v = sim_hb(mesh, frequencies, 0.7, 3, 1e7)
    sim_nonlinear_3_6v = sim_hb(mesh, frequencies, 3.6, 3, 1e7)
    sim_nonlinear_6_6v = sim_hb(mesh, frequencies, 6.6, 3, 1e7)

    sim_linear.calculate_charge("Electrode")
    sim_nonlinear_0_7v.calculate_charge("Electrode")
    sim_nonlinear_3_6v.calculate_charge("Electrode")
    sim_nonlinear_6_6v.calculate_charge("Electrode")

    impedance_linear = 1/(1j*2*np.pi*frequencies*sim_linear.q[0, :])
    impedance_nonlinear_0_7v = 0.7/(
        1j*2*np.pi*frequencies*sim_nonlinear_0_7v.q[0, :]
    )
    impedance_nonlinear_3_6v = 3.3/(
        1j*2*np.pi*frequencies*sim_nonlinear_3_6v.q[0, :]
    )
    impedance_nonlinear_6_6v = 6.6/(
        1j*2*np.pi*frequencies*sim_nonlinear_6_6v.q[0, :]
    )

    plt.plot(frequencies, np.abs(impedance_linear), label="Linear")
    plt.plot(
        frequencies,
        np.abs(impedance_nonlinear_0_7v),
        label="NL $0.7 \\mathrm{V}$"
    )
    plt.plot(
        frequencies,
        np.abs(impedance_nonlinear_3_6v),
        label="NL $3.3 \\mathrm{V}$"
    )
    plt.plot(
        frequencies,
        np.abs(impedance_nonlinear_6_6v),
        label="NL $6.6 \\mathrm{V}$"
    )

    plt.grid()
    plt.legend()
    plt.show()
