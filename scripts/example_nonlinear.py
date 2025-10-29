"""Implements an example on how to run a staionary nonlinear simulation."""

# Python standard libraries
import os

# Third party libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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


# Example data for the nonlinear stiffness matrix
c_nonlin_6x6_nonsymmetric = np.array([
    [-2.13e+14, -1.10e+14, -1.10e+14, -1.56e+15, -2.57e+13, -2.57e+13],
    [-1.10e+14, -2.13e+14, -1.10e+14, -2.57e+13, -1.56e+15, -2.57e+13],
    [-1.10e+14, -1.10e+14, -2.13e+14, -2.57e+13, -2.57e+13, -1.56e+15],
    [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
    [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
    [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
])


def create_chirp(
    start_frequency,
    end_frequency,
    number_of_time_steps,
    delta_t
):
    time_values = np.arange(number_of_time_steps)*delta_t
    return signal.chirp(
        time_values,
        start_frequency,
        time_values[-1],
        end_frequency,
        method="linear"
    )


def create_sinusoidal_excitation(
    amplitude,
    delta_t,
    number_of_time_steps,
    frequency
):
    """Creates a sinusoidal excitation array.

    Parameters:
        amplitude: Amplitude of the sin function.
        delta_t: Time difference between each time step.
        number_of_time_steps: Number of time steps of the excitation. Typically
            this is the same as for the simulation
        frequency: Frequency of the sin function.
    """
    time_steps = np.arange(number_of_time_steps)*delta_t
    return amplitude*np.sin(2*np.pi*frequency*time_steps)


def simulate_nl_time(CWD, mesh, delta_t, number_of_time_steps):
    # Simulation parameters
    GAMMA = 0.5
    BETA = 0.25
    ZETA = 10
    AMPLITUDE = 10
    FREQUENCY = 2.07e6

    nonlinearity = plutho.Nonlinearity()
    nonlinearity.set_cubic_rayleigh(ZETA)

    # Create simulation
    sim = plutho.NLPiezoTime(
        nl_time_sim_name,
        mesh,
        nonlinearity
    )

    # Set materials
    sim.add_material(
        material_name="pic181",
        material_data=pic181,
        physical_group_name=""  # Means all elements
    )

    # Set boundary conditions
    excitation = create_sinusoidal_excitation(
        amplitude=AMPLITUDE,
        delta_t=delta_t,
        number_of_time_steps=number_of_time_steps,
        frequency=FREQUENCY
    )
    sim.add_dirichlet_bc(
        plutho.FieldType.PHI,
        "Electrode",
        excitation
    )
    sim.add_dirichlet_bc(
        plutho.FieldType.PHI,
        "Ground",
        np.zeros(number_of_time_steps)
    )

    # Run simulation
    sim.assemble()
    sim.simulate(
        delta_t,
        number_of_time_steps,
        GAMMA,
        BETA,
        tolerance=5e-9,
        max_iter=40
    )
    sim.calculate_charge("Electrode")

    # Save results
    if not os.path.isdir(CWD):
        os.makedirs(CWD)

    u_file = os.path.join(CWD, "u.npy")
    q_file = os.path.join(CWD, "q.npy")
    np.save(u_file, sim.u)
    np.save(q_file, sim.q)


def plot_displacement_spectrum(
    working_directory,
    delta_t,
    number_of_time_steps
):
    node_index = 129

    u = np.load(os.path.join(working_directory, "u.npy"))
    u_r = u[2*node_index, :]
    u_z = u[2*node_index+1, :]
    U_r_jw = np.fft.fft(u_r)

    frequencies = np.fft.fftfreq(number_of_time_steps, delta_t)

    plt.plot(frequencies, np.abs(U_r_jw))
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

    DELTA_T = 1e-9
    NUMBER_OF_TIME_STEPS = 20000

    nl_time_sim_name = "nonlinear_time_dep_sim_20k_1e-9"
    nl_time_wd = os.path.join(CWD, nl_time_sim_name)

    ## Simulate
    if True:
        # simulate_nonlinear_stationary(CWD)
        simulate_nl_time(nl_time_wd, mesh, DELTA_T, NUMBER_OF_TIME_STEPS)

    ## Plot
    if False:
        plot_displacement_spectrum(nl_time_wd, DELTA_T, NUMBER_OF_TIME_STEPS)
