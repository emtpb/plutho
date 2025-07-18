"""Implements an example for quadratic nonlinearities in time domain"""

# Python standard libraries
import os

# Third party libraries
import numpy as np
import scipy
import matplotlib.pyplot as plt

# Local libraries
import plutho


c_nonlin_6x6x6 = np.array([
    [[-1.9362e+14, -1.6882e+13, -1.6882e+13,  0.0000e+00,  0.0000e+00,  0.0000e+00],
        [-1.6882e+13, -1.6882e+13,  3.1629e+15,
            0.0000e+00,  0.0000e+00,  0.0000e+00],
        [-1.6882e+13,  3.1629e+15, -1.6882e+13,
            0.0000e+00,  0.0000e+00,  0.0000e+00],
        [0.0000e+00,  0.0000e+00,  0.0000e+00, -
            1.5899e+15,  0.0000e+00,  0.0000e+00],
        [0.0000e+00,  0.0000e+00,  0.0000e+00,
            0.0000e+00, -4.4184e+13,  0.0000e+00],
        [0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -4.4184e+13]],

    [[-1.6882e+13, -1.6882e+13,  3.1629e+15,  0.0000e+00,  0.0000e+00,  0.0000e+00],
        [-1.6882e+13, -1.9362e+14, -1.6882e+13,
            0.0000e+00,  0.0000e+00,  0.0000e+00],
        [3.1629e+15, -1.6882e+13, -1.6882e+13,
            0.0000e+00,  0.0000e+00,  0.0000e+00],
        [0.0000e+00,  0.0000e+00,  0.0000e+00, -
            4.4184e+13,  0.0000e+00,  0.0000e+00],
        [0.0000e+00,  0.0000e+00,  0.0000e+00,
            0.0000e+00, -1.5899e+15,  0.0000e+00],
        [0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -4.4184e+13]],

    [[-1.6882e+13,  3.1629e+15, -1.6882e+13,  0.0000e+00,  0.0000e+00,  0.0000e+00],
        [3.1629e+15, -1.6882e+13, -1.6882e+13,
            0.0000e+00,  0.0000e+00,  0.0000e+00],
        [-1.6882e+13, -1.6882e+13, -1.9362e+14,
            0.0000e+00,  0.0000e+00,  0.0000e+00],
        [0.0000e+00,  0.0000e+00,  0.0000e+00, -
            4.4184e+13,  0.0000e+00,  0.0000e+00],
        [0.0000e+00,  0.0000e+00,  0.0000e+00,
            0.0000e+00, -4.4184e+13,  0.0000e+00],
        [0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -1.5899e+15]],

    [[0.0000e+00,  0.0000e+00,  0.0000e+00, -1.5899e+15,  0.0000e+00,  0.0000e+00],
        [0.0000e+00,  0.0000e+00,  0.0000e+00, -
            4.4184e+13,  0.0000e+00,  0.0000e+00],
        [0.0000e+00,  0.0000e+00,  0.0000e+00, -
            4.4184e+13,  0.0000e+00,  0.0000e+00],
        [-1.5899e+15, -4.4184e+13, -4.4184e+13,
            0.0000e+00,  0.0000e+00,  0.0000e+00],
        [0.0000e+00,  0.0000e+00,  0.0000e+00,
            0.0000e+00,  0.0000e+00,  7.7285e+14],
        [0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  7.7285e+14,  0.0000e+00]],

    [[0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -4.4184e+13,  0.0000e+00],
        [0.0000e+00,  0.0000e+00,  0.0000e+00,
            0.0000e+00, -1.5899e+15,  0.0000e+00],
        [0.0000e+00,  0.0000e+00,  0.0000e+00,
            0.0000e+00, -4.4184e+13,  0.0000e+00],
        [0.0000e+00,  0.0000e+00,  0.0000e+00,
            0.0000e+00,  0.0000e+00,  7.7285e+14],
        [-4.4184e+13, -1.5899e+15, -4.4184e+13,
            0.0000e+00,  0.0000e+00,  0.0000e+00],
        [0.0000e+00,  0.0000e+00,  0.0000e+00,  7.7285e+14,  0.0000e+00,  0.0000e+00]],

    [[0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -4.4184e+13],
        [0.0000e+00,  0.0000e+00,  0.0000e+00,
            0.0000e+00,  0.0000e+00, -4.4184e+13],
        [0.0000e+00,  0.0000e+00,  0.0000e+00,
            0.0000e+00,  0.0000e+00, -1.5899e+15],
        [0.0000e+00,  0.0000e+00,  0.0000e+00,
            0.0000e+00,  7.7285e+14,  0.0000e+00],
        [0.0000e+00,  0.0000e+00,  0.0000e+00,
            7.7285e+14,  0.0000e+00,  0.0000e+00],
        [-4.4184e+13, -4.4184e+13, -1.5899e+15,  0.0000e+00,  0.0000e+00,  0.0000e+00]]
])

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
    sin = amplitude*np.sin(2*np.pi*frequency*time_steps)
    window = scipy.signal.windows.tukey(
        number_of_time_steps
    )
    excitation = np.multiply(
        sin,
        window
    )
    return excitation


def simulate(CWD, simulation_name):
    mesh_file = os.path.join(CWD, "ring_mesh_third_order.msh")
    element_order = 3

    # Load/create ring mesh
    if not os.path.exists(mesh_file):
        plutho.Mesh.generate_rectangular_mesh(
            mesh_file,
            width=0.00635,
            height=0.001,
            x_offset=0.0026,
            mesh_size=0.0002,
            element_order=element_order
        )
    mesh = plutho.Mesh(mesh_file, element_order)

    # Create simulation
    sim = plutho.PiezoNonlinear(
        CWD,
        simulation_name,
        mesh
    )

    # Set materials
    sim.add_material(
        material_name="pic181",
        material_data=pic181,
        physical_group_name=""  # Means all elements
    )
    sim.set_nonlinearity_type(
        plutho.NonlinearType.Custom,
        nonlinear_matrix=c_nonlin_6x6x6
    )
    #sim.set_nonlinearity_type(
    #    plutho.NonlinearType.Rayleigh,
    #        zeta=1e3
    #)

    # Simulation parameters
    DELTA_T = 4e-8
    NUMBER_OF_TIME_STEPS = 10000

    # Set boundary conditions
    excitation = create_sinusoidal_excitation(
        amplitude=10,
        delta_t=DELTA_T,
        number_of_time_steps=NUMBER_OF_TIME_STEPS,
        frequency=0.12261e6
    )
    sim.add_dirichlet_bc(
        plutho.FieldType.PHI,
        "Electrode",
        excitation
    )
    sim.add_dirichlet_bc(
        plutho.FieldType.PHI,
        "Ground",
        np.zeros(NUMBER_OF_TIME_STEPS)
    )

    sim.setup_time_dependent_simulation(
        delta_t=DELTA_T,
        number_of_time_steps=NUMBER_OF_TIME_STEPS,
        gamma=0.5,
        beta=0.25
    )

    # Needed in order to get the charge
    electrode_elements = mesh.get_elements_by_physical_groups(
        ["Electrode"]
    )["Electrode"]
    electrode_normals = np.tile([0, 1], (len(electrode_elements), 1))

    # Run simulation
    sim.simulate(
        electrode_elements=electrode_elements,
        electrode_normals=electrode_normals,
        tolerance=1e-10
    )

    sim.save_simulation_settings()
    sim.save_simulation_results()


def plot_displacement_spectrum(simulation_folder):
    sim = plutho.PiezoNonlinear.load_simulation_settings(simulation_folder)
    sim.load_simulation_results()

    node_index = 129

    u_r = sim.solver.u[2*node_index, :]
    u_z = sim.solver.u[2*node_index+1, :]
    U_r_jw = np.fft.fft(u_r)

    number_of_time_steps = sim.solver.simulation_data.number_of_time_steps
    delta_t = sim.solver.simulation_data.delta_t
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

    sim_name = "nonlinear_time_quadratic_1v"
    # simulate(CWD, sim_name)
    plot_displacement_spectrum(os.path.join(CWD, sim_name))
