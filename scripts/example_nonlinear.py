"""Implements an example on how to run a staionary nonlinear simulation."""

# Python standard libraries
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


# Example data for the nonlinear stiffness matrix
c_nonlin_6x6_nonsymmetric = np.array([
    [-2.13e+14, -1.10e+14, -1.10e+14, -1.56e+15, -2.57e+13, -2.57e+13],
    [-1.10e+14, -2.13e+14, -1.10e+14, -2.57e+13, -1.56e+15, -2.57e+13],
    [-1.10e+14, -1.10e+14, -2.13e+14, -2.57e+13, -2.57e+13, -1.56e+15],
    [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
    [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
    [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
])


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


def simulate_nonlinear_stationary(CWD):
    sim_name = "nonlinear_stationary_sim"
    mesh_file = os.path.join(CWD, "ring_mesh.msh")

    # Load/create ring mesh
    if not os.path.exists(mesh_file):
        plutho.Mesh.generate_rectangular_mesh(
            mesh_file,
            width=0.00635,
            height=0.001,
            x_offset=0.0026,
            mesh_size=0.0001
        )
    mesh = plutho.Mesh(mesh_file)

    # Create simulation
    sim = plutho.PiezoNonlinear(
        sim_name,
        CWD,
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
        nonlinear_matrix=c_nonlin_6x6_nonsymmetric
    )

    # Set boundary conditions
    excitation = 1000
    sim.add_dirichlet_bc(
        plutho.FieldType.PHI,
        "Electrode",
        np.array(excitation)
    )
    sim.add_dirichlet_bc(
        plutho.FieldType.PHI,
        "Ground",
        np.array(0)
    )

    sim.setup_stationary_simulation()

    # For better convergence set u_r and u_z for one node to 0
    sim.dirichlet_nodes.append(0)
    sim.dirichlet_nodes.append(1)
    sim.dirichlet_values.append(0)
    sim.dirichlet_values.append(0)

    sim.simulate(
        tolerance=1e-10
    )


def simulate_nonlinaer_time_dep(CWD):
    sim_name = "nonlinear_time_dep_sim"
    mesh_file = os.path.join(CWD, "ring_mesh.msh")

    # Load/create ring mesh
    if not os.path.exists(mesh_file):
        plutho.Mesh.generate_rectangular_mesh(
            mesh_file,
            width=0.00635,
            height=0.001,
            x_offset=0.0026,
            mesh_size=0.0001
        )
    mesh = plutho.Mesh(mesh_file)

    # Create simulation
    sim = plutho.PiezoNonlinear(
        sim_name,
        CWD,
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
        nonlinear_matrix=c_nonlin_6x6_nonsymmetric
    )

    # Simulation parameters
    DELTA_T = 1e-8
    NUMBER_OF_TIME_STEPS = 1000

    # Set boundary conditions
    excitation = create_sinusoidal_excitation(
        amplitude=10,
        delta_t=DELTA_T,
        number_of_time_steps=NUMBER_OF_TIME_STEPS,
        frequency=2.07e6
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
        electrode_normals=electrode_normals
    )


if __name__ == "__main__":
    CWD = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "simulations"
    )

    if not os.path.isdir(CWD):
        os.makedirs(CWD)

    simulate_nonlinear_stationary(CWD)
    simulate_nonlinaer_time_dep(CWD)
