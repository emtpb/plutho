
# Python standard libraries
import os

# Third party libraries
import numpy as np
import matplotlib.pyplot as plt

# Local libraries
import piezo_fem as pfem


# Example data for the nonlinear stiffness matrix
c_nonlin_6x6_nonsymmetric = np.array([
    [-2.13e+14, -1.10e+14, -1.10e+14, -1.56e+15, -2.57e+13, -2.57e+13],
    [-1.10e+14, -2.13e+14, -1.10e+14, -2.57e+13, -1.56e+15, -2.57e+13],
    [-1.10e+14, -1.10e+14, -2.13e+14, -2.57e+13, -2.57e+13, -1.56e+15],
    [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
    [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
    [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
])


def excitation_sweep_linear_nonlinear(
    base_directory: str,
):
    """Sweeps over different voltage excitation and runs a nonlinear
    simulation using a set nonlinear stiffness matrix.

    Parameters:
        base_directory: Directory in which a simulation directory is created.
    """
    # Set mesh file path
    mesh_file = os.path.join(
        base_directory,
        "disc_mesh_0DOT0001.msh"
    )

    # Create and initialize simulation
    nonlin_sim = pfem.PiezoNonlinearStationary(
        "nonlinear_excitation_sweep",
        base_directory,
        mesh_file
    )
    nonlin_sim.set_material(
        "pic255",
        pfem.materials.pic255_alpha_m_nonzero
    )

    # Mesh data
    nodes, _ = nonlin_sim.mesh.get_mesh_nodes_and_elements()
    number_of_nodes = len(nodes)

    # Set excitation
    excitations = np.linspace(0, 1000, num=10)

    # Solution variables
    linear_sweep = []
    nonlinear_sweep = []
    linear_u = None
    nonlinear_u = None

    # Set to true if the simulations shall be run
    if True:
        # Linear sim
        nonlin_sim.run_simulation(
            excitations=excitations,
            is_linear=True,
            is_disc=False,
            nonlinear_matrix_factor=0,
            nonlinear_matrix=np.zeros((6, 6))
        )
        linear_u = nonlin_sim.u_combined

        # Nonlinear sim
        nonlin_sim.run_simulation(
            excitations=excitations,
            is_linear=False,
            is_disc=True,
            nonlinear_matrix_factor=1,
            nonlinear_matrix=c_nonlin_6x6_nonsymmetric
        )
        nonlinear_u = nonlin_sim.u_combined

        # Chooses the nodes with the maximum value
        for excitation in excitations:
            linear_sweep.append(np.max(
                linear_u[excitation][:2*number_of_nodes]
            ))
            nonlinear_sweep.append(np.max(
                nonlinear_u[excitation][:2*number_of_nodes]
            ))

    # Set to true if the results shall be plotted
    if True:
        plt.plot(excitations, linear_sweep, label="Linear")
        plt.plot(excitations, nonlinear_sweep, label="Nonlinear")
        plt.xlabel("Anregung $\\phi_{ex}$ / V")
        plt.ylabel("Auslenkng $u$ / m")
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

    excitation_sweep_linear_nonlinear(CWD)
