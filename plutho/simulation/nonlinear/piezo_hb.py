"""Module for the simulation of nonlinaer piezoelectric systems using the
harmonic balancing method."""

# Python standard libraries
from typing import Union, Tuple

# Third party libraries
import numpy as np
import numpy.typing as npt
import scipy
from scipy import sparse
from scipy.sparse import linalg

# Local libraries
from .base import assemble, NonlinearType
from ..base import MeshData
from plutho.materials import MaterialManager

class NonlinearPiezoSimHb:
    """Implementes a nonlinear FEM harmonic balancing simulation.
    """
    # Simulation parameters
    mesh_data: MeshData
    material_manager: MaterialManager

    # Boundary conditions
    dirichlet_nodes: npt.NDArray
    dirichlet_values: npt.NDArray

    # FEM matrices
    k: sparse.lil_array
    c: sparse.lil_array
    m: sparse.lil_array
    l: sparse.lil_array

    # Resulting fields
    u: npt.NDArray

    def __init__(
        self,
        mesh_data: MeshData,
        material_manager: MaterialManager,
    ):
        self.mesh_data = mesh_data
        self.material_manager = material_manager
        self.dirichlet_nodes = np.array([])
        self.dirichlet_valuse = np.array([])

    def assemble(
        self,
        nonlinear_order: int,
        nonlinear_type: NonlinearType,
        harmonic_order: int,
        **kwargs
    ):
        """Redirect to general nonlinear assembly function

        Parameters:
            nonlinear_order: Order of the nonlinearity.
            nonlinear_type: Type of the nonlinear material.
            **kwargs: Parameters for the nonlinear material.
        """
        # Get the default FEM matrices
        m, c, k, l = assemble(
            self.mesh_data,
            self.material_manager,
            nonlinear_order,
            nonlinear_type,
            **kwargs
        )
        self.nonlinear_order = nonlinear_order
        self.harmonic_order = harmonic_order

        self.m = m
        self.c = c
        self.k = k
        self.l = l

    def solve_linear(self):
        """Solves the linear problem Ku=f (ln is not used).
        """
        # Get FEM matrices
        k = self.k.copy()
        ln = self.l.copy()  # Not used

        # Apply boundary conditions
        k, ln = NonlinearPiezoSimHb.apply_dirichlet_bc(
            k,
            ln,
            self.dirichlet_nodes
        )

        f = self.get_load_vector(
            self.dirichlet_nodes,
            self.dirichlet_values
        )

        # Calculate u using phi as load vector
        self.u = linalg.spsolve(
            k,
            f
        ).todense()

    def solve_newton(
        self,
        angular_frequency: float,
        tolerance: float = 1e-10,
        max_iter: int = 100,
        u_start: Union[npt.NDArray, None] = None,
        alpha: float = 1,
        load_factor: float = 1
    ):
        """Solves the system of nonlinear equations using the Newton-Raphson
        method. Saves the result in self.u

        Parameters:
            tolerance: Upper bound for the residuum. Iteration stops when the
                residuum is lower than tolerance.
            max_iter: Maximum number of iteration when the tolerance is not
                met. Then the best solution (lowest residuum) is returned
            u_start: Initial guess for the nonlinear problem. If it is None
                a linear solution is calculated and used as a guess for
                nonlinear u.
            alpha: Damping parameter for updating the guess for u. Can be used
                to improve convergence. Choose a value between 0 and 1.
            load_factor: Multiplied with the load vector. Used to apply less
                than than the full excitation. Could improve conversion of the
                algorithm. Choose a value between 0 and 1.
        """
        number_of_nodes = len(self.mesh_data.nodes)

        # Get FEM matrices
        k = self.k.copy()
        c = self.c.copy()
        m = self.m.copy()
        l = self.l.copy()

        # Create the harmonic balancing matrices from the FEM matrices
        harmonic_order = self.harmonic_order
        nabla_hb_k = np.zeros(shape=(2*harmonic_order, 2*harmonic_order))
        nabla_hb = np.zeros(shape=(2*harmonic_order, 2*harmonic_order))
        for _k in range(harmonic_order):
            nabla_hb_k[2*_k:2*_k+2, 2*_k:2*_k+2] = np.array([
                [0, _k],
                [-_k, 0]
            ])
            nabla_hb[2*_k:2*_k+2, 2*_k:2*_k+2] = np.array([
                [1, 0],
                [0, 1]
            ])

        print(k.shape)

        m = sparse.kron(m, np.square(nabla_hb_k), format='lil')
        c = sparse.kron(c, nabla_hb_k, format='lil')
        k = sparse.kron(k, nabla_hb, format='lil')

        print("Number of nodes:", number_of_nodes)
        print("M:", m.shape)
        print("C", c.shape)
        print("K:", k.shape)

        # Apply boundary conditions
        m, c, k = NonlinearPiezoSimHb.apply_dirichlet_bc(
            m,
            c,
            k,
            self.dirichlet_nodes
        )


        f = self.get_load_vector(
            harmonic_order,
            self.dirichlet_nodes,
            self.dirichlet_values
        )

        # Set start value
        if u_start is None:
            # If no start value is given calculate one using a linear
            # simulation
            current_u = linalg.spsolve(
                k.tocsc(),
                f
            )
        else:
            current_u = u_start

        # Residual of the Newton algorithm
        def residual(u, omega):
            l_hb = np.zeros(shape=(2*3*number_of_nodes, 2*3*number_of_nodes))
            l_hb[self.dirichlet_nodes] = 0
            for i in range(3*number_of_nodes):
                a = u[2*i]
                b = u[2*i+1]
                l_hb[2*i:2*i+2] = l[i]*np.array([
                    3/4*(a**3+a*b**2),
                    3/4*(a**2*b+b**3)
                ])
            return omega**2*m@u+omega*c@u+k@u+l_hb-load_factor*f

        # Add a best found field parameter which can be returned when the
        # maximum number of iterations were done
        best = current_u.copy()
        best_norm = np.linalg.norm(residual(best, angular_frequency))

        # Check if the initial value already suffices the tolerance condition
        if best_norm < tolerance:
            return best

        for iteration_count in range(max_iter):
            # Calculate tangential stiffness matrix
            k_tangent = NonlinearPiezoSimHb. \
                calculate_tangent_matrix_hadamard(
                    current_u,
                    k,
                    l
                )

            # Solve for displacement increment
            delta_u = linalg.spsolve(
                k_tangent,
                residual(current_u, angular_frequency)
            )

            # Update step
            next_u = current_u - alpha * delta_u

            # Check for convergence
            norm = scipy.linalg.norm(residual(next_u, angular_frequency))
            if norm < tolerance:
                print(
                    f"Solve Newton found solution after {iteration_count} "
                    f"steps with residual {norm}"
                )
                return next_u
            elif norm < best_norm:
                best_norm = norm
                best = next_u.copy()

            if iteration_count % 100 == 0 and iteration_count > 0:
                print("Iteration:", iteration_count)

            # Update for next iteration
            current_u = next_u

        print("Simulation did not converge")
        print("Error from best iteration:", best_norm)
        self.u = best

    def get_load_vector(
        self,
        harmonic_order: float,
        nodes: npt.NDArray,
        values: npt.NDArray,
    ) -> npt.NDArray:
        """Calculates the load vector (right hand side) vector for the
        simulation.

        Parameters:
            nodes: Nodes at which the dirichlet value shall be set.
            values: Dirichlet value which is set at the corresponding node.

        Returns:
            Right hand side vector for the simulation.
        """
        number_of_nodes = len(self.mesh_data.nodes)

        # Can be initialized to 0 because external load and volume
        # charge density is 0.
        f = np.zeros(3*number_of_nodes*2*harmonic_order, dtype=np.float64)

        for node, value in zip(nodes, values):
            f[node] = value

        return f

# -------- Static functions --------

    @staticmethod
    def calculate_tangent_matrix_hadamard(
        u: npt.NDArray,
        k: sparse.sparray,
        ln: sparse.sparray
    ):
        # TODO Duplicate function in piezo_time.py
        """Calculates the tangent matrix based on an analytically
        expression.

        Parameters:
            u: Current mechanical displacement.
            k: FEM stiffness matrix.
            ln: FEM nonlinear stiffness matrix.
        """
        k_tangent = k+2*ln@sparse.diags_array(u)

        return k_tangent

    @staticmethod
    def apply_dirichlet_bc(
        m: sparse.lil_array,
        c: sparse.lil_array,
        k: sparse.lil_array,
        nodes: npt.NDArray
    ) -> Tuple[
        sparse.csc_array,
        sparse.csc_array,
        sparse.csc_array,
        sparse.csc_array
    ]:
        # TODO Parameters are not really ndarrays
        # Set rows of matrices to 0 and diagonal of K to 1 (at node points)

        # Matrices for u_r component
        for node in nodes:
            # Set rows to 0
            m[node, :] = 0
            c[node, :] = 0
            k[node, :] = 0

            # Set diagonal values to 1
            k[node, node] = 1

        return m.tocsc(), c.tocsc(), k.tocsc()
