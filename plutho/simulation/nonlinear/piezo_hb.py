"""Module for the simulation of nonlinaer piezoelectric systems using the
harmonic balancing method."""

# Python standard libraries
from typing import Union, Tuple, Callable

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
    lu: sparse.lil_array

    # Resulting fields
    u: npt.NDArray

    def __init__(
        self,
        mesh_data: MeshData,
        material_manager: MaterialManager,
        harmonic_order: int
    ):
        self.mesh_data = mesh_data
        self.material_manager = material_manager
        self.harmonic_order = harmonic_order
        self.dirichlet_nodes = np.array([])
        self.dirichlet_valuse = np.array([])

    def assemble(
        self,
        nonlinear_order: int,
        nonlinear_type: NonlinearType,
        **kwargs
    ):
        """Redirect to general nonlinear assembly function

        Parameters:
            nonlinear_order: Order of the nonlinearity.
            nonlinear_type: Type of the nonlinear material.
            **kwargs: Parameters for the nonlinear material.
        """
        # Get the default FEM matrices
        m, c, k, lu = assemble(
            self.mesh_data,
            self.material_manager,
            nonlinear_order,
            nonlinear_type,
            **kwargs
        )
        self.nonlinear_order = nonlinear_order

        self.m = m
        self.c = c
        self.k = k
        self.lu = lu

    def solve_linear(self):
        """Solves the linear problem Ku=f (ln is not used).
        """
        # Apply boundary conditions
        _, _, k = NonlinearPiezoSimHb.apply_dirichlet_bc(
            None,
            None,
            self.k.copy(),
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
        fem_m = self.m.copy()
        fem_c = self.c.copy()
        fem_k = self.k.copy()
        fem_lu = self.lu.copy()

        # Create the harmonic balancing matrices from the FEM matrices
        hb_m, hb_c, hb_k = self.get_harmonic_balancing_matrices(
            fem_m,
            fem_c,
            fem_k
        )

        # Apply boundary conditions
        hb_m, hb_c, hb_k = NonlinearPiezoSimHb.apply_dirichlet_bc(
            hb_m,
            hb_c,
            hb_k,
            self.dirichlet_nodes
        )

        f = self.get_load_vector(
            self.dirichlet_nodes,
            self.dirichlet_values
        )

        # Set start value
        if u_start is None:
            # If no start value is given calculate one using a linear
            # simulation
            current_u = linalg.spsolve(
                (
                    angular_frequency**2*hb_m.tocsc()
                    + angular_frequency**2*hb_c.tocsc()
                    + hb_k.tocsc()
                ),
                load_factor*f
            )
        else:
            current_u = u_start

        # Residual of the Newton algorithm
        def residual(u, nonlinear_force):
            return (
                angular_frequency**2*hb_m@u
                 + angular_frequency*hb_c@u
                 + hb_k@u
                 + nonlinear_force
                 - load_factor*f
            )

        # Nonlinear forces based on starting field guess
        nonlinear_force = self.calculate_nonlinear_force(
            current_u,
            fem_lu,
            number_of_nodes
        )

        print(nonlinear_force)

        # best_field tracks the field with the best norm
        # best norm tracks the value of the norm itself
        # Calculate the first norms based on the starting field guess
        best_field = current_u.copy()
        best_norm = np.linalg.norm(residual(best_field, nonlinear_force))

        # Check if the initial value already suffices the tolerance condition
        if best_norm < tolerance:
            print(f"Initial value is already best {best_norm}")
            self.u = best_field
            return

        # Run Newton method
        for iteration_count in range(max_iter):
            print(f"Iteration {iteration_count}")

            if iteration_count > 0:
                # Does not need to be updated at step 0
                nonlinear_force = self.calculate_nonlinear_force(
                    current_u,
                    fem_lu,
                    number_of_nodes
                )

                print(nonlinear_force)

            # Calculate tangential stiffness matrix
            k_tangent = NonlinearPiezoSimHb. \
                calculate_tangent_matrix_numerical(
                    current_u,
                    nonlinear_force,
                    residual
                )

            # Solve for displacement increment
            delta_u = linalg.spsolve(
                k_tangent,
                residual(current_u, nonlinear_force)
            )

            # Update step
            next_u = current_u - alpha * delta_u

            # Check for convergence
            norm = scipy.linalg.norm(residual(next_u, nonlinear_force))
            if norm < tolerance:
                print(
                    f"Newton found solution after {iteration_count} "
                    f"steps with residual {norm}"
                )
                self.u = next_u
                return
            elif norm < best_norm:
                best_norm = norm
                best_field = next_u.copy()

            if iteration_count % 100 == 0 and iteration_count > 0:
                print("Iteration:", iteration_count)

            # Update for next iteration
            current_u = next_u

        print("Simulation did not converge")
        print("Error from best iteration:", best_norm)

        self.u = best_field

    def calculate_nonlinear_force(
        self,
        u: npt.NDArray,
        fem_lu: sparse.lil_array,
        number_of_nodes: int
    ) -> npt.NDArray:
        """Calculates the forces due to the nonlinearity.

        Parameters:
            u: Current solution vector.
            fem_lu: FEM nonlinearity matrix.
            number_of_nodes: Number of nodes of the mesh.

        Returns:
            Vector of nonlinear forces.
        """
        # TODO Currently only for third order nonlinearity
        nonlinear_force = np.zeros(2*3*number_of_nodes)
        for i in range(3*number_of_nodes):
            for j in range(3*number_of_nodes):
                a = u[2*j]
                b = u[2*j+1]
                nonlinear_force[2*i:2*i+2] += fem_lu[i,j]*np.array([
                    3/4*(a**3+a*b**2),
                    3/4*(a**2*b+b**3)
                ])

        # Add dirichlet boundary conditions
        nonlinear_force[self.dirichlet_nodes] = 0

        return nonlinear_force

    def get_harmonic_balancing_matrices(
        self,
        fem_m: sparse.lil_array,
        fem_c: sparse.lil_array,
        fem_k: sparse.lil_array,
    ) -> Tuple[sparse.lil_array, sparse.lil_array, sparse.lil_array]:
        """Calculates the harmonic balancing matrices from the FEM matrices.
        The matrices are getting bigger by a factor of (2*harmonic_order)**2

        Parameters:
            m: FEM mass matrix.
            c: FEM damping matrix.
            k: FEM stiffness matrix.

        Returns:
            Tuple of HB mass matrix, HB damping matrix and HB stiffness
            matrix.
        """
        harmonic_order = self.harmonic_order

        # Set harmonic balance derivative matrices
        nabla_hb = np.zeros(shape=(2*harmonic_order, 2*harmonic_order))
        nabla_hb_const = np.zeros(shape=(2*harmonic_order, 2*harmonic_order))
        for i in range(harmonic_order):
            nabla_hb[2*i:2*i+2, 2*i:2*i+2] = np.array([
                [0, i],
                [-i, 0]
            ])
            nabla_hb_const[2*i:2*i+2, 2*i:2*i+2] = np.array([
                [1, 0],
                [0, 1]
            ])

        # Apply HB derivative matrices on FEM matrices
        fem_m = sparse.kron(fem_m, np.square(nabla_hb), format='lil')
        fem_c = sparse.kron(fem_c, nabla_hb, format='lil')
        fem_k = sparse.kron(fem_k, nabla_hb_const, format='lil')

        return fem_m, fem_c, fem_k

    def get_load_vector(
        self,
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
        f = np.zeros(
            3*number_of_nodes*2*self.harmonic_order,
            dtype=np.float64
        )

        for node, value in zip(nodes, values):
            f[node] = value

        return f

# -------- Static functions --------

    @staticmethod
    def calculate_tangent_matrix_numerical(
        u: npt.NDArray,
        nonlinear_force: npt.NDArray,
        residual: Callable
    ):
        """Calculates the tanget stiffness matrix numerically using a first
        order finite difference approximation.

        Parameters:
            u: Current solution vector.
            nonlinear_force: Vector of nonlinear forces.
            residual: Residual function.
        """
        h = 1e-14
        k_tangent = sparse.lil_matrix(
            (len(u), len(u)),
            dtype=np.float64
        )
        for i in range(len(u)):
            e_m = np.zeros(len(u))
            e_m[i] = h
            k_tangent[:, i] = (
                residual(u+e_m, nonlinear_force)
                 - residual(u, nonlinear_force)
            )/h

        return k_tangent.tocsc()

    @staticmethod
    def calculate_tangent_matrix_hadamard(
        u: npt.NDArray,
        k: sparse.sparray,
        lu: sparse.sparray
    ):
        # TODO This matrix is not valid anymore in harmonic balancing approach
        """Calculates the tangent matrix based on an analytically
        expression.

        Parameters:
            u: Current mechanical displacement.
            k: FEM stiffness matrix.
            ln: FEM nonlinear stiffness matrix.
        """
        k_tangent = k+3*lu@sparse.diags_array(np.square(u))

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
        # Set rows of matrices to 0 and diagonal of K to 1 (at node points)

        # Matrices for u_r component
        for node in nodes:
            # Set rows to 0
            if m is not None:
                m[node, :] = 0
            if c is not None:
                c[node, :] = 0
            if k is not None:
                k[node, :] = 0

                # Set diagonal values to 1
                k[node, node] = 1

        return m.tocsc(), c.tocsc(), k.tocsc()
