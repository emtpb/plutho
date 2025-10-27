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
from ...mesh.mesh import MeshData
from plutho.materials import MaterialManager


class NLPiezoHB:
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
        """Redirect to general nonlinear assembly function.

        Parameters:
            nonlinear_order: Order of the nonlinearity.
            nonlinear_type: Type of the nonlinear material.
            **kwargs: Parameters for the nonlinear material.
        """
        # Get the default FEM matrices
        m, c, k = assemble(
            self.mesh_data,
            self.material_manager,
        )
        self.nonlinear_order = nonlinear_order

        self.m = m
        self.c = c
        self.k = k

        raise NotImplementedError("Nonlinear assembly not implemented for HB")

    def solve_linear(self):
        """Solves the linear problem Ku=f (ln is not used).
        """
        # Apply boundary conditions
        _, _, k = NLPiezoHB.apply_dirichlet_bc(
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
            angular_frequency: Angualr frequency at which the simulation is
                done.
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
            fem_k,
            angular_frequency
        )

        # Apply boundary conditions
        hb_m, hb_c, hb_k = NLPiezoHB.apply_dirichlet_bc(
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
                    hb_m
                    + hb_c
                    + hb_k
                ),
                load_factor*f
            )
        else:
            current_u = u_start

        # Residual of the Newton algorithm
        def residual(u, nonlinear_force):
            return (
                hb_m@u
                 + hb_c@u
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

        # best_field tracks the field with the best norm
        # best norm tracks the value of the norm itself
        # Calculate the first norms based on the starting field guess
        best_field = current_u.copy()
        best_norm = np.linalg.norm(residual(best_field, nonlinear_force))

        # Check if the initial value already suffices the tolerance condition
        #if best_norm < tolerance:
        #    print(f"Initial value is already best {best_norm}")
        #    self.u = best_field
        #    return

        # Run Newton method
        for iteration_count in range(max_iter):
            if iteration_count > 0:
                # Does not need to be updated at step 0
                nonlinear_force = self.calculate_nonlinear_force(
                    current_u,
                    fem_lu,
                    number_of_nodes
                )

            # Calculate tangential stiffness matrix
            k_tangent = self.calculate_tangent_matrix_analytical(
                    current_u,
                    hb_m,
                    hb_c,
                    hb_k
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
                    f"Newton found solution after {iteration_count+1} "
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

        u_c = u[::2]
        u_s = u[1::2]

        lu_c = fem_lu.dot(
            3/4*u_c**3+3/4*np.multiply(u_c, u_s**2)
        )
        lu_s = fem_lu.dot(
            3/4*u_s**3+3/4*np.multiply(u_s, u_c**2)
        )

        nonlinear_force = np.zeros(2*3*number_of_nodes)
        nonlinear_force[::2] = lu_c
        nonlinear_force[1::2] = lu_s

        """
        nonlinear_force = np.zeros(2*3*number_of_nodes)
        for i in range(3*number_of_nodes):
            a = u[2*i]
            b = u[2*i+1]
            nonlinear_force[2*i:2*i+2] = fem_lu[i,i]*np.array([
                3/4*(a**3+a*b**2),
                3/4*(a**2*b+b**3)
            ])
        """

        # Add dirichlet boundary conditions
        nonlinear_force[self.dirichlet_nodes] = 0

        return nonlinear_force

    def get_harmonic_balancing_matrices(
        self,
        fem_m: sparse.lil_array,
        fem_c: sparse.lil_array,
        fem_k: sparse.lil_array,
        angular_frequency: float
    ) -> Tuple[sparse.lil_array, sparse.lil_array, sparse.lil_array]:
        """Calculates the harmonic balancing matrices from the FEM matrices.
        The matrices are getting bigger by a factor of (2*harmonic_order)**2

        Parameters:
            fem_m: FEM mass matrix.
            fem_c: FEM damping matrix.
            fem_k: FEM stiffness matrix.

        Returns:
            Tuple of HB mass matrix, HB damping matrix and HB stiffness
            matrix.
        """
        harmonic_order = self.harmonic_order

        # Create harmonic balance derivative matrices
        nabla_m = np.zeros(shape=(2*harmonic_order, 2*harmonic_order))
        nabla_c = np.zeros(shape=(2*harmonic_order, 2*harmonic_order))
        nabla_k = np.zeros(shape=(2*harmonic_order, 2*harmonic_order))
        for i in range(harmonic_order):
            k = i + 1
            nabla_m[2*i:2*i+2, 2*i:2*i+2] = np.array([
                [-(angular_frequency**2*k**2), 0],
                [0, -(angular_frequency**2*k**2)]
            ])
            nabla_c[2*i:2*i+2, 2*i:2*i+2] = np.array([
                [0, angular_frequency*k],
                [-(angular_frequency*k), 0]
            ])
            nabla_k[2*i:2*i+2, 2*i:2*i+2] = np.array([
                [1, 0],
                [0, 1]
            ])

        # Apply HB derivative matrices on FEM matrices
        fem_m = sparse.kron(fem_m, nabla_m, format='lil').tolil()
        fem_c = sparse.kron(fem_c, nabla_c, format='lil').tolil()
        fem_k = sparse.kron(fem_k, nabla_k, format='lil').tolil()

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

    def calculate_tangent_matrix_analytical(
        self,
        u: npt.NDArray,
        hb_m: sparse.lil_array,
        hb_c: sparse.lil_array,
        hb_k: sparse.lil_array
    ) -> sparse.csc_array:
        """Calculates the tangent matrix using a priori analytical
        calculations.

        Parameters:
            u: Solution vector containing mechanical and electrical field
                values.
            hb_m: Harmonic balancing mass matrix.
            hb_c: Harmonic balancing damping matrix.
            hb_k: Harmonic balancing stiffness matrix.

        Returns:
            The tangent stiffness matrix.
        """
        # TODO Only valid for harmonic_order = 1 and nonlinearity_order = 3
        fem_lu = self.lu
        lu_diag = fem_lu.diagonal()
        number_of_nodes = len(self.mesh_data.nodes)
        size = number_of_nodes*3*self.harmonic_order*2

        center_diag = np.zeros(size)
        for i in range(len(lu_diag)):
            diag_value_first = 3/4*lu_diag[i]*(3*u[2*i]**2+u[2*i+1]**2)
            diag_value_second = 3/4*lu_diag[i]*(u[2*i]**2+3*u[2*i+1]**2)

            center_diag[2*i] = diag_value_first
            center_diag[2*i+1] = diag_value_second

        left_diag = np.zeros(size-1)
        left_diag[::2] = np.multiply(u[1::2], lu_diag)

        right_diag = np.zeros(size-1)
        right_diag[::2] = np.multiply(u[:-1:2], lu_diag)

        return hb_m + hb_c + hb_k + sparse.diags(
            diagonals=[
                left_diag,
                center_diag,
                right_diag,
            ],
            offsets=[-1, 0, 1],
            format="csc"
        )


    # -------- Static functions --------

    @staticmethod
    def calculate_tangent_matrix_numerical(
        u: npt.NDArray,
        nonlinear_force: npt.NDArray,
        residual: Callable
    ) -> sparse.csc_array:
        """Calculates the tanget stiffness matrix numerically using a first
        order finite difference approximation.
        Important note: Calculation is very slow.

        Parameters:
            u: Current solution vector.
            nonlinear_force: Vector of nonlinear forces.
            residual: Residual function.

        Returns:
            Tangential stiffness matrix in sparse csc format.
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
    def apply_dirichlet_bc(
        m: sparse.lil_array,
        c: sparse.lil_array,
        k: sparse.lil_array,
        nodes: npt.NDArray
    ) -> Tuple[
        sparse.csc_array,
        sparse.csc_array,
        sparse.csc_array
    ]:
        """Applies dirichlet boundary conditions to the given matrix based on
        the given nodes. Essentially the rows of the m c and k matrices at the
        node indices are set to 0. The element of the k matrix at position
        (node_index, node_index) is set to 1.

        Parameters:
            m: Mass matrix.
            c: Damping matrix.
            k: Stiffness matrix.
            nodes: Nodes at which the dirichlet boundary conditions shall be
                applied.

        Returns:
            Modified mass, damping and stiffness matrix.
        """
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
