"""Module for the simulation of nonlinear piezoeletric systems
for the case of a time-stationary simulation. (d_t=0)
"""

# Python standard libraries

# Third party libraries
import numpy as np
import numpy.typing as npt
import scipy
from scipy import sparse
from scipy.sparse import linalg

# Local libraries
from ..base import MeshData, integral_ku, integral_kuv, integral_kve, \
    create_local_element_data
from piezo_fem.materials import MaterialManager
from piezo_fem.simulation.nonlinear.piezo_time import integral_u_nonlinear


class NonlinearPiezoSimStationary:
    """Implements a nonlinear time stationary simulation. The nonlinearity
    is embed in the hooke law: T_i=C_ij*S_j+L_ijk*S_j*S_k.
    In order to solve this system a Newton-Raphson algorithm is implemented.
    For comparison a simulation can also be done using the scipy least squares
    function. Additionaly a the corresponding linear system (L=0) can be solved
    too.
    """
    # Boundary conditions
    dirichlet_nodes: npt.NDArray
    dirichlet_values: npt.NDArray

    def __init__(
        self,
        mesh_data: MeshData,
        material_manager: MaterialManager,
    ):
        self.mesh_data = mesh_data
        self.material_manager = material_manager
        self.dirichlet_nodes = None
        self.dirichlet_valuse = None

    def assemble(self, nonlinear_elasticity_matrix: npt.NDArray):
        """Assembles the FEM matrices based on the set mesh_data and
        material_data. Resulting FEM matrices are saved in global variables.

        Parameters:
            nonlinear_elasticity_matrix: 4x4 nonlinear material matrix.
        """
        # TODO Assembly takes to long rework this algorithm
        # Maybe the 2x2 matrix slicing is not very fast
        nodes = self.mesh_data.nodes
        elements = self.mesh_data.elements
        self.local_elements = create_local_element_data(nodes, elements)

        number_of_nodes = len(nodes)
        ku = sparse.lil_matrix(
            (2*number_of_nodes, 2*number_of_nodes),
            dtype=np.float64
        )
        kuv = sparse.lil_matrix(
            (2*number_of_nodes, number_of_nodes),
            dtype=np.float64
        )
        kve = sparse.lil_matrix(
            (number_of_nodes, number_of_nodes),
            dtype=np.float64
        )
        lu = sparse.lil_matrix(
            (2*number_of_nodes, 2*number_of_nodes),
            dtype=np.float64
        )

        for element_index, element in enumerate(self.mesh_data.elements):
            # Get local element nodes and matrices
            local_element = self.local_elements[element_index]
            node_points = local_element.node_points
            jacobian_inverted_t = local_element.jacobian_inverted_t
            jacobian_det = local_element.jacobian_det

            # TODO Check if its necessary to calculate all integrals
            # --> Dirichlet nodes could be leaved out?
            # Multiply with jac_det because its integrated with respect to
            # local coordinates.
            # Mutiply with 2*pi because theta is integrated from 0 to 2*pi
            ku_e = (
                integral_ku(
                    node_points,
                    jacobian_inverted_t,
                    self.material_manager.get_elasticity_matrix(element_index)
                )
                * jacobian_det * 2 * np.pi
            )
            kuv_e = (
                integral_kuv(
                    node_points,
                    jacobian_inverted_t,
                    self.material_manager.get_piezo_matrix(element_index)
                )
                * jacobian_det * 2 * np.pi
            )
            kve_e = (
                integral_kve(
                    node_points,
                    jacobian_inverted_t,
                    self.material_manager.get_permittivity_matrix(
                        element_index
                    )
                )
                * jacobian_det * 2 * np.pi
            )
            lu_e = (
                integral_u_nonlinear(
                    node_points,
                    jacobian_inverted_t,
                    nonlinear_elasticity_matrix
                ) * jacobian_det * 2 * np.pi
            )

            # Now assemble all element matrices
            for local_p, global_p in enumerate(element):
                for local_q, global_q in enumerate(element):
                    # Ku_e is a 6x6 matrix and 2x2 matrices are sliced out
                    # of it.
                    ku[2*global_p:2*global_p+2, 2*global_q:2*global_q+2] += \
                        ku_e[2*local_p:2*local_p+2, 2*local_q:2*local_q+2]

                    # KuV_e is a 6x3 matrix and 2x1 vectors are sliced out.
                    # [:, None] converts to a column vector.
                    kuv[2*global_p:2*global_p+2, global_q] += \
                        kuv_e[2*local_p:2*local_p+2, local_q][:, None]

                    # KVe_e is a 3x3 matrix and therefore the values can
                    # be set directly.
                    kve[global_p, global_q] += kve_e[local_p, local_q]

                    # L_e is similar to Ku_e
                    lu[2*global_p:2*global_p+2, 2*global_q:2*global_q+2] += \
                        lu_e[2*local_p:2*local_p+2, 2*local_q:2*local_q+2]

        # Calculate block matrices
        zeros1x1 = np.zeros((number_of_nodes, number_of_nodes))
        zeros2x1 = np.zeros((2*number_of_nodes, number_of_nodes))
        zeros1x2 = np.zeros((number_of_nodes, 2*number_of_nodes))

        k = sparse.block_array([
            [ku, kuv],
            [kuv.T, -1*kve]
        ])
        ln = sparse.block_array([
            [lu, zeros2x1],
            [zeros1x2, zeros1x1]
        ])

        self.k = k.tolil()
        self.ln = ln.tolil()

    def solve_linear(self):
        """Solves the linear problem Ku=f (ln is not used).
        """
        number_of_nodes = len(self.mesh_data.nodes)

        # Get FEM matrices
        k = self.k.copy()
        ln = self.ln.copy()  # Not used

        # Apply boundary conditions
        NonlinearPiezoSimStationary.apply_dirichlet_bc(
            k,
            ln,
            self.dirichlet_nodes
        )

        f = NonlinearPiezoSimStationary.get_load_vector(
            self.dirichlet_nodes,
            self.dirichlet_values,
            number_of_nodes
        )

        # Calculate u using phi as load vector
        u = linalg.spsolve(
            k.tocsc(),
            f
        )

        return u

    def solve_least_squares(
        self,
        initial_u: npt.NDArray = None,
        load_factor: float = 1
    ):
        """Runs a nonlinear simulation using the scipy least squares
        algorithm.

        Parameters:
            initial_u: Start value for least squares. If None is given the
                solution of the linear system is used a start value.
            load_factor: Multiplied with the load vector. Used to apply less
                than than the full excitation. Could improve conversion of the
                algorithm. Choose a value between 0 and 1.
        """
        number_of_nodes = len(self.mesh_data.nodes)

        k = self.k.copy()
        ln = self.ln.copy()

        # Apply boundary conditions
        k, ln = NonlinearPiezoSimStationary.apply_dirichlet_bc(
            k,
            ln,
            self.dirichlet_nodes
        )

        f = NonlinearPiezoSimStationary.get_load_vector(
            self.dirichlet_nodes,
            self.dirichlet_values,
            number_of_nodes
        )

        # If no start value is given use the linear solution instead
        if initial_u is None:
            # Calculate start value using linear simulation
            initial_u = linalg.spsolve(
                k.tocsc(),
                f
            )

        # Residual for least squares
        def residuum(u):
            r = k@u+ln@np.square(u)-load_factor*f
            return r

        # Jacobian matrix calculated analytically
        def jacobian(u):
            return (k+2*ln@sparse.diags_array(u)).tocsr()

        results = scipy.optimize.least_squares(
            residuum,
            initial_u,
            jac=jacobian,
            method="trf",
            tr_solver="lsmr",
            verbose=2,
            xtol=1e-30,
            ftol=1e-30,
        )

        if not results.success:
            return None

        print("Least squares optmization was successfull")
        print("The remaining residual is", results.cost)

        return results.x

    def solve_newton(
        self,
        tolerance: float,
        max_iter: int = 10000,
        u_start: npt.NDArray = None,
        alpha: float = 1,
        load_factor: float = 1
    ):
        """Solves the system of nonlinear equations using the Newton-Raphson
        method.

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
        ln = self.ln.copy()

        # Apply boundary conditions
        k, ln = NonlinearPiezoSimStationary.apply_dirichlet_bc(
            k,
            ln,
            self.dirichlet_nodes
        )

        f = NonlinearPiezoSimStationary.get_load_vector(
            self.dirichlet_nodes,
            self.dirichlet_values,
            number_of_nodes
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

        # Residual or the Newton algorithm
        def residuum(u):
            return k@u+ln@np.square(u)-load_factor*f

        # Add a best found field parameter which can be returned when the
        # maximum number of iterations were done
        best = current_u.copy()
        best_norm = scipy.linalg.norm(residuum(best))

        # Check if the initial value already suffices the tolerance condition
        if best_norm < tolerance:
            return best

        for iteration_count in range(max_iter):
            # Calculate tangent stiffness matrix
            k_tangent = NonlinearPiezoSimStationary. \
                calculate_tangent_matrix_hadamard(
                    current_u,
                    k,
                    ln
                )

            # Solve for displacement increment
            delta_u = linalg.spsolve(
                k_tangent,
                residuum(current_u)
            )

            # Update step
            next_u = current_u - alpha * delta_u

            # Check for convergence
            norm = np.linalg.norm(residuum(next_u))
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
        return best

    @staticmethod
    def calculate_tangent_matrix_hadamard(
        u: npt.NDArray,
        k: sparse.sparray,
        ln: sparse.sparray
    ):
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
    def apply_dirchlet_bc_loadvector(
        load_vector: npt.NDArray,
        nodes: npt.NDArray,
        values: npt.NDArray
    ):
        """Sets dirichlet boundary conditions on the given load vector

        Parameters:
            load_vector: Load vector for which the bc is set.
            nodes: List of node indices on which the bc is defined.
            values: List of values per node index which gives the value of
                the bc at the specific node.

        Returns:
            load vector with the boundary condition set.
        """
        for node, value in zip(nodes, values):
            load_vector[node] = value

        return load_vector

    @staticmethod
    def apply_dirichlet_bc(
        k,
        ln,
        nodes: npt.NDArray
    ):
        """Sets dirichlet boundary condition for the given matrix.

        Parameters:
            k: Stiffness matrix.
            l: Nonlinear stiffness matrix.
            nodes: List of node indices at which the bc are defined.

        Returns:
            Matrix with set boundary conditions.
        """
        for node in nodes:
            k[node, :] = 0
            ln[node, :] = 0

            k[node, node] = 1

        return k, ln

    @staticmethod
    def get_load_vector(
        nodes: npt.NDArray,
        values: npt.NDArray,
        number_of_nodes: int
    ):
        """Calculates the load vector based on the dirichlet nodes and values.

        Parameters:
            nodes: Dirichlet nodes.
            values: Dirichlet values (related to the nodes list).
            number_of_nodes: Total number of nodes of the FEM simulation.
        """
        f = np.zeros(3*number_of_nodes, dtype=np.float64)

        for node, value in zip(nodes, values):
            f[node] = value

        return f
