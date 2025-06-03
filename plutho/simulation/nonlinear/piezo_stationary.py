"""Module for the simulation of nonlinear piezoeletric systems
for the case of a time-stationary simulation. (d_t=0)
"""

# Python standard libraries
from typing import List, Union, Tuple

# Third party libraries
import numpy as np
import numpy.typing as npt
import scipy
from scipy import sparse
from scipy.sparse import linalg

# Local libraries
from ..base import MeshData, integral_ku, integral_kuv, integral_kve, \
    create_local_element_data, LocalElementData
from plutho.materials import MaterialManager
from plutho.simulation.nonlinear.piezo_time import integral_u_nonlinear


class NonlinearPiezoSimStationary:
    """Implements a nonlinear time stationary simulation. The nonlinearity
    is embed in the hooke law: T_i=C_ij*S_j+L_ijk*S_j*S_k.
    In order to solve this system a Newton-Raphson algorithm is implemented.
    For comparison a simulation can also be done using the scipy least squares
    function. Additionaly a the corresponding linear system (L=0) can be solved
    too.
    """
    # TODO Missing 1/2 term for nonlinear part in hooke law

    # Simulation parameters
    mesh_data: MeshData
    material_manager: MaterialManager

    # Boundary conditions
    dirichlet_nodes: npt.NDArray
    dirichlet_values: npt.NDArray

    # FEM matrices
    k: sparse.sparray
    ln: sparse.sparray

    # Internal simulation data
    local_elements: List[LocalElementData]

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

    def assemble(self, nonlinear_elasticity_matrix: npt.NDArray):
        """Assembles the FEM matrices based on the set mesh_data and
        material_data. Resulting FEM matrices are saved in global variables.

        Parameters:
            nonlinear_elasticity_matrix: 4x4 nonlinear material matrix
                (rotational symmetric).
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
        # Get FEM matrices
        k = self.k.copy()
        ln = self.ln.copy()  # Not used

        # Apply boundary conditions
        k, ln = NonlinearPiezoSimStationary.apply_dirichlet_bc(
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
            k.tocsc(),
            f
        )

    def solve_newton(
        self,
        tolerance: float = 1e-11,
        max_iter: int = 1000,
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
        # Get FEM matrices
        k = self.k.copy()
        ln = self.ln.copy()

        # Apply boundary conditions
        k, ln = NonlinearPiezoSimStationary.apply_dirichlet_bc(
            k,
            ln,
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
                k.tocsc(),
                f
            )
        else:
            current_u = u_start

        # Residual of the Newton algorithm
        def residual(u):
            return k@u+ln@np.square(u)-load_factor*f

        # Add a best found field parameter which can be returned when the
        # maximum number of iterations were done
        best = current_u.copy()
        best_norm = np.linalg.norm(residual(best))

        # Check if the initial value already suffices the tolerance condition
        if best_norm < tolerance:
            return best

        for iteration_count in range(max_iter):
            # Calculate tangential stiffness matrix
            k_tangent = NonlinearPiezoSimStationary. \
                calculate_tangent_matrix_hadamard(
                    current_u,
                    k,
                    ln
                )

            # Solve for displacement increment
            delta_u = linalg.spsolve(
                k_tangent,
                residual(current_u)
            )

            # Update step
            next_u = current_u - alpha * delta_u

            # Check for convergence
            norm = scipy.linalg.norm(residual(next_u))
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
        nodes: npt.NDArray,
        values: npt.NDArray
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
        f = np.zeros(3*number_of_nodes, dtype=np.float64)

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
        k: sparse.sparray,
        ln: sparse.sparray,
        nodes: npt.NDArray
    ) -> Tuple[sparse.sparray, sparse.sparray]:
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
