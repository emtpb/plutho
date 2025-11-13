"""Module for the simulation of nonlinaer piezoelectric systems using the
harmonic balancing method."""

# Python standard libraries
from typing import Tuple

# Third party libraries
import numpy as np
import numpy.typing as npt
from scipy import sparse
import scipy.sparse.linalg as slin
import scipy

# Local libraries
from ...enums import SolverType
from .base import assemble, Nonlinearity
from ...mesh.mesh import Mesh
from ..solver import FEMSolver


__all__ = [
    "NLPiezoHB"
]


class NLPiezoHB(FEMSolver):
    """Implementes a nonlinear FEM harmonic balancing simulation.
    """
    # Nonlinear simulation
    nonlinearity: Nonlinearity

    # Harmonic balancing
    hb_order: int


    def __init__(
        self,
        simulation_name: str,
        mesh: Mesh,
        nonlinearity: Nonlinearity,
        hb_order: int
    ):
        super().__init__(simulation_name, mesh)

        self.solver_type = SolverType.PiezoHB

        self.nonlinearity = nonlinearity
        self.nonlinearity.set_mesh_data(self.mesh_data, self.node_points)
        self.hb_order = hb_order

    def assemble(self):
        """Redirect to general nonlinear assembly function"""
        self.material_manager.initialize_materials()

        m, c, k = assemble(
            self.mesh_data,
            self.material_manager
        )
        self.m = m
        self.c = c
        self.k = k

        self.nonlinearity.assemble(m, c, k)

    def simulate(
        self,
        frequencies: npt.NDArray,
        tolerance: float = 1e-11,
        max_iter: int = 300,
    ):
        dirichlet_nodes = np.array(self.dirichlet_nodes)
        dirichlet_values = np.array(self.dirichlet_values)
        number_of_nodes = len(self.mesh_data.nodes)

        m = self.m.copy()
        c = self.c.copy()
        k = self.k.copy()

        # Apply dirichlet bc to matrices
        m, c, k = self._apply_dirichlet_bc(
            m,
            c,
            k,
            dirichlet_nodes
        )
        self.nonlinearity.apply_dirichlet_bc(dirichlet_nodes)

        # Prepare array
        u = np.zeros(
            (len(frequencies), 3*self.hb_order*number_of_nodes),
            dtype=np.complex128
        )

        print("Starting harmonic balancing simulation")
        for frequency_index, frequency in enumerate(frequencies):
            f = self._get_load_vector(
                dirichlet_nodes,
                dirichlet_values[:, frequency_index]
            )

            # Get initial value
            if frequency_index > 0:
                u_i = u[frequency_index-1, :]
            else:
                u_i = u[frequency_index, :]

            # Check if initial value is already sufficient
            best_norm = scipy.linalg.norm(self.residual(
                m, c, k, u_i, f, frequency
            ))
            if best_norm < tolerance:
                u[frequency_index, :] = u_i
                continue
            best_u_i = u_i

            # Newton iteration
            converged = False
            for i in range(max_iter):
                # Calculate next guess for u using tangent matrix
                tangent_matrix = self.tangent_matrix(
                    u_i,
                    m,
                    c,
                    k,
                    frequency
                )
                delta_u = slin.spsolve(
                    tangent_matrix.tocsc(),
                    self.residual(m, c, k, u_i, f, frequency)
                )
                u_i_next = u_i - delta_u

                # Check for convergence
                norm = scipy.linalg.norm(self.residual(
                    m, c, k, u_i_next, f, frequency
                ))
                if norm < tolerance:
                    # Newton converged
                    u[frequency_index, :] = u_i_next
                    converged = True
                    print(f"Frequency step {frequency_index} converged after "
                        f"iteration {i+1}"
                    )
                    break
                elif norm < best_norm:
                    best_norm = norm
                    best_u_i = u_i_next

                if i % 100 == 0 and i > 0:
                    print("Iteration:", i)

            if not converged:
                print("Newton did not converge at frequency index "
                    f"{frequency_index}. Choosing best value: {best_norm}")
                u[frequency_index, :] = best_u_i

        self.u = u

    def residual(self, m, c, k, u, f, frequency):
        number_of_nodes = len(self.mesh_data.nodes)
        res = np.zeros(self.hb_order*3*number_of_nodes, dtype=np.complex128)
        angular_frequency = 2*np.pi*frequency

        for n in range(self.hb_order):
            u_n = u[n*3*number_of_nodes:(n+1)*3*number_of_nodes]
            f_n = f[n*3*number_of_nodes:(n+1)*3*number_of_nodes]
            res[n*3*number_of_nodes:(n+1)*3*number_of_nodes] = (
                - angular_frequency**2*(n+1)**2*m@u_n
                + 1j*angular_frequency*(n+1)*c@u_n
                + k@u_n
                - f_n
            )

        # Add nonlinearity
        if self.hb_order >= 3:
            u_3 = u[2*3*number_of_nodes:3*3*number_of_nodes]
            res[2*3*number_of_nodes:3*3*number_of_nodes] += \
                self.nonlinearity.evaluate_force_vector(u_3, m, c, k)

        return res

    def tangent_matrix(self, u, m, c, k, frequency):
        angular_frequency = 2*np.pi*frequency
        number_of_nodes = len(self.mesh_data.nodes)

        blocks =  []
        for n in range(self.hb_order):
            u_n = u[n*3*number_of_nodes:(n+1)*3*number_of_nodes]
            blocks.append(
                - angular_frequency**2*(n+1)**2*m
                + 1j*angular_frequency*(n+1)*c
                + k
                + self.nonlinearity.evaluate_jacobian(u_n, m, c, k)
            )

        # Add nonlinearity
        if self.hb_order >= 3:
            u_3 = u[2*3*number_of_nodes:3*3*number_of_nodes]
            blocks[2] += self.nonlinearity.evaluate_jacobian(u_3, m, c, k)

        return sparse.block_diag(blocks)

    def _get_load_vector(
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
        f = np.zeros(self.hb_order*3*number_of_nodes, dtype=np.float64)

        # Set dirichlet bc
        f[nodes] = values

        return f

    def _apply_dirichlet_bc(
        self,
        m: sparse.lil_array,
        c: sparse.lil_array,
        k: sparse.lil_array,
        nodes: npt.NDArray
    ) -> Tuple[
        sparse.csc_array,
        sparse.csc_array,
        sparse.csc_array
    ]:
        # Set rows of matrices to 0 and diagonal of K to 1 (at node points)
        m[nodes, :] = 0
        c[nodes, :] = 0
        k[nodes, :] = 0
        k[nodes, nodes] = 1

        return m.tocsc(), c.tocsc(), k.tocsc()
