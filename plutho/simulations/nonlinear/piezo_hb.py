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
        self.u_hb = []

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
            (len(frequencies), 2*3*self.hb_order*number_of_nodes)
        )

        # Linear tangent matrix can be created beforehand and scaled with
        # frequency later
        tan_k, tan_a1, tan_a2 = self.tangent_linear(m, c, k)

        print("Starting harmonic balancing simulation")
        for frequency_index, frequency in enumerate(frequencies):
            # Construct whole linear tangent matrix
            angular_frequency = 2*np.pi*frequency
            tangent_linear = (
                tan_k + angular_frequency * tan_a1 +
                angular_frequency**2 * tan_a2
            ).tocsc()

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
            residual = self.residual(tangent_linear, u_i, f, frequency)
            best_norm = np.linalg.norm(residual)
            if best_norm < tolerance:
                print("Initial value already sufficient")
                u[frequency_index, :] = u_i
                continue
            best_u_i = u_i

            # Newton iteration
            converged = False
            for i in range(max_iter):
                # Calculate next guess for u using tangent matrix
                # tangent_matrix = self.tangent_matrix_nl(
                #     u_i,
                #     m,
                #     c,
                #     k,
                #     frequency
                # ) + tangent_linear
                tangent_matrix = tangent_linear

                # delta_u = slin.spsolve(
                #     tangent_matrix,
                #     self.residual(tangent_linear, u_i, f, frequency)
                # )
                lu = slin.splu(tangent_matrix)
                delta_u = lu.solve(residual)
                u_i_next = u_i - delta_u

                #  Update residual
                residual = self.residual(
                    tangent_linear, u_i_next, f, frequency
                )

                # Check for convergence
                norm = np.linalg.norm(residual)
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

        for n in range(self.hb_order):
            idx = 2*n
            self.u_hb.append(
                u[:, idx*3*number_of_nodes:(idx+1)*3*number_of_nodes]
                + 1j*u[:, (idx+1)*3*number_of_nodes:(idx+2)*3*number_of_nodes]
            )

        self.u = self.u_hb[0]

    def residual(self, res_matrix, u, f):
        return res_matrix.dot(u)-f

    def tangent_linear(self, m, c, k):
        k_blocks = [
            [None for _ in range(2*self.hb_order)]
            for _ in range(2*self.hb_order)
        ]

        a1_blocks = [
            [None for _ in range(2*self.hb_order)]
            for _ in range(2*self.hb_order)
        ]

        a2_blocks = [
            [None for _ in range(2*self.hb_order)]
            for _ in range(2*self.hb_order)
        ]

        for n in range(self.hb_order):
            i = 2*n
            j = 2*n + 1
            np1 = n + 1

            # constant part (k)
            k_blocks[i][i] = k
            k_blocks[j][j] = k

            # ω part (c)
            a1_blocks[i][j] =  np1 * c
            a1_blocks[j][i] = -np1 * c

            # ω² part (m)
            a2_blocks[i][i] = -(np1**2) * m
            a2_blocks[j][j] = -(np1**2) * m

        # Convert to sparse once
        k  = sparse.block_array(k_blocks, format="csc")
        a1 = sparse.block_array(a1_blocks, format="csc")
        a2 = sparse.block_array(a2_blocks, format="csc")

        return k, a1, a2

    def tangent_nonlinear(self, u, m, c, k, frequency):
        number_of_nodes = len(self.mesh_data.nodes)
        angular_frequency = 2*np.pi*frequency

        blocks = [
            [None for _ in range(2*self.hb_order)]
            for _ in range(2*self.hb_order)
        ]
        # This can be done beforehand and not every iteration
        for n in range(self.hb_order):
            # cos eq derived after cos part
            blocks[2*n][2*n] = -(n+1)**2*angular_frequency**2*m+k
            # cos eq derived after sin part
            blocks[2*n][2*n+1] = (n+1)*angular_frequency*c
            # sin eq derived after cos part
            blocks[2*n+1][2*n] = -(n+1)*angular_frequency*c
            # sin eq derived after sin part
            blocks[2*n+1][2*n+1] = -(n+1)**2*angular_frequency**2*m+k

        # TODO Add nonlinarity

        return sparse.block_array(blocks)

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
        f = np.zeros(2*self.hb_order*3*number_of_nodes, dtype=np.float64)

        # TODO Right now only the base frequency cosine part is set
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
