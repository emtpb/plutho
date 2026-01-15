"""Module for the simulation of nonlinaer piezoelectric systems using the
harmonic balancing method."""

# Python standard libraries
from typing import Tuple, Callable

# Third party libraries
import numpy as np
import numpy.typing as npt
from scipy import sparse
import scipy.sparse.linalg as slin

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
        """Assembles the matrices based on the set material and mesh."""
        self.material_manager.initialize_materials()

        m, c, k = assemble(
            self.mesh_data,
            self.material_manager
        )
        self.m = m
        self.c = c
        self.k = k

        self.nonlinearity.assemble(k)

    def simulate_linear(
        self,
        frequency: float,
        dirichlet_nodes: npt.NDArray,
        amplitude: float
    ) -> npt.NDArray:
        """Runs a linear simulation for one frequency and returns the whole
        u vector.

        Parameters:
            frequency: Frequency at which the simulation is done.

        Returns:
            Harmonic balancing u vector.
        """
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

        # Prepare arrays
        f = self._get_load_vector(
            dirichlet_nodes,
            amplitude
        )

        # Calculate system matrix
        angular_frequency = 2*np.pi*frequency
        m, c, k = self.linear_system_matrix(m, c, k)
        s = m*angular_frequency**2+c*angular_frequency+k

        # Solve system
        lu = slin.splu(s)
        return lu.solve(f)

    @staticmethod
    def newton(
        u_init: npt.NDArray,
        residual_fun: Callable,
        tangent_fun: Callable,
        max_iter: int,
        tolerance: float,
        newton_damping: float
    ):
        # Set initial value
        u_i = u_init

        # Check if initial value is already sufficient
        residual = residual_fun(u_i)
        norm = np.linalg.norm(residual)
        if norm < tolerance:
            print("Initial value already sufficient")
            return u_i

        # Run the iteration
        converged = False
        for i in range(max_iter):
            # Calculate next guess for u using tangent matrix
            tangent_matrix = tangent_fun(u_i)

            # TODO Faster than slin.spsolve? -> Change in other solvers?
            lu = slin.splu(tangent_matrix)
            delta_u = lu.solve(residual)
            u_i_next = u_i - delta_u * newton_damping

            #  Update residual
            residual = residual_fun(u_i)

            # Check for convergence
            norm = np.linalg.norm(residual)
            if norm < tolerance:
                # Newton converged
                print(f"Newton converged after {i} iterations")
                return u_i_next

            # Update for next iteration
            u_i = u_i_next

        if not converged:
            print(f"Newton did not converge: Maximum iteration ({norm})")
            return u_i

    @staticmethod
    def newton_arclength(
        u_last: npt.NDArray,
        freq_last: float,
        freq_load_last: float,
        tangent_u_fun: Callable,
        tangent_vec_freq_load_fun: Callable,
        residual_fun: Callable,
        tolerance: float,
        max_iter: int
    ):
        # Set initial value
        u_0 = u_last

        # Predictor step
        lu = slin.splu(tangent_u_fun(u_0, freq_last, freq_load_last))
        P = -tangent_vec_freq_load_fun(u_last, freq_last, freq_load_last)
        delta_u_p_0 = lu.solve(P)
        u_p = u_last + delta_u_p_0

        # Compute frequency load increment
        arc_length = 1.1
        delta_freq_load_0 = arc_length/(np.linalg.norm(delta_u_p_0))
        ## Calculate direction
        current_stiffness_parameter = P@u_p/np.linalg.norm(u_p)
        if current_stiffness_parameter < 0:
            delta_freq_load_0 *= -1
        freq_load_0 = freq_load_last + delta_freq_load_0

        # Check if initial value is already sufficient
        residual = residual_fun(u_0, freq_last, freq_load_0)
        norm = np.linalg.norm(residual)
        if norm < tolerance:
            print("Initial value already sufficient")
            return u_0, freq_load_0

        # Set initial values for iteraion
        u_i = u_0
        freq_load_i = freq_load_0

        for i in range(max_iter):
            # RIKS arc-length constraint
            def riks(u, freq_load):
                return (
                    (u_0-u_last).T@(u-u_0)
                    + (freq_load_0-freq_load_last)*(freq_load-freq_load_0)
                )

            # RIKS arc-length constrained derived after load
            def riks_dload():
                return freq_load_0-freq_load_last

            # RIKS arc-length constrained derived after u
            def riks_du():
                return u_0-u_last

            # tmp: Test Tangent matrix
            k_tan = tangent_u_fun(u_i, freq_last, freq_load_i)
            P = tangent_vec_freq_load_fun(u_i, freq_last, freq_load_i)
            P_mat = P.reshape(-1, 1)
            du = riks_du()
            du_mat = sparse.csc_matrix(du.reshape(1, -1))
            dl_mat = sparse.csc_matrix([[riks_dload()]])
            blocks = [
                [k_tan, -P_mat],
                [du_mat, dl_mat]
            ]
            t = sparse.block_array(blocks, format="csc")
            x = np.zeros(u_0.shape[0]+1)
            x[:-1] = u_i
            x[-1] = freq_load_i

            def res_fun(u):
                x = np.zeros(u.shape[0])
                x[:-1] = residual_fun(u[:-1], freq_last, freq_load_i)
                x[-1] = riks(u[:-1], freq_load_i)
                return x
    
            print("Testing jacobian")
            NLPiezoHB.test_jacobian_static(
                t,
                x,
                res_fun
            )

            # Solve for increments
            P = -tangent_vec_freq_load_fun(u_i, freq_last, freq_load_i)
            u_p_next = lu.solve(P)
            u_g_next = lu.solve(-residual_fun(u_i, freq_last, freq_load_i))

            # Calculate increments
            denominator = riks_dload()+riks_du()@u_p_next
            print(denominator)
            if denominator < 1e-14:
                print("Singular system")
                break
            delta_freq_load_i = - (
                riks(u_i, freq_load_i)
                + riks_du()@u_g_next
            )/denominator
            delta_u_i = delta_freq_load_i*u_p_next+u_g_next

            # Increment
            freq_load_i_next = freq_load_i + delta_freq_load_i
            u_i_next = u_i + delta_u_i

            # Update residual
            residual = residual_fun(u_i_next, freq_last, freq_load_i_next)

            # Check for convergence
            norm = np.linalg.norm(residual)
            if norm < tolerance:
                # Newton converged
                print(f"Newton converged after {i+1} iterations")
                return u_i_next, freq_load_i_next
            elif np.abs(norm) > 1e10:
                print("Newton diverged")
                break

            # Update for next iteration
            u_i = u_i_next
            freq_load_i = freq_load_i_next
            lu = slin.splu(tangent_u_fun(u_i, freq_last, freq_load_i))

        print(f"Newton did not converged: Maximum iteration (norm: {norm})")
        return u_i, freq_load_i

    def simulate_path(
        self,
        frequency_start: float,
        frequency_steps: int,
        frequency_default_step: float,
        amplitude: float,
        tolerance: float = 1e-6,
        max_iter: int = 100,
    ):
        dirichlet_nodes = np.array(self.dirichlet_nodes)
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

        # Prepare array and guess initial value using linear simulation
        u = np.zeros(
            (frequency_steps, 2*3*self.hb_order*number_of_nodes)
        )
        u[0, :] = self.simulate_linear(
            frequency_start,
            dirichlet_nodes,
            amplitude
        )
        frequency_loads = np.zeros(frequency_steps)
        frequency_loads[0] = 1
        frequencies = np.zeros(frequency_steps+1)
        frequencies[0] = frequency_start

        # Linear tangent matrix can be created beforehand and scaled with
        # frequency later
        # Since its linear it is equal to the tangent matrix for the linear
        # part
        m_tan, c_tan, k_tan = self.linear_system_matrix(m, c, k)

        print("Starting harmonic balancing simulation")
        for index in range(frequency_steps):
            frequency = frequencies[index]
            print(f"Frequency index: {index}, Frequency: {frequency}")

            # Get load vector
            load_vector = self._get_load_vector(
                dirichlet_nodes,
                amplitude
            )

            # Define residual and tangent functions for newton
            def residual_fun(u, frequency, frequency_load):
                angular_frequency = 2*np.pi*frequency*frequency_load
                tangent_linear = (
                    k_tan
                    + angular_frequency * c_tan
                    + angular_frequency**2 * m_tan
                )

                return self.residual(
                    tangent_linear,
                    u,
                    load_vector,
                    frequency
                )

            def tangent_fun(u, frequency, frequency_load):
                angular_frequency = 2*np.pi*frequency*frequency_load
                return (
                    k_tan
                    + angular_frequency*c_tan
                    + angular_frequency**2*m_tan
                    + self.tangent_nonlinear(u, frequency)
                ).tocsc()

            def dresidual_df(u, frequency, frequency_load):
                # Residual derived after frequency load factor
                angular_frequency = 2*np.pi*frequency
                return (
                    2*frequency_load*angular_frequency**2*m_tan
                    + angular_frequency*c_tan
                )@u

            if index > 0:
                u_last = u[index-1, :]
            else:
                u_last = self.simulate_linear(
                    frequency,
                    dirichlet_nodes,
                    amplitude
                )

            # Solve newton arclen
            u_res, frequency_load = NLPiezoHB.newton_arclength(
                u_last,
                frequencies[index],
                frequency_loads[index],
                tangent_fun,
                dresidual_df,
                residual_fun,
                tolerance,
                max_iter
            )
            u[index, :] = u_res
            frequency_loads[index] = frequency_load
            frequencies[index+1] = frequency_load*frequency

        self.u = u
        self.frequencies = frequencies[:-1]


    def simulate(
        self,
        frequencies: npt.NDArray,
        tolerance: float = 1e-6,
        max_iter: int = 100,
        newton_damping: float = 1
    ):
        """Runs the nonlinear simulation at the given frequencies.

        Parameters:
            frequencies: Frequencies for which the simulation is done.
            tolerance: Tolerance for the residual norm.
            max_iter: Maximum number of iterations for each frequency.
            newton_damping: Multiplied with the newton iteration steps to
                reduces the step size.
        """
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

        # Prepare array and guess initial value using linear simulation
        u = np.zeros(
            (len(frequencies), 2*3*self.hb_order*number_of_nodes)
        )
        loads = np.zeros(len(frequencies))

        # Linear tangent matrix can be created beforehand and scaled with
        # frequency later
        # Since its linear it is equal to the tangent matrix for the linear
        # part
        m_tan, c_tan, k_tan = self.linear_system_matrix(m, c, k)

        print("Starting harmonic balancing simulation")
        for frequency_index, frequency in enumerate(frequencies):
            print(f"Frequency index: {frequency_index}")
            # Construct whole linear tangent matrix using the current frequency
            angular_frequency = 2*np.pi*frequency
            tangent_linear = (
                k_tan + angular_frequency * c_tan +
                angular_frequency**2 * m_tan
            ).tocsc()

            # Get load vector
            f = self._get_load_vector(
                dirichlet_nodes,
                dirichlet_values[:, frequency_index]
            )

            # Define residual and tangent functions for newton
            def residual_fun(u):
                return self.residual(
                    tangent_linear,
                    u,
                    f,
                    frequency
                )
            def rhs_arc_len_fun(u):
                return residual_fun(u) + f

            def tangent_fun(u):
                return tangent_linear + self.tangent_nonlinear(
                    u, frequency
                )

            # Get initial value
            if frequency_index > 0:
                u_init = u[frequency_index-1, :]
            else:
                u_init = self.simulate_linear(
                    frequencies[0],
                    dirichlet_nodes,
                    amplitude
                )

            # Solve newton normal
            """
            u[frequency_index, :] = NLPiezoHB.newton(
                u_init,
                residual_fun,
                tangent_fun,
                max_iter,
                tolerance,
                newton_damping
            )
            """
            # Solve newton arclen
            u_res, load_res = NLPiezoHB.newton_arclength(
                u_init,
                f,
                rhs_arc_len_fun,
                tangent_fun,
                max_iter,
                tolerance,
                u[frequency_index-1, :],
                loads[frequency_index-1]
            )
            u[frequency_index, :] = u_res
            loads[frequency_index] = load_res

        self.u = u

    def residual(
        self,
        linear_matrix: sparse.csc_array,
        u: npt.NDArray,
        f: npt.NDArray,
        frequency: float
    ) -> npt.NDArray:
        """Calculates the residual for the nonlinear simulation.

        Parameters:
            linear_matrix:"""
        return linear_matrix.dot(u)-f+self.residual_nonlinear(
            u, frequency
        )

    def linear_system_matrix(
        self,
        m_fem: sparse.csc_array,
        c_fem: sparse.csc_array,
        k_fem: sparse.csc_array
    ) -> Tuple[sparse.csc_array, sparse.csc_array, sparse.csc_array]:
        """Calculates the matrices for the linear system. Since they have to
        incoroporate the simulation frequency, a tuple of 3 matrices is
        returned. To obtain the whole system matrix it is necessary to combine
        them using the angular frequency:
        angular_frequency**2*m+angular_frequency*c+k

        Parameters:
            m: FEM mass matrix.
            c: FEM damping matrix.
            k: FEM stiffness matrix.

        Returns:
            A tuple of the HB mass matrix, HB damping matrix and HB stiffness
            matrix.
        """
        k_blocks = [
            [None for _ in range(2*self.hb_order)]
            for _ in range(2*self.hb_order)
        ]

        c_blocks = [
            [None for _ in range(2*self.hb_order)]
            for _ in range(2*self.hb_order)
        ]

        m_blocks = [
            [None for _ in range(2*self.hb_order)]
            for _ in range(2*self.hb_order)
        ]

        for n in range(self.hb_order):
            i = 2*n
            j = 2*n + 1
            np1 = n + 1

            # constant part (k)
            k_blocks[i][i] = k_fem
            k_blocks[j][j] = k_fem

            # ω part (c)
            c_blocks[i][j] =  np1 * c_fem
            c_blocks[j][i] = -np1 * c_fem

            # ω² part (m)
            m_blocks[i][i] = -(np1**2) * m_fem
            m_blocks[j][j] = -(np1**2) * m_fem

        # Convert to sparse once
        k_hb = sparse.block_array(k_blocks, format="csc")
        c_hb = sparse.block_array(c_blocks, format="csc")
        m_hb = sparse.block_array(m_blocks, format="csc")

        return m_hb, c_hb, k_hb

    def _time_to_freq(
        self,
        u_time: npt.NDArray,
        frequency: float
    ) -> npt.NDArray:
        """Converts the a vector with dimensions [dof, time] to the HB u
        vector.

        Parameters:
            u_time: Vector of u in time domain.
            frequency: Base frequency.

        Returns:
            The HB u vector in frequency domain.
        """
        N = 2*self.hb_order+1
        angular_frequency = 2*np.pi*frequency
        t = np.linspace(0, 1/frequency, num=N, endpoint=False)

        # Base functions
        harmonics = np.arange(1, self.hb_order+1)[:, np.newaxis]
        cos_basis = np.cos(harmonics * angular_frequency * t)
        sin_basis = np.sin(harmonics * angular_frequency * t)

        # Convert to freq domain
        u_cos = (2/N) * (u_time @ cos_basis.T)
        u_sin = (2/N) * (u_time @ sin_basis.T)

        # Reorder for simulation
        u_freq = np.empty(2*self.hb_order*u_time.shape[0])
        for n in range(self.hb_order):
            dof = u_time.shape[0]
            u_freq[2*n*dof:(2*n+1)*dof] = u_cos[:, n]
            u_freq[(2*n+1)*dof:(2*n+2)*dof] = u_sin[:, n]

        return u_freq

    def _freq_to_time(
        self,
        u_freq: npt.NDArray,
        frequency: float
    ) -> npt.NDArray:
        """Converts the HB u vector in frequency domain in a time domain
        vector.

        Parameters:
            u_freq: HB u vector in frequency domain.
            frequency: Base frequency.

        Returns:
            HB u vector in time domain.
        """
        number_of_nodes = len(self.mesh_data.nodes)
        dof = 3*number_of_nodes
        N = 2*self.hb_order+1
        angular_frequency = 2*np.pi*frequency
        t = np.linspace(0, 1/frequency, num=N, endpoint=False)

        # Base functions
        harmonics = np.arange(1, self.hb_order+1)[:, np.newaxis]
        cos_basis = np.cos(harmonics * angular_frequency * t)
        sin_basis = np.sin(harmonics * angular_frequency * t)

        u_time = np.zeros((dof, N))
        for n in range(self.hb_order):
            u_time += np.outer(
                u_freq[2*n*dof:(2*n+1)*dof],
                cos_basis[n, :]
            )
            u_time += np.outer(
                u_freq[(2*n+1)*dof:(2*n+2)*dof],
                sin_basis[n, :]
            )

        return u_time

    def residual_nonlinear(
        self,
        u: npt.NDArray,
        frequency: float
    ) -> npt.NDArray:
        """Calculates the nonlinear residual of the given HB u vector.

        Parameters:
            u: HB u vector.
            frequency: Base frequency.

        Returns:
            Nonlinear residual vector.
        """
        u_time = self._freq_to_time(u, frequency)

        # Apply nonlinearity in time domain
        f_nl_time = np.zeros(shape=u_time.shape)
        for i in range(u_time.shape[1]):
            f_nl_time[:, i] = self.nonlinearity.evaluate_force_vector(
                u_time[:, i]
            )

        return self._time_to_freq(f_nl_time, frequency)

    def tangent_nonlinear(
        self,
        u: npt.NDArray,
        frequency: float
    ) -> sparse.csc_array:
        """Calculates the tangent matrix for the nonlinear part.

        Parameters:
            u: Current HB u vector.
            frequency: Base frequency.

        Returns:
            Tangent stiffness matrix (jacobian) of the nonlinear part.
        """
        number_of_nodes = len(self.mesh_data.nodes)
        dof = 3*number_of_nodes
        angular_frequency = 2*np.pi*frequency
        N = 2*self.hb_order+1
        t = np.linspace(0, 1/frequency, num=N, endpoint=False)

        # Base functions
        harmonics = np.arange(1, self.hb_order+1)[:, np.newaxis]
        cos_basis = np.cos(harmonics * angular_frequency * t)
        sin_basis = np.sin(harmonics * angular_frequency * t)

        # Calculate idft
        u_time = np.zeros((dof, N))
        for n in range(self.hb_order):
            u_time += np.outer(u[2*n*dof:(2*n+1)*dof], cos_basis[n, :])
            u_time += np.outer(u[(2*n+1)*dof:(2*n+2)*dof], sin_basis[n, :])

        # Build jacobian from sub-blocks and calculate dft of df_du_time
        df_du_time = 3 * u_time ** 2
        blocks = []
        for i in range(self.hb_order):
            # Cos part
            row_blocks = []
            for j in range(self.hb_order):
                # Cos cos part
                integrand = df_du_time * cos_basis[i, :] * cos_basis[j, :]
                cc_diag = (2/N) * np.sum(integrand, axis=1)
                cc_diag_sparse = sparse.diags_array([cc_diag], offsets=[0])
                cc = self.nonlinearity.ln@cc_diag_sparse

                # Cos sin part
                integrand = df_du_time * cos_basis[i, :] * sin_basis[j, :]
                cs_diag = (2/N) * np.sum(integrand, axis=1)
                cs_diag_sparse = sparse.diags_array([cs_diag], offsets=[0])
                cs = self.nonlinearity.ln@cs_diag_sparse

                row_blocks.append(cc)
                row_blocks.append(cs)

            blocks.append(row_blocks)

            # Sin part
            row_blocks = []
            for j in range(self.hb_order):
                # Sin cos part
                integrand = df_du_time * sin_basis[i, :] * cos_basis[j, :]
                sc_diag = (2/N) * np.sum(integrand, axis=1)
                sc_diag_sparse = sparse.diags_array([sc_diag], offsets=[0])
                sc = self.nonlinearity.ln@sc_diag_sparse

                # Sin sin part
                integrand = df_du_time * sin_basis[i, :] * sin_basis[j, :]
                ss_diag = (2/N) * np.sum(integrand, axis=1)
                ss_diag_sparse = sparse.diags_array([ss_diag], offsets=[0])
                ss = self.nonlinearity.ln@ss_diag_sparse

                row_blocks.append(sc)
                row_blocks.append(ss)

            blocks.append(row_blocks)

        return sparse.bmat(blocks, format='csc')

    def _get_load_vector(
        self,
        nodes: npt.NDArray,
        amplitude: float
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

        # TODO: Right now only the base frequency cosine part is set
        # Set dirichlet bc
        f[nodes] = amplitude

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

    @staticmethod
    def test_jacobian_static(
        k_tan_analytic,
        u,
        residual_fun,
        epsilon: float = 1e-6
    ) -> Tuple[float, float]:
        """Tests the analytical jacobian matrix with a jacobian calculated from
        a finite difference approximation using the given u and epsilon.

        Parameters:
            u: Example u vector.

        Returns:
            Tuple of absolute and relative errors.
        """
        # TODO: Rework and make class independent
        # Create test data
        # Finite difference jacobian
        n = len(u)
        J_fd = np.zeros((n, n))

        for i in range(n):
            u_plus = u.copy()
            u_plus[i] += epsilon
            Fp = residual_fun(u_plus)

            u_minus = u.copy()
            u_minus[i] -= epsilon
            Fm = residual_fun(u_minus)

            J_fd[:, i] = (Fp - Fm) / (2*epsilon)

        # Vergleiche
        diff = np.abs(k_tan_analytic.toarray() - J_fd)
        max_diff = np.max(diff)
        rel_diff = max_diff / (np.max(np.abs(J_fd)) + 1e-10)

        print(f"Max absolute difference: {max_diff}")
        print(f"Max relative difference: {rel_diff}")

        return max_diff, rel_diff

    def test_jacobian(self, epsilon: float = 1e-6) -> Tuple[float, float]:
        """Tests the analytical jacobian matrix with a jacobian calculated from
        a finite difference approximation using the given u and epsilon.

        Parameters:
            u: Example u vector.

        Returns:
            Tuple of absolute and relative errors.
        """
        # TODO: Rework and make class independent
        # Create test data
        nodes, _ = self.mesh.get_mesh_nodes_and_elements()
        dof = 3*len(nodes)
        frequency = np.random.randn(1)[0] * 1e6 + 1
        angular_frequency = 2*np.pi*frequency
        u = np.random.randn(2*self.hb_order*dof) * 0.01
        f = np.random.randn(2*self.hb_order*dof) * 0.01
        m = sparse.random(dof, dof, density=0.1, format='csc')
        c = sparse.random(dof, dof, density=0.1, format='csc')
        k = sparse.random(dof, dof, density=0.1, format='csc')

        # Analytical jacobian
        hb_m, hb_c, hb_k = self.linear_system_matrix(m, c, k)
        hb_s = angular_frequency**2*hb_m+angular_frequency*hb_c+hb_k
        J_analytical = hb_s + self.tangent_nonlinear(u, frequency)

        # Finite difference jacobian
        n = len(u)
        J_fd = np.zeros((n, n))

        for i in range(n):
            u_plus = u.copy()
            u_plus[i] += epsilon
            Fp = self.residual(hb_s, u_plus, f, frequency)

            u_minus = u.copy()
            u_minus[i] -= epsilon
            Fm = self.residual(hb_s, u_minus, f, frequency)

            J_fd[:, i] = (Fp - Fm) / (2*epsilon)

        # Vergleiche
        diff = np.abs(J_analytical.toarray() - J_fd)
        max_diff = np.max(diff)
        rel_diff = max_diff / (np.max(np.abs(J_fd)) + 1e-10)

        print(f"Max absolute difference: {max_diff}")
        print(f"Max relative difference: {rel_diff}")

        return max_diff, rel_diff

    def calculate_charge(self, electrode_name: str, is_complex: bool = True):
        """Calculate the charge of hb method. Resulting charge is saved in
        self.q. The cos and sin parts of the HB displacement are combined to
        one complex value. The resultin charge has 2 dimensions, the first
        dimension is the harmonic order and the second dimension the
        frequencies.

        Parameters:
            electrode_name: Name of the electrode phyiscal group.
            is_complex:
        """
        nodes, _ = self.mesh.get_mesh_nodes_and_elements()
        dof = 3*len(nodes)
        hb_u = np.copy(self.u)
        q = np.zeros(shape=(self.hb_order, hb_u.shape[0]), dtype=np.complex128)
        for n in range(self.hb_order):
            self.u = hb_u[:, 2*n*dof:(2*n+1)*dof] \
                + 1j*hb_u[:, (2*n+1)*dof:(2*n+2)*dof]
            super().calculate_charge(electrode_name, is_complex=True)
            q[n, :] = self.q

        self.u = hb_u
        self.q = q
