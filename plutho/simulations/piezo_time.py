"""Module for the simulation of time domain piezoelectric systems."""


# Third party libraries
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as slin

# Local libraries
from .helpers import mat_apply_dbcs
from .integrals import integral_m, integral_ku, integral_kuv, \
    integral_kve
from ..enums import SolverType
from .solver import FEMSolver
from ..mesh.mesh import Mesh


__all__ = [
    "PiezoTime"
]


class PiezoTime(FEMSolver):
    """Class for the simulation of time domain piezoelectric systems.

    Attributes:
        m: Sparse mass matrix.
        c: Sparse damping matrix.
        k: Sparse stiffness matrix.
    """
    # FEM matrices
    m: sparse.lil_array
    c: sparse.lil_array
    k: sparse.lil_array

    def __init__(self, simulation_name: str, mesh: Mesh):
        super().__init__(simulation_name, mesh)

        self.solver_type = SolverType.PiezoTime

    def assemble(self):
        """Assembles the FEM matrices based on the set mesh_data and
        material_data.
        The matrices are stored in self.m, self.c, self.k.
        """
        # TODO Assembly takes to long rework this algorithm
        # Maybe the 2x2 matrix slicing is not very fast
        self.material_manager.initialize_materials()
        nodes = self.mesh_data.nodes
        element_order = self.mesh_data.element_order

        number_of_nodes = len(nodes)
        mu = sparse.lil_matrix(
            (2*number_of_nodes, 2*number_of_nodes),
            dtype=np.float64
        )
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

        for element_index, element in enumerate(self.mesh_data.elements):
            node_points = self.node_points[element_index]

            # TODO Check if its necessary to calculate all integrals
            # --> Dirichlet nodes could be leaved out?
            # Multiply with jac_det because its integrated with respect to
            # local coordinates.
            # Mutiply with 2*pi because theta is integrated from 0 to 2*pi
            mu_e = (
                self.material_manager.get_density(element_index)
                * integral_m(node_points, element_order)
                * 2 * np.pi
            )
            ku_e = (
                integral_ku(
                    node_points,
                    self.material_manager.get_elasticity_matrix(element_index),
                    element_order
                )
                * 2 * np.pi
            )
            kuv_e = (
                integral_kuv(
                    node_points,
                    self.material_manager.get_piezo_matrix(element_index),
                    element_order
                )
                * 2 * np.pi
            )
            kve_e = (
                integral_kve(
                    node_points,
                    self.material_manager.get_permittivity_matrix(
                        element_index
                    ),
                    element_order
                )
                * 2 * np.pi
            )

            # Now assemble all element matrices
            for local_p, global_p in enumerate(element):
                for local_q, global_q in enumerate(element):
                    # Mu_e is returned as a 3x3 matrix (should be 6x6)
                    # because the values are the same.
                    # Only the diagonal elements have values.
                    mu[2*global_p, 2*global_q] += mu_e[local_p][local_q]
                    mu[2*global_p+1, 2*global_q+1] += mu_e[local_p][local_q]

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

        # Calculate damping matrix
        # Currently in the simulations only one material is used
        # TODO Add algorithm to calculate cu for every element on its own
        # in order to use multiple materials
        cu = (
            self.material_manager.get_alpha_m(0) * mu
            + self.material_manager.get_alpha_k(0) * ku
        )

        # Calculate block matrices
        zeros1x1 = np.zeros((number_of_nodes, number_of_nodes))
        zeros2x1 = np.zeros((2*number_of_nodes, number_of_nodes))
        zeros1x2 = np.zeros((number_of_nodes, 2*number_of_nodes))

        m = sparse.bmat([
            [mu, zeros2x1],
            [zeros1x2, zeros1x1],
        ])
        c = sparse.bmat([
            [cu, zeros2x1],
            [zeros1x2, zeros1x1],
        ])
        k = sparse.bmat([
            [ku, kuv],
            [kuv.T, -1*kve]
        ])

        self.m = m.tolil()
        self.c = c.tolil()
        self.k = k.tolil()

    def simulate(
        self,
        delta_t: float,
        number_of_time_steps: int,
        gamma: float,
        beta: float
    ):
        """Runs the time domain simulation. Dirichlet boundary conditions must
        be set beforehand.
        The resulting field is stored in self.u, which contains the fields u_r,
        u_z and phi.

        Parameters:
            delta_t: Distance between two time steps.
            number_of_time_steps: Total number of time steps for the
                simulation.
            gamma: Newmark integration parameter.
            beta: Newmark integration parameter.
        """
        m = self.m
        c = self.c
        k = self.k

        node_count = len(self.mesh_data.nodes)
        dirichlet_nodes = np.array(self.dirichlet_nodes)
        dirichlet_values = np.array(self.dirichlet_values)

        # Init arrays
        # Displacement u which is calculated
        u = np.zeros((number_of_time_steps, 3*node_count), dtype=np.float64)
        # Displacement u derived after t (du/dt)
        v = np.zeros((number_of_time_steps, 3*node_count), dtype=np.float64)
        # v derived after u (d^2u/dt^2)
        a = np.zeros((number_of_time_steps, 3*node_count), dtype=np.float64)

        m, c, k = mat_apply_dbcs(
            m,
            c,
            k,
            dirichlet_nodes
        )

        k_star = (k+gamma/(beta*delta_t)*c+1/(beta*delta_t**2)*m).tocsr()

        print("Starting simulation")
        for time_index in range(number_of_time_steps-1):
            # Calculate load vector and add dirichlet boundary conditions
            f = np.zeros(3*node_count)
            f[dirichlet_nodes] = dirichlet_values[:, time_index+1]

            # Perform Newmark method
            # Predictor step
            u_tilde = (
                u[time_index, :]
                + delta_t*v[time_index, :]
                + delta_t**2/2*(1-2*beta)*a[time_index, :]
            )
            v_tilde = (
                v[time_index, :]
                + (1-gamma)*delta_t*a[time_index, :]
            )

            # Solve for next time step
            u[time_index+1, :] = slin.spsolve(
                k_star, (
                    f
                    - c*v_tilde
                    + (1/(beta*delta_t**2)*m
                       + gamma/(beta*delta_t)*c)*u_tilde))
            # Perform corrector step
            a[time_index+1, :] = (u[time_index+1, :]-u_tilde)/(beta*delta_t**2)
            v[time_index+1, :] = v_tilde + gamma*delta_t*a[time_index+1, :]

            if (time_index + 1) % 100 == 0:
                print(f"Finished time step {time_index+1}")

        self.u = u
