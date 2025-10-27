"""Module for the simulation of thermo-piezoelectric systems in the time
domain."""

# Python standard libraries
from typing import Union

# Third party libraries
import numpy as np
import numpy.typing as npt
import scipy.sparse.linalg as slin
from scipy import sparse

# Local libraries
from .solver import FEMSolver
from ..enums import SolverType
from ..mesh import Mesh
from .helpers import create_node_points, mat_apply_dbcs, calculate_volumes, \
    get_avg_temp_field_per_element
from .integrals import integral_m, integral_ku, integral_kuv, \
    integral_kve, integral_ktheta, integral_theta_load, integral_loss_scs_time


__all__ = [
    "ThermoPiezoTime"
]


class ThermoPiezoTime(FEMSolver):
    """Class for solving time domain themo-piezoelectric systems.

    Attributes:
        node_points: List of node points per elements.
        m: Sparse mass matrix.
        c: Sparse damping matrix.
        k: Sparse stiffness matrix.
        mech_loss: Mechanical loss field.
    """
    # Internal simulation data
    node_points: npt.NDArray

    # FEM matrices
    m: sparse.lil_array
    c: sparse.lil_array
    k: sparse.lil_array

    # Resulting fields
    mech_loss: npt.NDArray

    def __init__(self, simulation_name: str, mesh: Mesh):
        super().__init__(simulation_name, mesh)

        self.solver_type = SolverType.ThermoPiezoTime

    def assemble(self):
        """Assembles the FEM matrices based on the set mesh_data and
        material_data.
        The matrices are stored in self.m, self.c, self.k.
        """
        # TODO Assembly takes long rework this algorithm?
        # Maybe the 2x2 matrix slicing is not very fast
        self.material_manager.initialize_materials()
        nodes = self.mesh_data.nodes
        elements = self.mesh_data.elements
        element_order = self.mesh_data.element_order
        self.node_points = create_node_points(nodes, elements, element_order)

        number_of_nodes = len(nodes)
        mu = sparse.lil_matrix(
            (2*number_of_nodes, 2*number_of_nodes),
            dtype=np.float64)
        ku = sparse.lil_matrix(
            (2*number_of_nodes, 2*number_of_nodes),
            dtype=np.float64)
        kuv = sparse.lil_matrix(
            (2*number_of_nodes, number_of_nodes),
            dtype=np.float64)
        kve = sparse.lil_matrix(
            (number_of_nodes, number_of_nodes),
            dtype=np.float64)
        ctheta = sparse.lil_matrix(
            (number_of_nodes, number_of_nodes),
            dtype=np.float64)
        ktheta = sparse.lil_matrix(
            (number_of_nodes, number_of_nodes),
            dtype=np.float64)

        for element_index, element in enumerate(self.mesh_data.elements):
            node_points = self.node_points[element_index]

            # TODO Check if its necessary to calculate all integrals
            # --> Dirichlet nodes could be leaved out?
            # Multiply with jacobian_det because its integrated with respect
            # to local coordinates.
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
            ctheta_e = (
                integral_m(node_points, element_order)
                * self.material_manager.get_density(element_index)
                * self.material_manager.get_heat_capacity(element_index)
                * 2 * np.pi
            )
            ktheta_e = (
                integral_ktheta(node_points, element_order)
                * self.material_manager.get_thermal_conductivity(element_index)
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

                    # Mtheta_e and Ktheta_e are 3x3 matrices too.
                    ctheta[global_p, global_q] += ctheta_e[local_p, local_q]
                    ktheta[global_p, global_q] += ktheta_e[local_p, local_q]

        # Calculate damping matrix
        cu = (
            self.material_manager.get_alpha_m(0) * mu
            + self.material_manager.get_alpha_k(0) * ku
        )

        # Calculate block matrices
        zeros1x1 = np.zeros((number_of_nodes, number_of_nodes))
        zeros2x1 = np.zeros((2*number_of_nodes, number_of_nodes))
        zeros1x2 = np.zeros((number_of_nodes, 2*number_of_nodes))

        m = sparse.bmat([
            [mu, zeros2x1, zeros2x1],
            [zeros1x2, zeros1x1, zeros1x1],
            [zeros1x2, zeros1x1, zeros1x1]
        ])
        c = sparse.bmat([
            [cu, zeros2x1, zeros2x1],
            [zeros1x2, zeros1x1, zeros1x1],
            [zeros1x2, zeros1x1, ctheta]
        ])
        k = sparse.bmat([
            [ku, kuv, zeros2x1],
            [kuv.T, -1*kve, zeros1x1],
            [zeros1x2, zeros1x1, ktheta]
        ])

        self.m = m.tolil()
        self.c = c.tolil()
        self.k = k.tolil()

    def simulate(
        self,
        delta_t: float,
        number_of_time_steps: int,
        gamma: float,
        beta: float,
        theta_start: Union[None, npt.NDArray] = None
    ):
        """Runs the time domain simulation. Dirichlet boundary conditions must
        be set beforehand.
        The resulting displacement field is stored in self.u, which contains
        the fields u_r, u_z, phi and theta.
        The mechanical losses are calculated in each time step and used as a
        source for the thermal field.

        Parameters:
            delta_t: Distance between two time steps.
            number_of_time_steps: Total number of time steps for the
                simulation.
            gamma: Newmark integration parameter.
            beta: Newmark integration parameter.
            theta_start: Possible initial field for theta.
        """
        elements = self.mesh_data.elements
        nodes = self.mesh_data.nodes
        element_order = self.mesh_data.element_order
        points_per_element = int(1/2*(element_order+1)*(element_order+2))
        node_count = len(nodes)
        element_count = len(elements)
        dirichlet_nodes = np.array(self.dirichlet_nodes)
        dirichlet_values = np.array(self.dirichlet_values)

        m = self.m
        c = self.c
        k = self.k

        # Init arrays
        # Displacement u which is calculated
        u = np.zeros((number_of_time_steps, 4*node_count), dtype=np.float64)
        # Displacement u derived after t (du/dt)
        v = np.zeros((number_of_time_steps, 4*node_count), dtype=np.float64)
        # v derived after u (d^2u/dt^2)
        a = np.zeros((number_of_time_steps, 4*node_count), dtype=np.float64)

        m, c, k = mat_apply_dbcs(
            m,
            c,
            k,
            self.dirichlet_nodes
        )

        k_star = (k+gamma/(beta*delta_t)*c+1/(beta*delta_t**2)*m).tocsr()

        # Mechanical loss calculated during simulation (for thermal field)
        mech_loss = np.zeros(
            shape=(number_of_time_steps, element_count),
            dtype=np.float64
        )

        volumes = calculate_volumes(
            self.node_points,
            self.mesh_data.element_order
        )

        if theta_start is not None:
            if len(theta_start) != node_count:
                raise ValueError(
                    "theta start must have the size of number of nodes"
                )
            u[3*node_count:, 0] = theta_start

        print("Starting simulation")
        for time_index in range(number_of_time_steps-1):
            # Check if new assembly is needed when temperature dependent
            # material parameters are used
            if self.material_manager.is_temperature_dependent:
                temp_field_per_elements = get_avg_temp_field_per_element(
                    u[time_index, 3*node_count:],
                    self.mesh_data.elements
                )
                update = self.material_manager.update_temperature(
                    temp_field_per_elements
                )
                if update:
                    # Assemble the matrices with new parameters
                    print(f"Assemble new (time step: {time_index})")
                    self.assemble()
                    m, c, k = mat_apply_dbcs(
                        self.m,
                        self.c,
                        self.k,
                        self.dirichlet_nodes
                    )
                    k_star = (
                        k
                        + gamma/(beta*delta_t)*c
                        + 1/(beta*delta_t**2)*m
                    ).tocsr()

            # Calculate load vector and add dirichlet boundary conditions
            f = self.calculate_load_vector(mech_loss[time_index, :])
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
                    + gamma/(beta*delta_t)*c)*u_tilde)
            )
            # Perform corrector step
            a[time_index+1, :] = (u[time_index+1, :]-u_tilde)/(beta*delta_t**2)
            v[time_index+1, :] = v_tilde + gamma*delta_t*a[time_index+1, :]

            # Calculate power_loss
            for element_index, element in enumerate(elements):
                node_points = self.node_points[element_index]

                # Get field values at current element at specific time index
                u_e = np.zeros(2*points_per_element)
                u_e_t_minus_1 = np.zeros(2*points_per_element)
                u_e_t_minus_2 = np.zeros(2*points_per_element)
                for i in range(points_per_element):
                    u_e[2*i] = u[time_index+1, 2*element[i]]
                    u_e[2*i+1] = u[time_index+1, 2*element[i]+1]
                    u_e_t_minus_1[2*i] = u[time_index, 2*element[i]]
                    u_e_t_minus_1[2*i+1] = u[time_index, 2*element[i]+1]
                    u_e_t_minus_2[2*i] = u[time_index-1, 2*element[i]]
                    u_e_t_minus_2[2*i+1] = u[time_index-1, 2*element[i]+1]

                # The mech loss of the element is divided by the volume
                # because it must be a power density.
                if time_index > 0:
                    mech_loss[time_index+1, element_index] = (
                        integral_loss_scs_time(
                            node_points,
                            u_e,
                            u_e_t_minus_1,
                            u_e_t_minus_2,
                            delta_t,
                            self.material_manager.get_elasticity_matrix(
                                element_index
                            ),
                            element_order
                        )
                        * 2 * np.pi
                        * self.material_manager.get_alpha_k(element_index)
                        * 1/volumes[element_index]
                    )

            if (time_index + 1) % 100 == 0:
                print(f"Finished time step {time_index+1}")

        self.u = u
        self.mech_loss = mech_loss

    def calculate_load_vector(
        self,
        mech_loss_density: npt.NDArray
    ) -> npt.NDArray:
        """Calculates the load vector (right hand side) vector for the
        simulation based on the mechanical loss density.

        Parameters:
            mech_loss_density: Power loss density of every element of the
                current time step.

        Returns:
            Right hand side vector for the simulation.
        """
        nodes = self.mesh_data.nodes
        element_order = self.mesh_data.element_order
        points_per_element = int(1/2*(element_order+1)*(element_order+2))
        number_of_nodes = len(nodes)

        # Can be initialized to 0 because external load and volume charge
        # density is 0.
        f = np.zeros(4*number_of_nodes, dtype=np.float64)

        # Calculation for theta load.
        # It needs to be assembled every step and every element since the
        # power is time- and position-dependent
        f_theta = np.zeros(number_of_nodes)

        for element_index, element in enumerate(self.mesh_data.elements):
            node_points = np.zeros(shape=(2, points_per_element))
            for node_index in range(points_per_element):
                node_points[:, node_index] = [
                    nodes[element[node_index]][0],
                    nodes[element[node_index]][1]
                ]

            f_theta_e = integral_theta_load(
                node_points,
                mech_loss_density[element_index],
                element_order
            ) * 2 * np.pi

            for local_p, global_p in enumerate(element):
                f_theta[global_p] += f_theta_e[local_p]

        f[3*number_of_nodes:] = f_theta

        return f
