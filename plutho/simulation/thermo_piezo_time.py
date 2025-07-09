"""Main module for the simulation of piezoelectric systems with
thermal field."""

# Python standard libraries
from typing import List
import numpy as np
import numpy.typing as npt
import scipy.sparse.linalg as slin
from scipy import sparse

# Local libraries
from .base import SimulationData, MeshData, \
    create_node_points, \
    local_to_global_coordinates, b_operator_global, integral_m, \
    integral_ku, integral_kuv, integral_kve, apply_dirichlet_bc, \
    quadratic_quadrature, calculate_volumes, \
    gradient_local_shape_functions_2d, get_avg_temp_field_per_element
from .piezo_time import calculate_charge
from .thermo_time import integral_ktheta, integral_theta_load
from ..materials import MaterialManager


def loss_integral_scs(
    node_points: npt.NDArray,
    u_e_t: npt.NDArray,
    u_e_t_minus_1: npt.NDArray,
    u_e_t_minus_2: npt.NDArray,
    delta_t: float,
    elasticity_matrix: npt.NDArray,
    element_order: int
):
    """Calculates the integral of dS/dt*c*dS/dt over one triangle. Since foward
    difference quotient of second oder is used the last 2 values of e_u are
    needed.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        u_e_t: Values of u_e at the current time point.
            Format: [u1_r, u1_z, u2_r, u2_z, u3_r, u3_z].
        u_e_t_minus_1: Values of u_e at one time point earlier. Same format
            as u_e_t.
        u_e_t_minus_2: Values of u_e at two time points earlier. Same format
            as u_e_t.
        delta_t: Difference between the time steps.
        jacobian_inverted_t: Jacobian matrix inverted and transposed, needed
            for calculation of global derivatives.
        elasticity_matrix: Elasticity matrix for the current element
            (c matrix).
        element_order: Order of the shape functions.
    """
    def inner(s, t):
        dn = gradient_local_shape_functions_2d(s, t, element_order)
        jacobian = np.dot(node_points, dn.T)
        jacobian_inverted_t = np.linalg.inv(jacobian).T
        jacobian_det = np.linalg.det(jacobian)

        r = local_to_global_coordinates(node_points, s, t, element_order)[0]
        b_opt = b_operator_global(
            node_points,
            jacobian_inverted_t,
            s,
            t,
            element_order
        )

        s_e = np.dot(b_opt, u_e_t)
        s_e_t_minus_1 = np.dot(b_opt, u_e_t_minus_1)
        s_e_t_minus_2 = np.dot(b_opt, u_e_t_minus_2)
        dt_s = (3*s_e-4*s_e_t_minus_1+s_e_t_minus_2)/(2*delta_t)

        return np.dot(
            dt_s.T,
            np.dot(
                elasticity_matrix.T,
                dt_s
            )
        ) * r * jacobian_det

    return quadratic_quadrature(inner, element_order)


class ThermoPiezoSimTime:
    """Class for the coupled simulation of mechanical-electric and
    thermal field.

    Parameters:
        mesh_data: MeshData format.
        material_data: MaterialData format.
        simulation_data: SimulationData format.

    Attributes:
        mesh_data: Contains the mesh information.
        material_data: Contains the information about the materials.
        simulation_data: Contains the information about the simulation.
        dirichlet_nodes: List of nodes for which dirichlet values are set.
        dirichlet_values: List of values of the corresponding dirichlet nodes
            per time step.
        m: Mass matrix.
        c: Damping matrix.
        k: Stiffness matrix.
        u: Resulting vector of size 4*number_of_nodes containing u_r, u_z, v
            and theta. u[node_index, time_index]. An offset needs to be
            added to the node_index in order to access the different field
            paramteters:
                u_r: 0,
                u_z: 1*number_of_nodes,
                v: 2*number_of_nodes,
                theta: 3*number_of_nodes
        q: Resulting charges for each time step q[time_index].
        mech_loss: Mechanical loss for each element per time step
            mech_loss[element_index, time_index].
    """
    # Simulation parameters
    mesh_data: MeshData
    material_manager: MaterialManager
    simulation_data: SimulationData

    # Dirichlet boundary condition
    dirichlet_nodes: npt.NDArray
    dirichlet_values: npt.NDArray

    # FEM matrices
    m: sparse.lil_array
    c: sparse.lil_array
    k: sparse.lil_array

    # Resulting fields
    u: npt.NDArray
    q: npt.NDArray
    mech_loss: npt.NDArray

    # Internal simulation data
    node_points: npt.NDArray

    def __init__(
        self,
        mesh_data: MeshData,
        material_manager: MaterialManager,
        simulation_data: SimulationData
    ):
        self.mesh_data = mesh_data
        self.material_manager = material_manager
        self.simulation_data = simulation_data

    def assemble(self):
        """Assembles the FEM matrices based on the set mesh_data and
        material_data.
        The matrices are stored in self.m, self.c, self.k.
        """
        # TODO Assembly takes long rework this algorithm?
        # Maybe the 2x2 matrix slicing is not very fast
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

    def solve_time(
        self,
        electrode_elements: npt.NDArray,
        electrode_normals: npt.NDArray,
        theta_start = None
    ):
        """Runs the simulation using the assembled m, c and k matrices as well
        as the set excitation.
        Calculates the displacement field, potential field and the thermal
        field of the given model (all saved in u).
        Calculates the electric charge in the elctrode and the
        mechanical losses of the whole body.
        Saves u[node_index, time_index], q[time_index] and
        mechanical_losses[element_index, time_index].

        Parameters:
            electrode_elements: List of all elements which are in
                the electrode.
            electrode_normals: List of normal vectors corresponding to the
                electrode_elements list.
            set_symmetric_bc: True if the symmetric boundary condition for u
                shall be set. False otherwise.
        """
        number_of_time_steps = self.simulation_data.number_of_time_steps
        beta = self.simulation_data.beta
        gamma = self.simulation_data.gamma
        delta_t = self.simulation_data.delta_t
        elements = self.mesh_data.elements
        nodes = self.mesh_data.nodes
        element_order = self.mesh_data.element_order
        points_per_element = int(1/2*(element_order+1)*(element_order+2))
        number_of_elements = len(elements)
        number_of_nodes = len(nodes)

        m = self.m
        c = self.c
        k = self.k

        # Init arrays
        # Displacement u which is calculated
        u = np.zeros((m.shape[0], number_of_time_steps), dtype=np.float64)
        # Displacement u derived after t (du/dt)
        v = np.zeros((m.shape[0], number_of_time_steps), dtype=np.float64)
        # v derived after u (d^2u/dt^2)
        a = np.zeros((m.shape[0], number_of_time_steps), dtype=np.float64)

        # Charge calculated during simulation (for impedance)
        q = np.zeros(number_of_time_steps, dtype=np.float64)

        m, c, k = apply_dirichlet_bc(
            m,
            c,
            k,
            self.dirichlet_nodes
        )

        k_star = (k+gamma/(beta*delta_t)*c+1/(beta*delta_t**2)*m).tocsr()

        # Mechanical loss calculated during simulation (for thermal field)
        mech_loss = np.zeros(
            shape=(number_of_elements, number_of_time_steps),
            dtype=np.float64
        )

        volumes = calculate_volumes(
            self.node_points,
            self.mesh_data.element_order
        )

        if theta_start is not None:
            if len(theta_start) != number_of_nodes:
                raise ValueError(
                    "theta start must have the size of number of nodes"
                )
            u[3*number_of_nodes:, 0] = theta_start

        print("Starting simulation")
        for time_index in range(number_of_time_steps-1):
            # Check if new assembly is needed when temperature dependent
            # material parameters are used
            if self.material_manager.is_temperature_dependent:
                temp_field_per_elements = get_avg_temp_field_per_element(
                    u[3*number_of_nodes:, time_index],
                    self.mesh_data.elements
                )
                update = self.material_manager.update_temperature(
                    temp_field_per_elements
                )
                if update:
                    # Assemble the matrices with new parameters
                    print(f"Assemble new (time step: {time_index})")
                    self.assemble()
                    m, c, k = apply_dirichlet_bc(
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
            f = self.get_load_vector(
                    self.dirichlet_nodes,
                    self.dirichlet_values[:, time_index+1],
                    mech_loss[:, time_index]
            )

            # Perform Newmark method
            # Predictor step
            u_tilde = (
                u[:, time_index]
                + delta_t*v[:, time_index]
                + delta_t**2/2*(1-2*beta)*a[:, time_index]
            )
            v_tilde = (
                v[:, time_index]
                + (1-gamma)*delta_t*a[:, time_index]
            )

            # Solve for next time step
            u[:, time_index+1] = slin.spsolve(
                k_star, (
                    f
                    - c*v_tilde
                    + (1/(beta*delta_t**2)*m
                       + gamma/(beta*delta_t)*c)*u_tilde)
            )
            # Perform corrector step
            a[:, time_index+1] = (u[:, time_index+1]-u_tilde)/(beta*delta_t**2)
            v[:, time_index+1] = v_tilde + gamma*delta_t*a[:, time_index+1]

            if electrode_elements is not None:
                q[time_index+1] = calculate_charge(
                    u[:, time_index+1],
                    self.material_manager,
                    electrode_elements,
                    electrode_normals,
                    nodes,
                    element_order
                )

            # Calculate power_loss
            for element_index, element in enumerate(elements):
                node_points = self.node_points[element_index]

                # Get field values at current element at specific time index
                u_e = np.zeros(2*points_per_element)
                u_e_t_minus_1 = np.zeros(2*points_per_element)
                u_e_t_minus_2 = np.zeros(2*points_per_element)
                for i in range(points_per_element):
                    u_e[2*i] = u[2*element[i], time_index+1]
                    u_e[2*i+1] = u[2*element[i]+1, time_index+1]
                    u_e_t_minus_1[2*i] = u[2*element[i], time_index]
                    u_e_t_minus_1[2*i+1] = u[2*element[i]+1, time_index]
                    u_e_t_minus_2[2*i] = u[2*element[i], time_index-1]
                    u_e_t_minus_2[2*i+1] = u[2*element[i]+1, time_index-1]

                # The mech loss of the element is divided by the volume
                # because it must be a power density.
                if time_index > 0:
                    mech_loss[element_index, time_index+1] = (
                        loss_integral_scs(
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

        self.q = q
        self.u = u
        self.mech_loss = mech_loss

    def get_load_vector(
        self,
        dirichlet_nodes: npt.NDArray,
        values: npt.NDArray,
        mech_loss_density: npt.NDArray
    ) -> npt.NDArray:
        """Calculates the load vector (right hand side) vector for the
        simulation.

        Parameters:
            nodes: Nodes where the dirichlet bc is set.
            values: Values at which the corresponding nodes are set.
            mech_loss_density: Power loss density of every element of the
                current time step.

        Returns:
            Right hand side vector for the simulation.
        """
        # For u and v the load vector is set to the corresponding dirichlet
        # values given by values_u and values_v
        # at the nodes nodes_u and nodes_v since there is no inner charge and
        # no forces given.
        # For the temperature field the load vector represents the temperature
        # sources given by the mechanical losses.
        nodes = self.mesh_data.nodes
        element_order = self.mesh_data.element_order
        points_per_element = int(1/2*(element_order+1)*(element_order+2))
        number_of_nodes = len(nodes)

        # Can be initialized to 0 because external load and volume charge
        # density is 0.
        f = np.zeros(4*number_of_nodes, dtype=np.float64)

        # Set dirichlet values for given ndoes and values
        for node, value in zip(dirichlet_nodes, values):
            f[node] = value

        # Calculation for theta load.
        # It needs to be assembled every step and every element since the
        # power is time- and position-dependent
        f_theta = np.zeros(number_of_nodes)

        for element_index, element in enumerate(self.mesh_data.elements):
            node_points = np.zeros(shape=(2, points_per_element))
            for node_index in range(points_per_element):
                node_points[element_index, :, node_index] = [
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
