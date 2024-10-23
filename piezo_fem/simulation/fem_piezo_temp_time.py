"""Main module for the simulation of piezoelectric systems with
thermal field."""

# Python standard libraries
import numpy as np
import numpy.typing as npt
import scipy.sparse as sparse
import scipy.sparse.linalg as slin
from scipy import integrate

# Local libraries
from .base import MaterialData, SimulationData, MeshData, \
    local_shape_functions, gradient_local_shape_functions, \
    local_to_global_coordinates, b_operator_global, integral_m, \
    integral_ku, integral_kuv, integral_kve, apply_dirichlet_bc, \
    quadratic_quadrature
from .fem_piezo_time import calculate_charge
from .fem_heat_conduction_time import integral_ktheta, integral_theta_load


def integral_volume(node_points: npt.NDArray) -> float:
    """Calculates the volume of the triangle given by the node points.
    HINT: Must be multiplied with 2*np.pi and the jacobian determinant in order
    to give the correct volume of any rotationsymmetric triangle.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.

    Returns:
        Volume of the triangle.
    """
    def inner(s, t):
        r = local_to_global_coordinates(node_points, s, t)[0]
        return r

    return quadratic_quadrature(inner)


def loss_integral_scs(
        node_points: npt.NDArray,
        u_e_t: npt.NDArray,
        u_e_t_minus_1: npt.NDArray,
        u_e_t_minus_2: npt.NDArray,
        delta_t: float,
        jacobian_inverted_t: npt.NDArray,
        elasticity_matrix: npt.NDArray):
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
    """
    def inner(s, t):
        r = local_to_global_coordinates(node_points, s, t)[0]
        b_opt = b_operator_global(node_points, s, t, jacobian_inverted_t)
        #dt_s = np.dot(
        #    b_opt,
        #    (3*u_e_t-4*u_e_t_minus_1+u_e_t_minus_2)/(2*delta_t)
        #)
        S = np.dot(b_opt, u_e_t)
        S_t_minus_1 = np.dot(b_opt, u_e_t_minus_1)
        S_t_minus_2 = np.dot(b_opt, u_e_t_minus_2)
        dt_s = (3*S-4*S_t_minus_1+S_t_minus_2)/(2*delta_t)
        return np.dot(dt_s.T, np.dot(elasticity_matrix.T, dt_s))*r

    return quadratic_quadrature(inner)


def loss_integral_scds(
        node_points: npt.NDArray,
        u_e_t: npt.NDArray,
        u_e_t_minus_1: npt.NDArray,
        u_e_t_minus_2: npt.NDArray,
        delta_t: float,
        jacobian_inverted_t: npt.NDArray,
        elasticity_matrix: npt.NDArray):
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
    """
    def inner(s, t):
        b_opt = b_operator_global(node_points, s, t, jacobian_inverted_t)
        dt_s = np.dot(
            b_opt,
            (3*u_e_t-4*u_e_t_minus_1+u_e_t_minus_2)/(2*delta_t)
        )
        return np.dot(
            np.dot(b_opt, u_e_t).T,
            np.dot(elasticity_matrix.T, dt_s))

    return quadratic_quadrature(inner)


def loss_integral_eeds(
        node_points,
        u_e_t: npt.NDArray,
        u_e_t_minus_1: npt.NDArray,
        u_e_t_minus_2: npt.NDArray,
        v,
        delta_t: float,
        jacobian_inverted_t: npt.NDArray,
        piezo_matrix,):
    def inner(s, t):
        b_opt = b_operator_global(node_points, s, t, jacobian_inverted_t)
        #dt_s = np.dot(
        #    b_opt,
        #    (3*u_e_t-4*u_e_t_minus_1+u_e_t_minus_2)/(2*delta_t)
        #)
        S = np.dot(b_opt, u_e_t)
        S_t_minus_1 = np.dot(b_opt, u_e_t_minus_1)
        S_t_minus_2 = np.dot(b_opt, u_e_t_minus_2)
        dt_s = (3*S-4*S_t_minus_1+S_t_minus_2)/(2*delta_t)
        e = -1*np.dot(gradient_local_shape_functions(), v)

        return np.dot(e.T, np.dot(piezo_matrix, dt_s))

    return quadratic_quadrature(inner)


def energy_integral_theta(
        node_points: npt.NDArray,
        theta: npt.NDArray):
    """Integrates the given element over the given theta field.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        theta: List of the temperature field values of the points
            [theta1, theta2, theta3].
    """
    def inner(s, t):
        n = local_shape_functions(s, t)
        r = local_to_global_coordinates(node_points, s, t)[0]

        return np.dot(n.T, theta)*r

    return quadratic_quadrature(inner)


class PiezoSimTherm:
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
        temp_field_energy: Energy stored in the temperature field for every
            time step.
    """
    # Simulation parameters
    mesh_data: MeshData
    material_data: MaterialData
    simulation_data: SimulationData

    # Dirichlet boundary condition
    dirichlet_nodes: npt.NDArray
    dirichlet_values: npt.NDArray

    # FEM matrices
    m: npt.NDArray
    c: npt.NDArray
    k: npt.NDArray

    # Resulting fields
    u: npt.NDArray
    q: npt.NDArray
    mech_loss: npt.NDArray
    temp_field_energy: npt.NDArray

    def __init__(
            self,
            mesh_data: MeshData,
            material_data: MaterialData,
            simulation_data: SimulationData):
        self.mesh_data = mesh_data
        self.material_data = material_data
        self.simulation_data = simulation_data

    def assemble(self):
        """Assembles the FEM matrices based on the set mesh_data and
        material_data.
        The matrices are stored in self.m, self.c, self.k.
        """
        # TODO Assembly takes long rework this algorithm?
        # Maybe the 2x2 matrix slicing is not very fast
        nodes = self.mesh_data.nodes

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

        for element in self.mesh_data.elements:
            dn = gradient_local_shape_functions()
            # Get node points of element in format
            # [x1 x2 x3]
            # [y1 y2 y3] where (xi, yi) are the coordinates for Node i
            node_points = np.array([
                [nodes[element[0]][0],
                 nodes[element[1]][0],
                 nodes[element[2]][0]],
                [nodes[element[0]][1],
                 nodes[element[1]][1],
                 nodes[element[2]][1]]
            ])
            jacobian = np.dot(node_points, dn.T)
            jacobian_inverted_t = np.linalg.inv(jacobian).T
            jacobian_det = np.linalg.det(jacobian)

            # TODO Check if its necessary to calculate all integrals
            # --> Dirichlet nodes could be leaved out?
            # Multiply with jac_det because its integrated with respect to
            # local coordinates.
            # Mutiply with 2*pi because theta is integrated from 0 to 2*pi
            mu_e = (
                self.material_data.density
                * integral_m(node_points)
                * jacobian_det * 2 * np.pi
            )
            ku_e = (
                integral_ku(
                    node_points,
                    jacobian_inverted_t,
                    self.material_data.elasticity_matrix)
                * jacobian_det * 2 * np.pi
            )
            kuv_e = (
                integral_kuv(
                    node_points,
                    jacobian_inverted_t,
                    self.material_data.piezo_matrix)
                * jacobian_det * 2 * np.pi
            )
            kve_e = (
                integral_kve(
                    node_points,
                    jacobian_inverted_t,
                    self.material_data.permittivity_matrix)
                * jacobian_det * 2 * np.pi
            )
            ctheta_e = (
                integral_m(node_points)
                * self.material_data.density
                * self.material_data.heat_capacity
                * jacobian_det * 2 * np.pi
            )
            ktheta_e = (
                integral_ktheta(
                    node_points,
                    jacobian_inverted_t)
                * self.material_data.thermal_conductivity
                * jacobian_det*2*np.pi
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
        cu = self.material_data.alpha_m * mu + self.material_data.alpha_k * ku

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

    def solve_time(self,
                   electrode_elements: npt.NDArray,
                   set_symmetric_bc: bool = False):
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
            set_symmetric_bc: True if the symmetric boundary condition for u
                shall be set. False otherwise.
        """
        number_of_time_steps = self.simulation_data.number_of_time_steps
        beta = self.simulation_data.beta
        gamma = self.simulation_data.gamma
        delta_t = self.simulation_data.delta_t
        elements = self.mesh_data.elements
        nodes = self.mesh_data.nodes
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
            self.dirichlet_nodes[0],
            self.dirichlet_nodes[1],
            number_of_nodes)

        k_star = (k+gamma/(beta*delta_t)*c+1/(beta*delta_t**2)*m).tocsr()

        # Mechanical loss calculated during simulation (for thermal field)
        mech_loss = np.zeros(
            shape=(number_of_elements, number_of_time_steps),
            dtype=np.float64)
        temp_field_energy = np.zeros(
            shape=(number_of_time_steps),
            dtype=np.float64)

        print("Starting simulation")
        for time_index in range(number_of_time_steps-1):
            # Calculate load vector and add dirichlet boundary conditions
            if set_symmetric_bc:
                f = self.get_load_vector(
                        self.dirichlet_nodes[0],
                        self.dirichlet_values[0][time_index+1],
                        self.dirichlet_nodes[1],
                        self.dirichlet_values[1][time_index+1],
                        mech_loss[:, time_index])
            else:
                # If it is a ring there are no boundary conditions for u_r
                # or u_z.
                f = self.get_load_vector(
                        [],
                        [],
                        self.dirichlet_nodes[1],
                        self.dirichlet_values[1][time_index+1],
                        mech_loss[:, time_index])

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
                       + gamma/(beta*delta_t)*c)*u_tilde))
            # Perform corrector step
            a[:, time_index+1] = (u[:, time_index+1]-u_tilde)/(beta*delta_t**2)
            v[:, time_index+1] = v_tilde + gamma*delta_t*a[:, time_index+1]

            # Calculate charge
            q[time_index+1] = calculate_charge(
                u[:, time_index+1],
                self.material_data.permittivity_matrix,
                self.material_data.piezo_matrix,
                electrode_elements,
                self.mesh_data.nodes
            )

            # Calculate power_loss
            # TODO Calculate charge and power loss together (one loop)
            if time_index > 1:
                for element_index, element in enumerate(elements):
                    dn = gradient_local_shape_functions()
                    node_points = np.array([
                        [nodes[element[0]][0],
                         nodes[element[1]][0],
                         nodes[element[2]][0]],
                        [nodes[element[0]][1],
                         nodes[element[1]][1],
                         nodes[element[2]][1]]
                    ])
                    jacobian = np.dot(node_points, dn.T)
                    jacobian_inverted_t = np.linalg.inv(jacobian).T
                    jacobian_det = np.linalg.det(jacobian)
                    u_e = np.array([u[2*element[0], time_index],
                                    u[2*element[0]+1, time_index],
                                    u[2*element[1], time_index],
                                    u[2*element[1]+1, time_index],
                                    u[2*element[2], time_index],
                                    u[2*element[2]+1, time_index]])
                    u_e_t_minus_1 = np.array([
                                    u[2*element[0], time_index-1],
                                    u[2*element[0]+1, time_index-1],
                                    u[2*element[1], time_index-1],
                                    u[2*element[1]+1, time_index-1],
                                    u[2*element[2], time_index-1],
                                    u[2*element[2]+1, time_index-1]])
                    u_e_t_minus_2 = np.array([
                                    u[2*element[0], time_index-2],
                                    u[2*element[0]+1, time_index-2],
                                    u[2*element[1], time_index-2],
                                    u[2*element[1]+1, time_index-2],
                                    u[2*element[2], time_index-2],
                                    u[2*element[2]+1, time_index-2]])
                    v_e = np.array([
                        u[element[0]+2*number_of_nodes, time_index],
                        u[element[1]+2*number_of_nodes, time_index],
                        u[element[2]+2*number_of_nodes, time_index]
                    ])
                    theta_e = np.array([
                        u[element[0]+3*number_of_nodes, time_index],
                        u[element[1]+3*number_of_nodes, time_index],
                        u[element[2]+3*number_of_nodes, time_index]])
                    temp_field_energy[time_index] += (
                        energy_integral_theta(
                            node_points,
                            theta_e)
                        * self.material_data.heat_capacity
                        * self.material_data.density
                        * 2*np.pi*jacobian_det
                    )
                    # The mech loss of the element is divided by the volume
                    # because it must be a power density.
                    # TODO Is it possible to calculate the mech loss for the
                    # points directly instead of calculating it for the
                    # element and then splitting it later equally on the
                    # points?
                    volume = integral_volume(node_points)*2*np.pi*jacobian_det
                    mech_loss[element_index, time_index+1] = (
                        loss_integral_scs(
                            node_points,
                            u_e,
                            u_e_t_minus_1,
                            u_e_t_minus_2,
                            delta_t,
                            jacobian_inverted_t,
                            self.material_data.elasticity_matrix)
                        * self.material_data.alpha_k
                        #- loss_integral_eeds(
                        #    node_points,
                        #    u_e,
                        #    u_e_t_minus_1,
                        #    u_e_t_minus_2,
                        #    v_e,
                        #    delta_t,
                        #    jacobian_inverted_t,
                        #    self.material_data.piezo_matrix
                        #)
                    ) * 2*np.pi*jacobian_det * 1/volume

            if (time_index + 1) % 100 == 0:
                print(f"Finished time step {time_index+1}")

        self.q = q
        self.u = u
        self.mech_loss = mech_loss
        self.temp_field_energy = temp_field_energy

    def get_load_vector(
            self,
            nodes_u: npt.NDArray,
            values_u: npt.NDArray,
            nodes_v: npt.NDArray,
            values_v: npt.NDArray,
            mech_loss_density: npt.NDArray) -> npt.NDArray:
        """Calculates the load vector (right hand side) vector for the
        simulation.

        Parameters:
            nodes_u: Nodes where the dirichlet bc for u is set.
            values_u: Values of the u boundary condition for the nodes.
                Contains a tuple of values [u_r, u_z].
            nodes_v: Nodes where the dirichlet bc for v is set.
            values_v: Value of the v boundary condition for the nodes.
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
        number_of_nodes = len(nodes)

        # Can be initialized to 0 because external load and volume charge
        # density is 0.
        f = np.zeros(4*number_of_nodes, dtype=np.float64)

        # Set dirichlet values for u_r only
        for node, value in zip(nodes_u, values_u):
            f[2*node] = value[0]

        # Set dirichlet values for v
        # Use offset because v nodes have higher index than u nodes
        offset = 2*number_of_nodes

        for node, value in zip(nodes_v, values_v):
            f[node+offset] = value

        # Calculation for theta load.
        # It needs to be assembled every step and every element since the
        # power is time- and position-dependent
        f_theta = np.zeros(number_of_nodes)

        for element_index, element in enumerate(self.mesh_data.elements):
            dn = gradient_local_shape_functions()
            node_points = np.array([
                [nodes[element[0]][0],
                 nodes[element[1]][0],
                 nodes[element[2]][0]],
                [nodes[element[0]][1],
                 nodes[element[1]][1],
                 nodes[element[2]][1]]
            ])
            jacobian = np.dot(node_points, dn.T)
            jacobian_det = np.linalg.det(jacobian)

            f_theta_e = integral_theta_load(
                node_points,
                mech_loss_density[element_index])*2*np.pi*jacobian_det

            for local_p, global_p in enumerate(element):
                f_theta[global_p] += f_theta_e[local_p]

        f[3*number_of_nodes:] = f_theta

        return f
