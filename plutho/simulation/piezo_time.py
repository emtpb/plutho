"""Main module for the simulation of piezoelectric systems."""

# Python standard libraries
from typing import List
import numpy as np
import numpy.typing as npt
import scipy.sparse as sparse
import scipy.sparse.linalg as slin

# Local libraries
from .base import SimulationData, MeshData, \
    gradient_local_shape_functions, \
    local_to_global_coordinates, b_operator_global, integral_m, \
    integral_ku, integral_kuv, integral_kve, apply_dirichlet_bc, \
    line_quadrature, create_local_element_data, LocalElementData
from ..materials import MaterialManager


def charge_integral_u(
    node_points: npt.NDArray,
    u_e: npt.NDArray,
    piezo_matrix: npt.NDArray,
    jacobian_inverted_t: npt.NDArray
):
    """Calculates the integral of eBu of the given element.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        u_e: List of u values at the nodes of the triangle
            [u1_r, u1_z, u2_r, u2_z, u3_r, u3_z].
        piezo_matrix: Piezo matrix for the current element (e matrix).
        jacobian_inverted_t: Jacobian matrix inverted and transposed, needed
            for calculation of global derivatives.

    Returns:
        Float: Integral of eBu of the current triangle.
    """
    def inner(s, t):
        r = local_to_global_coordinates(node_points, s, t)[0]
        b_opt_global = b_operator_global(
            node_points,
            s,
            t,
            jacobian_inverted_t)
        # [1] commponent is taken because the normal of the
        # boundary line is in z-direction
        # -1 because the normal points inwards
        return -np.dot(np.dot(piezo_matrix, b_opt_global), u_e)[1]*r

    return line_quadrature(inner)


def charge_integral_v(
    node_points: npt.NDArray,
    v_e: npt.NDArray,
    permittivity_matrix: npt.NDArray,
    jacobian_inverted_t: npt.NDArray
):
    """Calculates the integral of epsilonBVe of the given element.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        v_e: List of u values at the nodes of the triangle
            [v1, v2, v3].
        permittivity_matrix: Permittivity matrix for the current
            element (e matrix).
        jacobian_inverted_t: Jacobian matrix inverted and transposed, needed
            for calculation of global derivatives.

    Returns:
        Float: Integral of epsilonBVe of the current triangle.
    """
    def inner(s, t):
        r = local_to_global_coordinates(node_points, s, t)[0]
        dn = gradient_local_shape_functions()
        global_dn = np.dot(jacobian_inverted_t, dn)

        return -np.dot(np.dot(permittivity_matrix, global_dn), v_e)[1]*r

    return line_quadrature(inner)


def calculate_charge(
    u: npt.NDArray,
    material_manager: MaterialManager,
    elements: npt.NDArray,
    nodes
) -> float:
    """Calculates the charge of the given elements.

    Parameters:
        u: List of u values for every node.
        material_manager: MaterialManager object.
        elements: Elements for which the charge shall be calculated.
        nodes: All nodes used in the simulation.
    """
    number_of_nodes = len(nodes)
    q = 0

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

        # TODO Check why the jacobian_det did not work
        # It should work with line elements instead of triangle elements
        jacobian = np.dot(node_points, dn.T)
        jacobian_inverted_t = np.linalg.inv(jacobian).T

        # Since its a line integral along the top edge of the model where the
        # triangles are aligned in a line.
        # it is necessarry to use a different determinant. In this case the
        # determinant is the difference between the.
        # first 2 points of the triangle (which are always on the boundary
        # line in gmsh).
        jacobian_det = nodes[element[0]][0]-nodes[element[1]][0]

        u_e = np.array([u[2*element[0]],
                        u[2*element[0]+1],
                        u[2*element[1]],
                        u[2*element[1]+1],
                        u[2*element[2]],
                        u[2*element[2]+1]])
        ve_e = np.array([u[element[0]+2*number_of_nodes],
                         u[element[1]+2*number_of_nodes],
                         u[element[2]+2*number_of_nodes]])

        q_u = charge_integral_u(
            node_points,
            u_e,
            material_manager.get_piezo_matrix(element_index),
            jacobian_inverted_t
        ) * 2 * np.pi * jacobian_det
        q_v = charge_integral_v(
            node_points,
            ve_e,
            material_manager.get_permittivity_matrix(element_index),
            jacobian_inverted_t
        ) * 2 * np.pi * jacobian_det
        q += q_u - q_v

    return q


class PiezoSimTime:
    """Class for the simulation of mechanical-electric fields.

    Parameters:
        mesh_data: MeshData format.
        material_manager: MaterialManager object.
        simulation_data: SimulationData format.

    Attributes:
        mesh_data: Contains the mesh information.
        material_manager: Contains information about the materials.
        simulation_data: Contains the information about the simulation.
        dirichlet_nodes: List of nodes for which dirichlet values are set.
        dirichlet_values: List of values of the corresponding dirichlet nodes
            per time step.
        m: Mass matrix.
        c: Damping matrix.
        k: Stiffness matrix.
        u: Resulting vector of size 4*number_of_nodes containing u_r, u_z
            and v. u[node_index, time_index]. An offset needs to be
            added to the node_index in order to access the different field
            paramteters:
                u_r: 0,
                u_z: 1*number_of_nodes,
                v: 2*number_of_nodes
        q: Resulting charges for each time step q[time_index].
    """
    # Simulation parameters
    mesh_data: MeshData
    material_manager: MaterialManager
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

    # Internal simulation data
    local_elements: List[LocalElementData]

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
        # TODO Assembly takes to long rework this algorithm
        # Maybe the 2x2 matrix slicing is not very fast
        nodes = self.mesh_data.nodes
        elements = self.mesh_data.elements
        self.local_elements = create_local_element_data(nodes, elements)

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
            mu_e = (
                self.material_manager.get_density(element_index)
                * integral_m(node_points)
                * jacobian_det * 2 * np.pi
            )
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

    def solve_time(
        self,
        electrode_elements: npt.NDArray
    ):
        """Runs the simulation using the assembled m, c and k matrices as well
        as the set excitation.
        Calculates the displacement field and potential field of the given
        model (all saved in u).
        Calculates the electric charge in the electrode.
        Saves u[node_index, time_index] and q[time_index].

        Parameters:
            electrode_elements: List of all elements which are in
            the electrode.
        """
        number_of_time_steps = self.simulation_data.number_of_time_steps
        beta = self.simulation_data.beta
        gamma = self.simulation_data.gamma
        delta_t = self.simulation_data.delta_t

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

        # Set initial value of potential at electrode nodes to given excitation
        # TODO Could be removed?
        for index, node in enumerate(self.dirichlet_nodes):
            u[node, 0] = self.dirichlet_values[index, 0]

        print("Starting simulation")
        for time_index in range(number_of_time_steps-1):
            # Calculate load vector and add dirichlet boundary conditions
            f = self.get_load_vector(
                    self.dirichlet_nodes,
                    self.dirichlet_values[:, time_index+1]
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
                       + gamma/(beta*delta_t)*c)*u_tilde))
            # Perform corrector step
            a[:, time_index+1] = (u[:, time_index+1]-u_tilde)/(beta*delta_t**2)
            v[:, time_index+1] = v_tilde + gamma*delta_t*a[:, time_index+1]

            # Calculate charge
            if electrode_elements is not None:
                q[time_index+1] = calculate_charge(
                    u[:, time_index+1],
                    self.material_manager,
                    electrode_elements,
                    self.mesh_data.nodes
                )

            if (time_index + 1) % 100 == 0:
                print(f"Finished time step {time_index+1}")

        self.q = q
        self.u = u

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
