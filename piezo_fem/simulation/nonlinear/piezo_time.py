"""Module for the simulation of nonlinear piezoelectric systems"""

# Third party libraries
from typing import List
import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse import linalg

# Local libraries
from ..base import SimulationData, MeshData, gradient_local_shape_functions, \
        local_to_global_coordinates, b_operator_global, integral_m, \
        integral_ku, integral_kuv, integral_kve, \
        create_local_element_data, LocalElementData, \
        quadratic_quadrature
from piezo_fem.simulation.piezo_time import charge_integral_u, \
    charge_integral_v
from piezo_fem.materials import MaterialManager


def integral_u_nonlinear(
        node_points: npt.NDArray,
        jacobian_inverted_t: npt.NDArray,
        nonlinear_elasticity_matrix: npt.NDArray):
    """Calculates the Ku integral

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        jacobian_inverted_t: Jacobian matrix inverted and transposed, needed
            for calculation of global derivatives.
        elasticity_matrix: Elasticity matrix for the current element
            (c matrix).

    Returns:
        npt.NDArray: 6x6 Ku matrix for the given element.
    """
    def inner(s, t):
        b_op = b_operator_global(node_points, s, t, jacobian_inverted_t)
        r = local_to_global_coordinates(node_points, s, t)[0]

        return np.dot(np.dot(b_op.T, nonlinear_elasticity_matrix), b_op)*r

    return quadratic_quadrature(inner)


class NonlinearPiezoSim:

    # Simulation parameters
    mesh_data: MeshData
    material_manager: MaterialManager
    simulation_data: SimulationData

    # Dirichlet boundary condition
    dirichlet_nodes_u: npt.NDArray
    dirichlet_values_u: npt.NDArray
    dirichlet_nodes_phi: npt.NDArray
    dirichlet_values_phi: npt.NDArray

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
            simulation_data: SimulationData):
        self.mesh_data = mesh_data
        self.material_manager = material_manager
        self.simulation_data = simulation_data

    def assemble(self, nonlinear_elasticity_matrix):
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
            l_e = (
                integral_u_nonlinear(
                    node_points,
                    jacobian_inverted_t,
                    nonlinear_elasticity_matrix
                ) * jacobian_det * 2 * np.pi
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

                    # L_e is similar to Ku_e
                    lu[2*global_p:2*global_p+2, 2*global_q:2*global_q+2] += \
                        l_e[2*local_p:2*local_p+2, 2*local_q:2*local_q+2]

        # Calculate damping matrix
        # Currently in the simulations only one material is used
        # TODO Add algorithm to calculate cu for every element on its own
        # in order to use multiple materials
        cu = (
            self.material_manager.get_alpha_m(0) * mu
            + self.material_manager.get_alpha_k(0) * ku
        )

        self.mu = mu.tolil()
        self.cu = cu.tolil()
        self.kuv = kuv.tolil()
        self.kv = kve.tolil()
        self.k_snake = lu.tolil()
        self.ku = ku.tolil()

        """
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
        ll = sparse.bmat([
            [lu, zeros2x1],
            [zeros1x2, zeros1x1]
        ])

        self.m = m.tolil()
        self.c = c.tolil()
        self.k = k.tolil()
        self.l = ll.tolil()
        """

    def solve_time(
            self,
            electrode_elements: npt.NDArray):
        number_of_time_steps = self.simulation_data.number_of_time_steps
        delta_t = self.simulation_data.delta_t
        number_of_nodes = len(self.mesh_data.nodes)

        mu = self.mu
        ku = self.ku
        cu = self.cu
        kuv = self.kuv
        kv = self.kv
        k_snake = self.k_snake

        # Init arrays
        # Displacement u which is calculated
        u = np.zeros(
            (2*number_of_nodes, number_of_time_steps),
            dtype=np.float64
        )
        phi = np.zeros(
            (number_of_nodes, number_of_time_steps),
            dtype=np.float64
        )

        # Charge calculated during simulation (for impedance)
        q = np.zeros(number_of_time_steps, dtype=np.float64)

        # Set boundary conditions
        system_matrix_u = NonlinearPiezoSim.apply_dirichlet_bc(
            (mu+delta_t/2*cu).tolil(),
            self.dirichlet_nodes_u
        ).tocsc()
        system_matrix_phi = NonlinearPiezoSim.apply_dirichlet_bc(
            kv,
            self.dirichlet_nodes_phi
        ).tocsc()

        print("Starting nonlinear piezoelectric simulation")
        for time_index in range(1, number_of_time_steps-1):

            # Solve for phi
            right_side_phi = NonlinearPiezoSim.apply_dirchlet_bc_loadvector(
                kuv.T*u[:, time_index],
                self.dirichlet_nodes_phi,
                self.dirichlet_values_phi[:, time_index]
            )
            phi[:, time_index+1] = linalg.spsolve(
                system_matrix_phi,
                right_side_phi
            )

            # TODO The whole scheme can be made faster using matrix inversion
            # Solve for u
            p = -kuv*phi[:, time_index+1]
            r = k_snake*np.square(u[:, time_index])+ku*u[:, time_index]
            right_side_u = NonlinearPiezoSim.apply_dirchlet_bc_loadvector(
                (
                    (delta_t)**2*(p-r)
                    + delta_t/2*cu*u[:, time_index-1]
                    + mu*(2*u[:, time_index]-u[:, time_index-1])
                ),
                self.dirichlet_nodes_u,
                self.dirichlet_values_u[:, time_index]
            )
            u[:, time_index+1] = linalg.spsolve(
                system_matrix_u,
                right_side_u
            )

            if electrode_elements is not None:
                # Caluclate charge
                q[time_index] = NonlinearPiezoSim.calculate_charge(
                    u[:, time_index+1],
                    phi[:, time_index+1],
                    self.material_manager,
                    electrode_elements,
                    self.mesh_data.nodes
                )
            if time_index % 100 == 0:
                print(f"Finished time step {time_index}")

        self.phi = phi
        self.q = q
        self.u = u

    def get_load_vector(
            self,
            nodes: npt.NDArray,
            values: npt.NDArray) -> npt.NDArray:
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
        f = np.zeros(number_of_nodes, dtype=np.float64)

        for node, value in zip(nodes, values):
            f[node] = value

        return f

    @staticmethod
    def calculate_charge(
            u: npt.NDArray,
            phi: npt.NDArray,
            material_manager: MaterialManager,
            electrode_elements: npt.NDArray,
            nodes: npt.NDArray) -> float:
        q = 0

        for element_index, element in enumerate(electrode_elements):
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
            jacobian_det = nodes[element[0]][0]-nodes[element[1]][0]

            u_e = np.array([
                u[2*element[0]],
                u[2*element[0]+1],
                u[2*element[1]],
                u[2*element[1]+1],
                u[2*element[2]],
                u[2*element[2]+1]
            ])
            phi_e = np.array([
                phi[element[0]],
                phi[element[1]],
                phi[element[2]]
            ])

            q_u = charge_integral_u(
                node_points,
                u_e,
                material_manager.get_piezo_matrix(element_index),
                jacobian_inverted_t
            ) * 2 * np.pi * jacobian_det
            q_v = charge_integral_v(
                node_points,
                phi_e,
                material_manager.get_permittivity_matrix(element_index),
                jacobian_inverted_t
            ) * 2 * np.pi * jacobian_det
            q += q_u - q_v

        return q

    @staticmethod
    def apply_dirchlet_bc_loadvector(
            load_vector: npt.NDArray,
            nodes: npt.NDArray,
            values: npt.NDArray):
        for node, value in zip(nodes, values):
            load_vector[node] = value

        return load_vector

    @staticmethod
    def apply_dirichlet_bc(
            system_matrix,
            nodes: npt.NDArray):
        for node in nodes:
            system_matrix[node, :] = 0
            system_matrix[node, node] = 1

        return system_matrix

