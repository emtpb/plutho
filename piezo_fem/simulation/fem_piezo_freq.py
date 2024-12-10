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
from .fem_piezo_time import calculate_charge
from ..materials import MaterialManager


class PiezoFreqSim:
    """Class for the simulation of mechanical-electric fields.

    Parameters:
        mesh_data: MeshData format.
        material_manager: MaterialManager object.
        simulation_data: SimulationData format.

    Attributes:
        mesh_data: Contains the mesh information.
        material_manager: Contains the information about the materials.
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
            material_manager: MaterialManager):
        self.mesh_data = mesh_data
        self.material_manager = material_manager

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
            dtype=np.complex128
        )
        ku = sparse.lil_matrix(
            (2*number_of_nodes, 2*number_of_nodes),
            dtype=np.complex128
        )
        kuv = sparse.lil_matrix(
            (2*number_of_nodes, number_of_nodes),
            dtype=np.complex128
        )
        kve = sparse.lil_matrix(
            (number_of_nodes, number_of_nodes),
            dtype=np.complex128
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
                self.material_manager.get_density()
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
        cu = (
            self.material_manager.get_alpha_m() * mu
            + self.material_manager.get_alpha_k() * ku
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


    def get_load_vector(
            self,
            nodes_u: npt.NDArray,
            values_u: npt.NDArray,
            nodes_v: npt.NDArray,
            values_v: npt.NDArray) -> npt.NDArray:
        """Calculates the load vector (right hand side) vector for the
        simulation.

        Parameters:
            nodes_u: Nodes where the dirichlet bc for u is set.
            values_u: Values of the u boundary condition for the nodes.
                Contains a tuple of values [u_r, u_z].
            nodes_v: Nodes where the dirichlet bc for v is set.
            values_v: Value of the v boundary condition for the nodes.

        Returns:
            Right hand side vector for the simulation.
        """
        number_of_nodes = len(self.mesh_data.nodes)

        # Can be initialized to 0 because external load and volume
        # charge density is 0.
        f = np.zeros(3*number_of_nodes, dtype=np.float64)

        # Set dirichlet values for u_r only
        for node, value in zip(nodes_u, values_u):
            f[2*node] = value[0]

        # Set dirichlet values for v
        # Use offset because v nodes have higher index than u nodes
        offset = 2*number_of_nodes

        for node, value in zip(nodes_v, values_v):
            f[node+offset] = value

        return f


    def solve_frequency(
            self,
            frequencies,
            excitation_amplitudes,
            electrode_elements,
            set_symmetric_bc):
        if len(frequencies) != len(excitation_amplitudes):
            raise ValueError(
                "frequencies and excitation_elements must have the same length"
            )

        m = self.m
        c = self.c
        k = self.k

        u = np.zeros((m.shape[0], len(frequencies)), dtype=np.complex128)
        q = np.zeros((len(frequencies)), dtype=np.complex128)

        m, c, k = apply_dirichlet_bc(
            m,
            c,
            k,
            self.dirichlet_nodes[0],
            self.dirichlet_nodes[1],
            len(self.mesh_data.nodes)
        )

        for index, val in enumerate(zip(
                frequencies, excitation_amplitudes
            )):
            frequency, amplitude = val
            angular_frequency = 2*np.pi*frequency

            a = (
                - angular_frequency**2*m
                + 1j*angular_frequency*c
                + k
            )
            if set_symmetric_bc:
                f = self.get_load_vector(
                    self.dirichlet_nodes[0],
                    np.zeros((len(self.dirichlet_nodes[0]), 2)),
                    self.dirichlet_nodes[1],
                    self.dirichlet_values[1][index]
                )
            else:
                # If it is a ring there are no boundary conditions for u_r
                # or u_z.
                f = self.get_load_vector(
                    np.array([]),
                    np.array([]),
                    self.dirichlet_nodes[1],
                    self.dirichlet_values[1][index]
                )

            u[:, index] = slin.spsolve(a, f)

            q[index] = calculate_charge(
                u[:, index],
                self.material_manager,
                electrode_elements,
                self.mesh_data.nodes
            )

        self.u = u
        self.q = q
