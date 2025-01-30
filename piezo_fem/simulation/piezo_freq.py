"""Main module for the simulation of piezoelectric systems."""

# Python standard libraries
from typing import List
import numpy as np
import numpy.typing as npt
import scipy.sparse as sparse
import scipy.sparse.linalg as slin

# Local libraries
from .base import MeshData, local_to_global_coordinates, b_operator_global, \
    integral_m, integral_ku, integral_kuv, integral_kve, apply_dirichlet_bc, \
    create_local_element_data, LocalElementData, quadratic_quadrature, \
    calculate_volumes
from .piezo_time import calculate_charge
from ..materials import MaterialManager


def loss_integral_scs(
        node_points: npt.NDArray,
        u_e: npt.NDArray,
        angular_frequency: float,
        jacobian_inverted_t: npt.NDArray,
        elasticity_matrix: npt.NDArray):
    """Calculate sthe integral of dS/dt*c*dS/dt over one triangle in the
    frequency domain for the given frequency.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of one
            triangle.
        u_e: Displacement at this element [u1_r, u1_z, u_2r, u_2z, u_3r, u_3z].
        angular_frequency: Angular frequency at which the loss shall be
            calculated.
        jacobian_inverted_t: Jacobian matrix inverted and transposed, needed
            for calculation of global derivates.
        elasticity_matrix: Elasticity matrix for the current element (c matrix)
    """
    def inner(s, t):
        r = local_to_global_coordinates(node_points, s, t)[0]
        b_opt = b_operator_global(node_points, s, t, jacobian_inverted_t)

        s_e = np.dot(b_opt, u_e)
        dt_s = 1j*angular_frequency*s_e

        # return np.dot(dt_s.T, np.dot(elasticity_matrix.T, dt_s))*r
        return np.dot(np.conjugate(s_e).T, np.dot(elasticity_matrix.T, s_e))*r

    return quadratic_quadrature(inner)


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
    frequencies: npt.NDArray

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

    # Internal simulation data
    local_elements: List[LocalElementData]

    def __init__(
            self,
            mesh_data: MeshData,
            material_manager: MaterialManager,
            frequencies: npt.NDArray):
        self.mesh_data = mesh_data
        self.material_manager = material_manager
        self.frequencies = frequencies

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

    def solve_frequency(
            self,
            electrode_elements: npt.NDArray,
            calculate_mech_loss: bool):
        """Run the frequency simulation using the given frequencies.
        If electrode elements are given the charge at those elements is
        calculated.

        Parameters:
            frequencies: Array of frequencies at which the simulation is done.
            electrode_elements: Array of element indices. At those indices
                the charge is calculated, summed up and saved in q."""
        m = self.m
        c = self.c
        k = self.k

        frequencies = self.frequencies
        number_of_nodes = len(self.mesh_data.nodes)
        number_of_elements = len(self.mesh_data.elements)
        u = np.zeros(
            (3*number_of_nodes, len(frequencies)),
            dtype=np.complex128
        )
        q = np.zeros((len(frequencies)), dtype=np.complex128)
        mech_loss = np.zeros(
            (number_of_elements, len(frequencies)),
            dtype=np.complex128
        )

        m, c, k = apply_dirichlet_bc(
            m,
            c,
            k,
            self.dirichlet_nodes
        )

        volumes = calculate_volumes(self.local_elements)
        print(np.sum(volumes))
        self.material_manager.print_material_data(0)
        print(
            f"Starting frequency simulation. There are {len(frequencies)} "
            "frequency steps."
        )
        for frequency_index, frequency in enumerate(frequencies):
            angular_frequency = 2*np.pi*frequency

            a = (
                - angular_frequency**2*m
                + 1j*angular_frequency*c
                + k
            )

            f = self.get_load_vector(
                self.dirichlet_nodes,
                self.dirichlet_values,
                frequency_index
            )

            u[:, frequency_index] = slin.spsolve(a, f)

            if electrode_elements is not None:
                q[frequency_index] = calculate_charge(
                    u[:, frequency_index],
                    self.material_manager,
                    electrode_elements,
                    self.mesh_data.nodes
                )

            # Calculate mech_loss for every element
            for element_index, element in enumerate(self.mesh_data.elements):
                # Get local element data
                local_element = self.local_elements[element_index]
                node_points = local_element.node_points
                jacobian_inverted_t = local_element.jacobian_inverted_t
                jacobian_det = local_element.jacobian_det

                # Get field values
                u_e = np.array([
                    u[2*element[0], frequency_index],
                    u[2*element[0]+1, frequency_index],
                    u[2*element[1], frequency_index],
                    u[2*element[1]+1, frequency_index],
                    u[2*element[2], frequency_index],
                    u[2*element[2]+1, frequency_index]])

                if calculate_mech_loss:
                    mech_loss[element_index, frequency_index] = (
                        loss_integral_scs(
                            node_points,
                            u_e,
                            angular_frequency,
                            jacobian_inverted_t,
                            self.material_manager.get_elasticity_matrix(
                                element_index
                            )
                        )
                        * 2 * np.pi * jacobian_det
                        * self.material_manager.get_alpha_k(element_index)
                        * 1/volumes[element_index]
                        # TODO Actually this must be multiplied with -1
                        * 1/2 * angular_frequency**2
                    )

            if frequency_index % 100 == 0 and frequency_index > 0:
                print(f"Frequency step {frequency_index} finished")

        self.u = u
        self.q = q
        self.mech_loss = mech_loss

    def get_load_vector(
            self,
            dirichlet_nodes: npt.NDArray,
            dirichlet_values: npt.NDArray,
            frequency_index: int) -> npt.NDArray:
        """Calculates the load vector (right hand side) vector for the
        simulation.

        Parameters:
            dirichlet_nodes: Nodes at which the dirichlet value shall be set.
            dirichlet_values: Dirichlet value which is set at the corresponding
                node.

        Returns:
            Right hand side vector for the simulation.
        """
        number_of_nodes = len(self.mesh_data.nodes)

        # Can be initialized to 0 because external load and volume
        # charge density is 0.
        f = np.zeros(3*number_of_nodes, dtype=np.float64)

        for node, value in zip(dirichlet_nodes, dirichlet_values):
            f[node] = value[frequency_index]

        return f
