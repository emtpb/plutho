"""Module for the simulation of freqwuency domain piezoelectric systems."""

# Python standard libraries
from typing import Union
import numpy as np
import numpy.typing as npt
import scipy.sparse as sparse
import scipy.sparse.linalg as slin

# Local libraries
from .solver import FEMSolver
from .helpers import create_node_points, calculate_volumes, \
    mat_apply_dbcs
from .integrals import integral_m, integral_ku, integral_kuv, integral_kve, \
    integral_loss_scs
from ..enums import SolverType
from ..mesh import Mesh


__all__ = [
    "PiezoFreq"
]


class PiezoFreq(FEMSolver):
    """Simulation class for frequency-domain piezoelectric simulations.

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
    mech_loss: Union[npt.NDArray, None]

    def __init__(self, simulation_name: str, mesh: Mesh):
        super().__init__(simulation_name, mesh)

        self.solver_type = SolverType.PiezoFreq
        self.mech_loss = None

    def assemble(self):
        """Assembles the FEM matrices based on the set mesh_data and
        material_data.
        The matrices are stored in self.m, self.c, self.k.
        """
        # TODO Assembly takes to long rework this algorithm
        # Maybe the 2x2 matrix slicing is not very fast
        self.material_manager.initialize_materials()

        # Check if some materials are set
        if len(self.material_manager.materials) == 0:
            raise ValueError("Before assembly some materials must be added.")

        nodes = self.mesh_data.nodes
        elements = self.mesh_data.elements
        element_order = self.mesh_data.element_order
        self.node_points = create_node_points(nodes, elements, element_order)

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
        frequencies: npt.NDArray,
        calculate_mech_loss: bool = False
    ):
        """Run the frequency domain simulation for each given frequency.
        The resulting displacement field is saved in self.u.

        Parameters:
            frequencies: Array of frequencies at which the simulation is done.
            calculate_mech_loss: Set to true if the mechanical losses shall be
                calculated. They are saved in self.mech_loss.
        """
        m = self.m
        c = self.c
        k = self.k

        number_of_nodes = len(self.mesh_data.nodes)
        number_of_elements = len(self.mesh_data.elements)
        element_order = self.mesh_data.element_order
        points_per_element = int(1/2*(element_order+1)*(element_order+2))
        dirichlet_nodes = np.array(self.dirichlet_nodes)
        dirichlet_values = np.array(self.dirichlet_values)

        u = np.zeros(
            (len(frequencies), 3*number_of_nodes),
            dtype=np.complex128
        )
        mech_loss = np.zeros(
            (len(frequencies), number_of_elements),
            dtype=np.complex128
        )

        m, c, k = mat_apply_dbcs(
            m,
            c,
            k,
            dirichlet_nodes
        )

        volumes = calculate_volumes(
            self.node_points,
            self.mesh_data.element_order
        )

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

            # Apply dirichlet bc for f
            f = np.zeros(3*number_of_nodes)
            f[dirichlet_nodes] = dirichlet_values[:, frequency_index]

            u[frequency_index, :] = slin.spsolve(a, f)

            # Calculate mech_loss for every element
            if calculate_mech_loss:
                for element_index, element in enumerate(
                    self.mesh_data.elements
                ):
                    node_points = self.node_points[element_index]

                    # Get field values
                    u_e = np.zeros(2*points_per_element, dtype=np.complex128)
                    for i in range(points_per_element):
                        u_e[2*i] = u[frequency_index, 2*element[i]]
                        u_e[2*i+1] = u[frequency_index, 2*element[i]+1]

                    mech_loss[frequency_index, element_index] = (
                        integral_loss_scs(
                            node_points,
                            u_e,
                            self.material_manager.get_elasticity_matrix(
                                element_index
                            ),
                            element_order
                        )
                        * 2 * np.pi
                        * self.material_manager.get_alpha_k(element_index)
                        * 1/volumes[element_index]
                        # TODO Actually this must be multiplied with -1
                        * 1/2 * angular_frequency**2
                    )

            if frequency_index % 100 == 0 and frequency_index > 0:
                print(f"Frequency step {frequency_index} finished")

        self.u = u

        if calculate_mech_loss:
            self.mech_loss = mech_loss
