"""Module for the simulation of nonlinear piezoelectric systems"""

# Third party libraries
from typing import List
import numpy as np
import numpy.typing as npt
import scipy
from scipy import sparse
from scipy.sparse import linalg

# Local libraries
from .base import assemble, NonlinearType
from ..base import SimulationData, MeshData, gradient_local_shape_functions, \
    LocalElementData
from plutho.simulation.piezo_time import charge_integral_u, \
    charge_integral_v
from plutho.materials import MaterialManager


class NonlinearPiezoSimTime:
    """Simulates eletro-mechanical fields for the given mesh, material and
    simulation information. The simulation is using a nonlinear stiffness
    matrix which is used for a quadratic term in the hooke law.

    Attributes:
        mesh_data: Contains the mesh information.
        material_manager: Organizes material parameters. Can use materials
            with different temperatures.
        simulation_data: Contains settings for the simulation.
        dirichlet_nodes_u: Nodes at which a dirichlet bc shall be set for the
            mechanical field.
        dirichlet_values_u: Values which are set at the node from the nodes
            list for the mechanical field.
        dirichlet_nodes_phi: Nodes at which a dirichlet bc shall be set for
            the electrical field.
        dirichlet_values_phi: Values which are set at the node from the nodes
            list for the electrical field.
        m: FEM mass matrix for the mechanical field.
        k: FEM stiffness matrix for the mechanical field.
        c: FEM damping matrix for the mechanical field.
        ln: FEM stiffness matrix for the nonlinear part of the
            mechanical field.
        u: Mechanical field vector u(element, t).
        q: Charge q(t).
        local_elements: List of element data for the local triangles.

    Parameters:
        mesh_data: MeshData object containing the mesh.
        material_manager: MaterialManager object containing the set materials
            for each region.
        simulation_data:SimulationData object containing information on the
            time domain simulation.
    """

    # Simulation parameters
    mesh_data: MeshData
    material_manager: MaterialManager
    simulation_data: SimulationData

    # Dirichlet boundary condition
    dirichlet_nodes: npt.NDArray
    dirichlet_values: npt.NDArray

    # Resulting fields
    u: npt.NDArray
    q: npt.NDArray

    # Internal simulation data
    local_elements: List[LocalElementData]

    # FEM matrices
    m: sparse.lil_array
    c: sparse.lil_array
    ln: sparse.lil_array
    k: sparse.lil_array

    def __init__(
        self,
        mesh_data: MeshData,
        material_manager: MaterialManager,
        simulation_data: SimulationData
    ):
        self.mesh_data = mesh_data
        self.material_manager = material_manager
        self.simulation_data = simulation_data

        self.dirichlet_nodes = np.array([])
        self.dirichlet_values = np.array([])

    def assemble(self, nonlinear_type: NonlinearType, **kwargs):
        """Redirect to general nonlinear assembly function"""
        m, c, k, ln = assemble(
            self.mesh_data,
            self.material_manager,
            nonlinear_type,
            **kwargs
        )
        self.m = m
        self.c = c
        self.k = k
        self.ln = ln

    def solve_time_implicit(
        self,
        tolerance: float = 1e-11,
        max_iter: int = 300,
        electrode_elements: npt.NDArray = np.array([]),
        electrode_normals: npt.NDArray = np.array([])
    ):
        number_of_time_steps = self.simulation_data.number_of_time_steps
        delta_t = self.simulation_data.delta_t
        number_of_nodes = len(self.mesh_data.nodes)
        beta = self.simulation_data.beta
        gamma = self.simulation_data.gamma

        m = self.m.copy()
        c = self.c.copy()
        k = self.k.copy()
        ln = self.ln.copy()

        # Time integration constants
        a1 = 1/(beta*(delta_t**2))
        a2 = 1/(beta*delta_t)
        a3 = (1-2*beta)/(2*beta)
        a4 = gamma/(beta*delta_t)
        a5 = 1-gamma/beta
        a6 = (1-gamma/(2*beta))*delta_t

        # Init arrays
        # Displacement u which is calculated
        u = np.zeros(
            (3*number_of_nodes, number_of_time_steps),
            dtype=np.float64
        )
        # Displacement u derived after t (du/dt)
        v = np.zeros(
            (3*number_of_nodes, number_of_time_steps),
            dtype=np.float64
        )
        # v derived after u (d^2u/dt^2)
        a = np.zeros(
            (3*number_of_nodes, number_of_time_steps),
            dtype=np.float64
        )
        # Charge
        q = np.zeros(number_of_time_steps, dtype=np.float64)

        # Apply dirichlet bc to matrices
        m, c, k, ln = NonlinearPiezoSimTime.apply_dirichlet_bc(
            m,
            c,
            k,
            ln,
            self.dirichlet_nodes
        )

        print("Starting nonlinear time domain simulation")
        for time_index in range(number_of_time_steps-1):
            # Calculate load vector
            f = self.get_load_vector(
                self.dirichlet_nodes,
                self.dirichlet_values[:, time_index+1]
            )

            # Residual for the newton iterations
            def residual(next_u, current_u, v, a, f):
                return (
                    m@(a1*(next_u-current_u)-a2*v-a3*a)
                    + c@(a4*(next_u-current_u)+a5*v+a6*a)
                    + k@next_u+ln@np.square(next_u)-f
                )

            # Values of the current time step
            current_u = u[:, time_index]
            current_v = v[:, time_index]
            current_a = a[:, time_index]

            # Current iteration value of u of the next time step
            # as a start value this is set to the last converged u of the last
            # time step
            u_i = current_u.copy()

            # Best values during newton are saved
            best_u_i = u_i.copy()
            best_norm = scipy.linalg.norm(residual(
                u_i,
                current_u,
                current_v,
                current_a,
                f
            ))

            # Placeholder for result of newton
            next_u = np.zeros(3*number_of_nodes)
            converged = False

            if best_norm > tolerance:
                # Start newton iterations
                for i in range(max_iter):
                    # Calculate tangential stiffness matrix
                    k_tangent = NonlinearPiezoSimTime. \
                        calculate_tangent_matrix_hadamard(
                            u_i,
                            k,
                            ln
                        )
                    delta_u = linalg.spsolve(
                        (a1*m+a4*c+k_tangent),
                        residual(u_i, current_u, current_v, current_a, f)
                    )
                    u_i_next = u_i - delta_u

                    # Check for convergence
                    norm = scipy.linalg.norm(
                        residual(u_i_next, current_u, current_v, current_a, f)
                    )
                    # print(
                    #     f"Time step: {time_index}, iteration: {i}, "
                    #     f"norm: {norm}"
                    # )
                    if norm < tolerance:
                        print(
                            f"Newton converged at time step {time_index} "
                            f"after {i+1} iteration(s)"
                        )
                        # print(u_i_next)
                        next_u = u_i_next.copy()
                        converged = True
                        break
                    elif norm < best_norm:
                        best_norm = norm
                        best_u_i = u_i_next.copy()

                    if i % 100 == 0 and i > 0:
                        print("Iteration:", i)

                    # Update for next iteration
                    u_i = u_i_next.copy()
                if not converged:
                    print(
                        "Newton did not converge.. Choosing best value: "
                        f"{best_norm}"
                    )
                    next_u = best_u_i.copy()
            else:
                print("Start value norm already below tolerance")
                next_u = best_u_i.copy()

            # Calculate next v and a
            a[:, time_index+1] = (
                a1*(next_u-current_u)
                - a2*current_v
                - a3*current_a
            )
            v[:, time_index+1] = (
                a4*(next_u-current_u)
                + a5*current_v
                + a6*current_a
            )

            # Set u array value
            u[:, time_index+1] = next_u

            # Calculate charge
            if (electrode_elements is not None
                    and electrode_elements.shape[0] > 0):
                q[time_index] = NonlinearPiezoSimTime.calculate_charge(
                    u[:2*number_of_nodes, time_index+1],
                    u[2*number_of_nodes:, time_index+1],
                    self.material_manager,
                    electrode_elements,
                    electrode_normals,
                    self.mesh_data.nodes
                )

        self.u = u
        self.q = q


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

    @staticmethod
    def calculate_charge(
        u: npt.NDArray,
        phi: npt.NDArray,
        material_manager: MaterialManager,
        electrode_elements: npt.NDArray,
        element_normals: npt.NDArray,
        nodes: npt.NDArray
    ) -> float:
        """Calculates the charge using u and phi on the given electrode
        elements.

        Parameters:
            u: Mechanical displacement field.
            phi: Electrical potential field.
            material_manager: Contains the material data.
            electrode_elements: Indices of the element on which the charge
                is calculated.
            nodes: All nodes from the mesh.

        Returns:
            The charge on the given elements summed up."""
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
            jacobian_det = np.sqrt(
                np.square(nodes[element[1]][0]-nodes[element[0]][0])
                + np.square(nodes[element[1]][1]-nodes[element[0]][1])
            )

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

            # Now take the component normal to the line (outer direction)
            q_u_normal = np.dot(q_u, element_normals[element_index])
            q_v_normal = np.dot(q_v, element_normals[element_index])

            q += q_u_normal - q_v_normal

        if np.isnan(q):
            print("Calculated charge is nan")

        return q

    @staticmethod
    def apply_dirichlet_bc(
        m: sparse.lil_array,
        c: sparse.lil_array,
        k: sparse.lil_array,
        ln: sparse.lil_array,
        nodes: npt.NDArray
    ):
        # TODO Parameters are not really ndarrays
        # Set rows of matrices to 0 and diagonal of K to 1 (at node points)

        # Matrices for u_r component
        for node in nodes:
            # Set rows to 0
            m[node, :] = 0
            c[node, :] = 0
            k[node, :] = 0
            ln[node, :] = 0

            # Set diagonal values to 1
            k[node, node] = 1

        return m.tocsc(), c.tocsc(), k.tocsc(), ln.tocsc()

    @staticmethod
    def calculate_tangent_matrix_hadamard(
        u: npt.NDArray,
        k: sparse.csc_array,
        ln: sparse.csc_array
    ):
        # TODO Duplicate function in piezo_stationary.py
        """Calculates the tangent matrix based on an analytically
        expression.

        Parameters:
            u: Current mechanical displacement.
            k: FEM stiffness matrix.
            ln: FEM nonlinear stiffness matrix.
        """
        k_tangent = k+2*ln@sparse.diags_array(u)

        return k_tangent
