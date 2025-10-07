"""Module for the simulation of nonlinear piezoelectric systems"""

# Third party libraries
from typing import Tuple, Union
import numpy as np
import numpy.typing as npt
import scipy
from scipy import sparse, integrate, optimize
from scipy.sparse import linalg

# Local libraries
from .base import assemble, NonlinearType, integral_u_nonlinear_quadratic
from ..base import SimulationData, MeshData, MaterialData, FieldType, \
    create_node_points
from plutho.simulation.piezo_time import calculate_charge
from plutho.materials import MaterialManager
from plutho.mesh.mesh import Mesh


class NonlinearPiezoSimTime:

    def __init__(
        self,
        simulation_name: str,
        mesh: Mesh
    ):
        self.simulation_name = simulation_name

        # Setup simulation data
        self.boundary_conditions = []
        self.dirichlet_nodes = []
        self.dirichlet_values = []

        # Setup mesh and material data
        nodes, elements = mesh.get_mesh_nodes_and_elements()
        self.mesh = mesh
        self.mesh_data = MeshData(nodes, elements, mesh.element_order)
        self.node_points = create_node_points(
            nodes,
            elements,
            mesh.element_order
        )

        self.material_manager = MaterialManager(len(elements))

    def add_material(
        self,
        material_name: str,
        material_data: MaterialData,
        physical_group_name: str
    ):
        """Adds a material to the simulation

        Parameters:
            material_name: Name of the material.
            material_data: Material data.
            physical_group_name: Name of the physical group for which the
                material shall be set. If this is None or "" the material will
                be set for the whole model.
        """
        if physical_group_name is None or physical_group_name == "":
            element_indices = np.arange(len(self.mesh_data.elements))
        else:
            element_indices = self.mesh.get_elements_by_physical_groups(
                [physical_group_name]
            )[physical_group_name]

        self.material_manager.add_material(
            material_name,
            material_data,
            physical_group_name,
            element_indices
        )

    def add_dirichlet_bc(
        self,
        field_type: FieldType,
        physical_group_name: str,
        values: npt.NDArray
    ):
        """Adds a dirichlet boundary condition (dirichlet bc) to the
        simulation. The given values are for each time step. Therefore each
        value is applied to every node from the given physical_group.

        Parameters:
            field_type: The type of field for which the bc shall be set.
            physical_group_name: Name of the physical group for which the
                boundary condition shall be set.
            values: List of values per time step. Value of the bc for each time
                step. The value for one time step is applied to every element
                equally. Length: number_of_time_step.
        """
        # Save boundary condition for serialization
        self.boundary_conditions.append({
            "field_type": field_type.name,
            "physical_group": physical_group_name,
            "values": values.tolist()
        })

        # Apply the given values for all nodes from the given physical_group
        node_indices = self.mesh.get_nodes_by_physical_groups(
            [physical_group_name]
        )[physical_group_name]

        number_of_nodes = len(self.mesh_data.nodes)

        for node_index in node_indices:
            real_index = 0

            # Depending on the variable type and the simulation types
            # the corresponding field variable may be found at different
            # positions of the solution vector
            if field_type is FieldType.PHI:
                real_index = 2*number_of_nodes+node_index
            elif field_type is FieldType.U_R:
                real_index = 2*node_index
            elif field_type is FieldType.U_Z:
                real_index = 2*node_index+1
            else:
                raise ValueError(f"Unknown variable type {field_type}")

            self.dirichlet_nodes.append(real_index)
            self.dirichlet_values.append(values)

    def clear_dirichlet_bcs(self):
        """Resets the dirichlet boundary conditions."""
        self.boundary_conditions = []
        self.dirichlet_nodes = []
        self.dirichlet_values = []

    def set_electrode(self, electrode_name: str):
        self.electrode_elements = self.mesh.get_elements_by_physical_groups(
            [electrode_name]
        )[electrode_name]
        self.electrode_normals = np.tile(
            [0, 1],
            (len(self.electrode_elements), 1)
        )

    def assemble(
        self,
        nonlinear_type: NonlinearType,
        nonlinear_data: Union[float, npt.NDArray]
    ):
        """Redirect to general nonlinear assembly function"""
        self.material_manager.initialize_materials()
        m, c, k = assemble(
            self.mesh_data,
            self.material_manager
        )
        self.nonlinear_type = nonlinear_type
        self.m = m
        self.c = c
        self.k = k
        self.ln = self._assemble_nonlinear_quadratic(
            nonlinear_type,
            nonlinear_data
        )

    def _assemble_nonlinear_quadratic(
        self,
        nonlinear_type: NonlinearType,
        nonlinear_data: Union[float, npt.NDArray]
    ) -> sparse.lil_array:
        if nonlinear_type is NonlinearType.Rayleigh:
            if not isinstance(nonlinear_data, float):
                raise ValueError(
                    "When setting Rayleigh nonlinearity, a float parameter"
                    "must be given."
                )

            return nonlinear_data * self.k

        # Custom nonlinearity with full nonlinear tensor
        if len(nonlinear_data.shape) != 3:
            raise ValueError("A 6x6x6 tensor must be given.")

        # Reduce to axisymmetric 4x4x4 matrix
        # TODO Update this for order 3 or make it n dimensional
        voigt_map = [0, 1, 3, 2]
        nm_reduced = np.zeros(shape=(4, 4, 4))
        for i_new, i_old in enumerate(voigt_map):
            for j_new, j_old in enumerate(voigt_map):
                for k_new, k_old in enumerate(voigt_map):
                    nm_reduced[
                        i_new,
                        j_new,
                        k_new
                    ] = nonlinear_data[i_old, j_old, k_old]

        nodes = self.mesh_data.nodes
        elements = self.mesh_data.elements
        element_order = self.mesh_data.element_order
        number_of_nodes = len(nodes)
        node_points = self.node_points

        lu = sparse.lil_array(
            (4*number_of_nodes**2, 2*number_of_nodes),
            dtype=np.float64
        )

        for element_index, element in enumerate(elements):
            current_node_points = node_points[element_index]

            lu_e = integral_u_nonlinear_quadratic(
                current_node_points,
                nm_reduced,
                element_order
            ) * 2 * np.pi

            for local_i, global_i in enumerate(element):
                for local_j, global_j in enumerate(element):
                    for local_k, global_k in enumerate(element):
                        # lu_e is 6x6x6 but has to be assembled into an
                        # 4*N**2x2*N sparse matrix
                        # The k-th plane is therefore append using the i-th
                        # index
                        # Since lu_e is 6x6x6, 2x2x2 slices are taken out and
                        # used for assembling
                        k_0 = lu_e[
                            2*local_i:2*(local_i+1),
                            2*local_j:2*(local_j+1),
                            2*local_k
                        ]
                        k_1 = lu_e[
                            2*local_i:2*(local_i+1),
                            2*local_j:2*(local_j+1),
                            2*local_k+1
                        ]

                        lu[
                            (
                                2*global_i
                                + 3*number_of_nodes*global_k
                            ):(
                                2*(global_i+1)
                                + 3*number_of_nodes*global_k
                            ),
                            2*global_j:2*(global_j+1),
                        ] += k_0
                        lu[
                            (
                                2*global_i
                                + 3*number_of_nodes*global_k+1
                            ):(
                                2*(global_i+1)
                                + 3*number_of_nodes*global_k+1
                            ),
                            2*global_j:2*(global_j+1),
                        ] += k_1

        ln = sparse.lil_array(
            (9*number_of_nodes**2, 3*number_of_nodes),
            dtype=np.float64
        )

        for k in range(3*number_of_nodes):
            ln[
                k*3*number_of_nodes:k*3*number_of_nodes+2*number_of_nodes,
                :2*number_of_nodes
            ] = lu[
                k*2*number_of_nodes:(k+1)*2*number_of_nodes, :
            ]

        return ln

    def simulate(
        self,
        delta_t: float,
        number_of_time_steps: int,
        gamma: float,
        beta: float,
        tolerance: float = 1e-11,
        max_iter: int = 300,
        u_start: Union[npt.NDArray, None] = None
    ):
        dirichlet_nodes = np.array(self.dirichlet_nodes)
        dirichlet_values = np.array(self.dirichlet_values)

        self.simulation_data = SimulationData(
            delta_t,
            number_of_time_steps,
            gamma,
            beta
        )
        number_of_nodes = len(self.mesh_data.nodes)

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
        m, c, k, ln = self._apply_dirichlet_bc(
            m,
            c,
            k,
            ln,
            dirichlet_nodes
        )

        def nonlinear_product(u, L):
            result = np.zeros(3*number_of_nodes)

            for index in range(3*number_of_nodes):
                L_k = L[index*3*number_of_nodes:(index+1)*3*number_of_nodes, :]

                result[index] = u @ (L_k @ u)

            return result

        if u_start is not None:
            u[:, 0] = u_start

        print("Starting nonlinear time domain simulation")
        for time_index in range(number_of_time_steps-1):
            # Calculate load vector
            f = self._get_load_vector(
                dirichlet_nodes,
                dirichlet_values[:, time_index+1]
            )

            # Residual for the newton iterations
            def residual(next_u, current_u, v, a, f):
                #return (
                #    m@(a1*(next_u-current_u)-a2*v-a3*a)
                #    + c@(a4*(next_u-current_u)+a5*v+a6*a)
                #    + k@next_u+next_u.T@ln@next_u-f
                #)
                return (
                    m@(a1*(next_u-current_u)-a2*v-a3*a)
                    + c@(a4*(next_u-current_u)+a5*v+a6*a)
                    + k@next_u+nonlinear_product(
                        next_u,
                        ln
                    )-f
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
            self.converged = False

            if best_norm > tolerance:
                # Start newton iterations
                for i in range(max_iter):
                    # Calculate tangential stiffness matrix
                    k_tangent = NonlinearPiezoSimTime. \
                        _calculate_tangent_matrix(
                            u_i,
                            k,
                            ln,
                            number_of_nodes
                        )
                    delta_u = linalg.spsolve(
                        (a1*m+a4*c+k_tangent),
                        residual(u_i, current_u, current_v, current_a, f)
                    )
                    u_i_next = u_i - delta_u

                    # Check for convergence
                    norm = scipy.linalg.norm(
                        residual(u_i_next, current_u, current_v, current_a,
                                 f)
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
                        self.converged = True
                        break
                    elif norm < best_norm:
                        best_norm = norm
                        best_u_i = u_i_next.copy()

                    if i % 100 == 0 and i > 0:
                        print("Iteration:", i)

                    # Update for next iteration
                    u_i = u_i_next.copy()
                if not self.converged:
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
            if (self.electrode_elements is not None
                    and self.electrode_elements.shape[0] > 0):
                q[time_index] = calculate_charge(
                    u[:, time_index+1],
                    self.material_manager,
                    self.electrode_elements,
                    self.electrode_normals,
                    self.mesh_data.nodes,
                    self.mesh_data.element_order
                )

        self.u = u
        self.q = q

    def _get_load_vector(
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

    def _apply_dirichlet_bc(
        self,
        m: sparse.lil_array,
        c: sparse.lil_array,
        k: sparse.lil_array,
        ln: sparse.lil_array,
        nodes: npt.NDArray
    ) -> Tuple[
        sparse.csc_array,
        sparse.csc_array,
        sparse.csc_array,
        sparse.csc_array
    ]:
        number_of_nodes = len(self.mesh_data.nodes)
        # TODO Parameters are not really ndarrays
        # Set rows of matrices to 0 and diagonal of K to 1 (at node points)

        # Matrices for u_r component
        m[nodes, :] = 0
        c[nodes, :] = 0
        k[nodes, :] = 0
        k[nodes, nodes] = 1

        for index in range(3*number_of_nodes):
            ln[nodes+index*3*number_of_nodes, :] = 0

        return m.tocsc(), c.tocsc(), k.tocsc(), ln.tocsc()

    @staticmethod
    def _calculate_tangent_matrix_hadamard(
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

    @staticmethod
    def _calculate_tangent_matrix(
        u: npt.NDArray,
        k: sparse.csc_array,
        ln: sparse.csc_array,
        number_of_nodes: int
    ):
        k_tangent = sparse.lil_matrix((
            3*number_of_nodes, 3*number_of_nodes
        ))

        k_tangent += k

        m = np.arange(3*number_of_nodes)

        # TODO This calculation is very slow
        for i in range(3*number_of_nodes):
            l_k = ln[3*number_of_nodes*i:3*number_of_nodes*(i+1), :]
            l_i = ln[3*number_of_nodes*m+i, :]

            k_tangent += (l_k*u[i]).T
            k_tangent += (l_i*u[i]).T

        return k_tangent
