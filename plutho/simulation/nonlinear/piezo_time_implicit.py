"""Module for the simulation of nonlinear piezoelectric systems using an
implicit formulation"""

# Third party libraries
from typing import Tuple, Union
import numpy as np
import numpy.typing as npt
import scipy
from scipy import sparse, integrate, optimize
from scipy.sparse import linalg

# Local libraries
from .base import assemble, NonlinearType, integral_u_nonlinear_quadratic, \
    integral_m, integral_ku, integral_kuv, integral_kve
from ..base import SimulationData, MeshData, MaterialData, FieldType, \
    create_node_points
from plutho.simulation.piezo_time import calculate_charge
from plutho.materials import MaterialManager
from plutho.mesh.mesh import Mesh

class NonlinearPiezoSimTimeImplicit:
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

    def assemble_linear(
        self,
        mesh_data: MeshData,
        material_manager: MaterialManager
    ) -> Tuple[
        sparse.lil_array,
        sparse.lil_array,
        sparse.lil_array
    ]:
        # TODO Assembly takes to long rework this algorithm
        # Maybe the 2x2 matrix slicing is not very fast
        # TODO Change to lil_array
        nodes = mesh_data.nodes
        elements = mesh_data.elements
        element_order = mesh_data.element_order
        node_points = create_node_points(nodes, elements, element_order)

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
        kv = sparse.lil_matrix(
            (number_of_nodes, number_of_nodes),
            dtype=np.float64
        )

        for element_index, element in enumerate(elements):
            current_node_points = node_points[element_index]

            # TODO Check if its necessary to calculate all integrals
            # --> Dirichlet nodes could be leaved out?
            # Multiply with jac_det because its integrated with respect to
            # local coordinates.
            # Mutiply with 2*pi because theta is integrated from 0 to 2*pi
            mu_e = (
                material_manager.get_density(element_index)
                * integral_m(current_node_points, element_order)
                * 2 * np.pi
            )
            ku_e = (
                integral_ku(
                    current_node_points,
                    material_manager.get_elasticity_matrix(element_index),
                    element_order
                ) * 2 * np.pi
            )
            kuv_e = (
                integral_kuv(
                    current_node_points,
                    material_manager.get_piezo_matrix(element_index),
                    element_order
                ) * 2 * np.pi
            )
            kve_e = (
                integral_kve(
                    current_node_points,
                    material_manager.get_permittivity_matrix(
                        element_index
                    ),
                    element_order
                ) * 2 * np.pi
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
                    kv[global_p, global_q] += kve_e[local_p, local_q]

        # Calculate damping matrix
        # Currently in the simulations only one material is used
        # TODO Add algorithm to calculate cu for every element on its own
        # in order to use multiple materials
        cu = (
            material_manager.get_alpha_m(0) * mu
            + material_manager.get_alpha_k(0) * ku
        )

        return mu, ku, kuv, kv, cu

    def assemble(
        self,
        nonlinear_type: NonlinearType,
        nonlinear_data: Union[float, npt.NDArray]
    ):
        """Redirect to general nonlinear assembly function"""
        self.material_manager.initialize_materials()
        mu, ku, kuv, kv, cu = self.assemble_linear(
            self.mesh_data,
            self.material_manager
        )
        self.nonlinear_type = nonlinear_type
        self.ln = self._assemble_nonlinear_quadratic(
            nonlinear_type,
            nonlinear_data
        )

        self.mu = mu
        self.ku = ku
        self.kuv = kuv
        self.kv = kv
        self.cu = cu

    def _assemble_nonlinear_quadratic(
        self,
        nonlinear_type: NonlinearType,
        nonlinear_data: Union[float, npt.NDArray]
    ) -> sparse.lil_array:
        if nonlinear_type is NonlinearType.Rayleigh:
            if not isinstance(nonlinear_data, float):
                raise ValueError(
                    "When setting Rayleigh nonlinearity, a float parameter must"
                    "be given."
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
            delta_t,
            number_of_time_steps
        ):
            dirichlet_nodes = np.array(self.dirichlet_nodes)
            dirichlet_values = np.array(self.dirichlet_values)

            number_of_nodes = len(self.mesh_data.nodes)

            mu = self.mu.copy()
            ku = self.ku.copy()
            kuv = self.kuv.copy()
            kv = self.kv.coppy()

            alpha_k = self.material_manager.get_alpha_k(0)
            alpha_m = self.material_manager.get_alpha_m(0)

            c = alpha_m*mu+alpha_k*ku
            ln = self.ln.copy()

            # Apply dirichlet bc to matrices
            mu, ku, kuv_u, kuv_phi, kv = _apply_dirichlet_bc_implicit(
                mu,
                ku,
                kuv,
                kv,
                dirichlet_nodes
            )

            # Init arrays
            # Displacement u which is calculated
            y_0 = np.zeros(6*number_of_nodes)

            time_interval = (0, number_of_time_steps*delta_t)
            time_steps = np.arange(number_of_time_steps)*delta_t

            def nonlinear_product(u, L):
                result = np.zeros(3*number_of_nodes)

                for index in range(3*number_of_nodes):
                    L_k = L[index*3*number_of_nodes:(index+1)*3*number_of_nodes, :]

                    result[index] = u @ (L_k @ u)

                return result

            def find_nearest(array, value):
                return (np.abs(array - value)).argmin()

            def rhs(t, y):

                current_u = y[:3*number_of_nodes]
                current_v = y[3*number_of_nodes:]

                time_index = find_nearest(time_steps, t)

                f = self._get_load_vector(
                    dirichlet_nodes,
                    dirichlet_values[:, time_index]
                )
                eq = f-c@current_v-k@current_u-nonlinear_product(current_u, ln)

                a = linalg.spsolve(m, eq)

                return np.concatenate((current_v, a))

            sol = integrate.solve_ivp(
                rhs,
                time_interval,
                y_0,
                t_eval=time_steps,
                method="RK45"
            )

            if sol.status != 0:
                print("Something went wrong while simulating")
                print(sol.message)

            print(sol)

            self.u = sol.y[:3*number_of_nodes, :]

    def _apply_dirichlet_bc_implicit(
        self,
        mu: sparse.lil_array,
        ku: sparse.lil_array,
        kuv: sparse.lil_array,
        kv: sparse.lil_array,
        cu: sparse.lil_array,
        nodes: npt.NDArray
    ) -> Tuple[
        sparse.csc_array,
        sparse.csc_array,
        sparse.csc_array,
        sparse.csc_array,
        sparse.csc_array
    ]:
        number_of_nodes = len(self.mesh_data.nodes)
        nodes_u = []
        nodes_phi = []
        for node in nodes:
            if node > 2*number_of_nodes:
                nodes_phi.append(node)
            else:
                nodes_u.append(nodes)

        kuv_u = kuv.copy()
        kuv_phi = kuv.copy()

        # For u field matrices
        mu[nodes_u, :] = 0
        kuv_u[nodes_u, :] = 0
        ku[nodes, :] = 0
        ku[nodes, nodes] = 1

        # For phi field matrices
        kuv_phi[nodes, :] = 0
        kv[nodes, :] = 0
        kv[nodes, nodes] = 1

        return mu.tocsc(), ku.tocsc(), kuv_u.tocsc(), kuv_phi.tocsc(), \
            kv.tocsc()
