"""Base module for various functions neede for the nonlinear simulation."""

# Python standard libraries
from typing import Tuple, Union

# Third party libraries
import numpy as np
import numpy.typing as npt
from scipy import sparse

# Local libraries
from ...mesh.mesh import MeshData
from ..integrals import local_to_global_coordinates, b_operator_global, \
    integral_m, integral_ku, integral_kuv, integral_kve, \
    quadratic_quadrature, gradient_local_shape_functions_2d
from ..helpers import create_node_points
from plutho.materials import MaterialManager
from ...enums import NonlinearType


__all__ = [
    "Nonlinearity"
]


#
# -------- Functions --------
#

def assemble(
    mesh_data: MeshData,
    material_manager: MaterialManager
) -> Tuple[
    sparse.lil_array,
    sparse.lil_array,
    sparse.lil_array
]:
    """Creates m, c, k FEM matrices based on the given mesh and material
    data

    Parameters:
        mesh_data: MeshData object containing the mesh nodes and elements.
        material_manager: MaterialManager object containing the material data

    Returns:
        The assembled mass matrix m, damping matrix c and stiffness matrix k.
    """
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

    # Calculate block matrices
    zeros1x1 = np.zeros((number_of_nodes, number_of_nodes))
    zeros2x1 = np.zeros((2*number_of_nodes, number_of_nodes))
    zeros1x2 = np.zeros((number_of_nodes, 2*number_of_nodes))

    m = sparse.block_array([
        [mu, zeros2x1],
        [zeros1x2, zeros1x1],
    ]).tolil()
    c = sparse.block_array([
        [cu, zeros2x1],
        [zeros1x2, zeros1x1],
    ]).tolil()
    k = sparse.block_array([
        [ku, kuv],
        [kuv.T, -1*kv]
    ]).tolil()

    return m, c, k


def integral_u_nonlinear_quadratic(
    node_points: npt.NDArray,
    nonlinear_elasticity_tensor: npt.NDArray,
    element_order: int
):
    """Calculates the nonlinear quadratic integral over the given node points.

    Parameters:
        node_points: Points of the current element.
        nonlinear_elasticity_tensor: Tensor for a quadratic nonlinear material
            model.
        element_order: Order of the element.
        """
    def inner(s: float, t: float):
        dn = gradient_local_shape_functions_2d(s, t, element_order)
        jacobian = np.dot(node_points, dn.T)
        jacobian_inverted_t = np.linalg.inv(jacobian).T
        jacobian_det = np.linalg.det(jacobian)

        b_op = b_operator_global(
            node_points,
            jacobian_inverted_t,
            s,
            t,
            element_order
        )
        r = local_to_global_coordinates(node_points, s, t, element_order)[0]

        B_TC = np.einsum("im,mjk->ijk", b_op.T, nonlinear_elasticity_tensor)
        B_TCB = np.einsum("ink,nj->ijk", B_TC, b_op)
        B_TCBB = np.einsum("ijp,pk->ijk", B_TCB, b_op)

        return B_TCBB * r * jacobian_det

    return quadratic_quadrature(inner, element_order)

#
# -------- Classes --------
#
class Nonlinearity:
    """Class for handling the different nonlinearity types. Adds an additional
    layer of abstraction so the solver can be the same for all nonlinear
    types.

    Attributes:
        mesh_data: Mesh data object.
        node_points: List of node points for each element.
        nonlinear_type: The type of the nonlinearity.
        zeta: Nonlinear data parameter.
        custom_data: Nonlinear data parameter.
    """
    mesh_data: Union[MeshData, None]
    node_points: Union[npt.NDArray, None]

    # Nonlinear data
    nonlinear_type: Union[NonlinearType, None]
    zeta: Union[float, None]
    custom_data: Union[npt.NDArray, None]

    def __init__(self):
        self.nonlinear_type = None
        self.mesh_data = None
        self.node_points = None

    def set_mesh_data(self, mesh_data: MeshData, node_points: npt.NDArray):
        """Sets the mesh data.

        Parameters:
            mesh_data: MeshData to set.
            node_points: List of node points for each element.
        """
        self.mesh_data = mesh_data
        self.node_points = node_points

    def set_quadratic_rayleigh(self, zeta: float):
        """Sets a quadratic rayleigh nonlinearity which only needs one
        parameter.

        Parameters:
            zeta: Nonlinearity parameter.
        """
        self.nonlinear_type = NonlinearType.QuadraticRayleigh
        self.zeta = zeta

    def set_quadratic_custom(self, nonlinear_data: npt.NDArray):
        """Sets a quadratic nonlinearity with a custom material tensor.

        Parameters:
            nonlinear_data: Nonlinear material tensor (6x6x6).
        """
        self.nonlinear_type = NonlinearType.QuadraticCustom

        # Reduce to axisymmetric 4x4x4 matrix
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

        self.custom_data = nm_reduced

    def set_cubic_rayleigh(self, zeta: float):
        """Sets a cubic rayleigh nonlinearity which only needs one parameter.

        Parameters:
            zeta: Nonlinearity parameter.
        """
        self.nonlinear_type = NonlinearType.CubicRayleigh
        self.zeta = zeta

    def set_cubic_custom(self, nonlinear_data: npt.NDArray):
        """Sets a cubic nonlinearity with a custom nonlinear material tensor

        Parameters:
            nonlinear_data: Nonlinear material tensor (6x6x6x6).
        """
        self.nonlinear_type = NonlinearType.CubicCustom

        # Reduce to axisymmetric 4x4x4x4 matrix
        voigt_map = [0, 1, 3, 2]
        nm_reduced = np.zeros(shape=(4, 4, 4, 4))
        for i_new, i_old in enumerate(voigt_map):
            for j_new, j_old in enumerate(voigt_map):
                for k_new, k_old in enumerate(voigt_map):
                    for l_new, l_old in enumerate(voigt_map):
                        nm_reduced[
                            i_new,
                            j_new,
                            k_new,
                            l_new
                        ] = nonlinear_data[i_old, j_old, k_old, l_old]

        self.custom_data = nm_reduced

    def evaluate_force_vector(
        self,
        u: npt.NDArray,
    ) -> npt.NDArray:
        """Evaluates the nonlinear force vector based on the previously set
        nonlinearity and the given displacement field u and the FEM matrices.

        Parameters:
            u: Current displacement field.

        Returns:
            Vector of nonlinear forces.
        """
        if self.nonlinear_type is None:
            raise ValueError("Cannot evaluate force vector, since no \
                nonlinear type is set")

        match self.nonlinear_type:
            case NonlinearType.QuadraticRayleigh:
                return self.ln@(u**2)
            case NonlinearType.QuadraticCustom:
                # Use jit compiler --> wimp? or inline c?
                number_of_nodes = len(self.mesh_data.nodes)
                res = np.zeros(3*number_of_nodes)

                for index in range(3*number_of_nodes):
                    L_k = self.ln[
                        index*3*number_of_nodes:(index+1)*3*number_of_nodes,
                        :
                    ]

                    res[index] = u @ (L_k @ u)

                return res
            case NonlinearType.CubicRayleigh:
                return self.ln@(u**3)
            case NonlinearType.CubicCustom:
                raise NotImplementedError()

    def evaluate_jacobian(
        self,
        u: npt.NDArray,
        k: sparse.csc_array
    ) -> sparse.csc_array:
        """Evaluates the jacobian of the nonlinear force vector based on the
        current displacement u and the FEM matrices.

        Parameters:
            u: Current displacement vector.
            k: FEM stiffness matrix.

        Returns:
            Jacobian for nonlinearity.
        """
        if self.nonlinear_type is None:
            raise ValueError("Cannot evaluate jacobian, since no \
                nonlinear type is set")

        match self.nonlinear_type:
            case NonlinearType.QuadraticRayleigh:
                return 2*self.ln@sparse.diags_array(u)
            case NonlinearType.QuadraticCustom:
                """
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
                """
                raise NotImplementedError()
            case NonlinearType.CubicRayleigh:
                return 3*self.ln@sparse.diags_array(u**2)
            case NonlinearType.CubicCustom:
                raise NotImplementedError()

    def apply_dirichlet_bc(self, dirichlet_nodes: npt.NDArray):
        """Applies the dirichlet boundary conditions on the nonlinear matrix.
        """
        if self.nonlinear_type is None:
            raise ValueError("Cannot apply dirichlet boundary \
                conditions, since no nonlinear type is set")

        # TODO Is a different handling for nonlinear types necessary?
        match self.nonlinear_type:
            case NonlinearType.QuadraticRayleigh | \
                    NonlinearType.CubicRayleigh:
                self.ln[dirichlet_nodes, :] = 0
            case NonlinearType.QuadraticCustom:
                raise NotImplementedError()
            case NonlinearType.CubicCustom:
                raise NotImplementedError()

    def assemble(
        self,
        k: sparse.lil_array
    ):
        """Assembles the nonlinar FEM matrix. It is saved in self.ln.

        Parameters:
            k: FEM stiffness matrix.
        """
        number_of_nodes = len(self.mesh_data.nodes)

        if self.nonlinear_type is None:
            raise ValueError("Cannot assemble matrices, since no \
                nonlinear type is set")

        match self.nonlinear_type:
            case NonlinearType.QuadraticRayleigh | \
                    NonlinearType.CubicRayleigh:
                self.ln = sparse.lil_array(
                        (3*number_of_nodes, 3*number_of_nodes)
                )
                # Only take the part for the mechanical field
                self.ln[:2*number_of_nodes, :2*number_of_nodes] = \
                    k[:2*number_of_nodes, :2*number_of_nodes]*self.zeta
            case NonlinearType.QuadraticCustom:
                self._assemble_quadratic_custom()
            case NonlinearType.CubicCustom:
                self._assemble_cubic_custom()

    def _assemble_quadratic_custom(self):
        """Local function. Assembles the nonlinear FEM matrix for a quadratic
        custom nonlinearity type.
        """
        if self.mesh_data is None:
            raise ValueError("Mesh data is not set")

        nodes = self.mesh_data.nodes
        elements = self.mesh_data.elements
        element_order = self.mesh_data.element_order
        number_of_nodes = len(nodes)
        node_points = create_node_points(
            nodes,
            elements,
            element_order
        )

        lu = sparse.lil_array(
            (4*number_of_nodes**2, 2*number_of_nodes),
            dtype=np.float64
        )

        for element_index, element in enumerate(elements):
            current_node_points = node_points[element_index]

            lu_e = integral_u_nonlinear_quadratic(
                current_node_points,
                self.custom_data,
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

        self.ln = ln


    def _assemble_cubic_custom(self):
        pass
