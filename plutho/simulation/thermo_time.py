"""Extra module for the simulation of heat conduction over time."""

# Python standard libraries
from typing import List

# Third party libraries
import numpy as np
import numpy.typing as npt
import scipy.sparse.linalg as slin
from scipy import sparse

# Local libraries
from .base import SimulationData, MeshData, \
    local_shape_functions_2d, gradient_local_shape_functions_2d, \
    local_to_global_coordinates, integral_m, \
    quadratic_quadrature, line_quadrature, create_node_points, \
    apply_dirichlet_bc, get_avg_temp_field_per_element
from ..materials import MaterialManager


def integral_heat_flux(
    node_points: npt.NDArray,
    heat_flux: npt.NDArray,
    element_order: int
):
    """Integrates the heat flux using the shape functions.

    Parameters:
        node_points: List of node points [[x1, x2, ..], [y1, y2, ..]].
        heat_flux: Heat flux at the points.
        element_order: Order of the shape functions.

    Returns:
        npt.NDArray heat flux integral on each point.
    """
    def inner(s):
        n = local_shape_functions_2d(s, 0, element_order)
        r = local_to_global_coordinates(node_points, s, 0, element_order)[0]

        return n*heat_flux*r

    return line_quadrature(inner, element_order)


def integral_ktheta(
    node_points: npt.NDArray,
    element_order: int
):
    """Calculates the Ktheta integral.

    Parameters:
        node_points: List of node points [[x1, x2, ..], [y1, y2, ..]].
        element_order: Order of the shape functions.

    Returns:
        npt.NDArray: 3x3 Ktheta matrix for the given element.
    """
    def inner(s, t):
        dn = gradient_local_shape_functions_2d(s, t, element_order)
        jacobian = np.dot(node_points, dn.T)
        jacobian_inverted_t = np.linalg.inv(jacobian).T
        jacobian_det = np.linalg.det(jacobian)

        global_dn = np.dot(jacobian_inverted_t, dn)
        r = local_to_global_coordinates(node_points, s, t, element_order)[0]

        return np.dot(global_dn.T, global_dn)*r*jacobian_det

    return quadratic_quadrature(inner, element_order)


def integral_theta_load(
    node_points: npt.NDArray,
    mech_loss: float,
    element_order: int
):
    """Returns the load value for the temperature field (f) for the specific
    element.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        point_loss: Loss power on each node (heat source).
        element_order: Order of the shape functions.

    Returns:
        npt.NDArray: f vector value at the specific ndoe
    """
    def inner(s, t):
        dn = gradient_local_shape_functions_2d(s, t, element_order)
        jacobian = np.dot(node_points, dn.T)
        jacobian_det = np.linalg.det(jacobian)

        n = local_shape_functions_2d(s, t)
        r = local_to_global_coordinates(node_points, s, t, element_order)[0]

        return n*mech_loss*r*jacobian_det

    return quadratic_quadrature(inner, element_order)


class ThermoSimTime:
    """Simulates a heat conduction problem for the given mesh, material and
    simulation information.

    Attributes:
        mesh_data: Contains the mesh information.
        material_manager: Organizes the material parameters.
        simulation_data: Contains the simulation data.
        theta: Temperature field defined for every node.
        f: Load vector for the temperature field contains the information
            from the given losses.
        """
    # Simulation settings
    mesh_data: MeshData
    material_manager: MaterialManager
    simulation_data: SimulationData

    # FEM Vectors
    theta: npt.NDArray
    f: npt.NDArray

    # Convective bc
    enable_convection_bc: bool
    convective_b_e: List[int]
    convective_alpha: float
    convective_outer_temp: float

    # Internal simulation data
    node_points: npt.NDArray

    # Dirichlet data
    dirichlet_nodes: npt.NDArray
    dirichlet_values: npt.NDArray

    def __init__(
            self,
            mesh_data: MeshData,
            material_manager: MaterialManager,
            simulation_data: SimulationData):
        self.mesh_data = mesh_data
        self.material_manager = material_manager
        self.simulation_data = simulation_data
        self.f = np.zeros(
            (
                len(self.mesh_data.nodes),
                self.simulation_data.number_of_time_steps
            )
        )
        self.enable_convection_bc = False

    def assemble(self):
        """Assembles the FEM matrices based on the set mesh_data and
        material_data.
        The matrices are stored in self.c, self.k.
        """
        # TODO Assembly takes long rework this algorithm?
        # Maybe the 2x2 matrix slicing is not very fast
        nodes = self.mesh_data.nodes
        elements = self.mesh_data.elements
        element_order = self.mesh_data.element_order
        self.node_points = create_node_points(nodes, elements, element_order)

        number_of_nodes = len(nodes)
        c = sparse.lil_matrix(
            (number_of_nodes, number_of_nodes),
            dtype=np.float64)
        k = sparse.lil_matrix(
            (number_of_nodes, number_of_nodes),
            dtype=np.float64)

        for element_index, element in enumerate(self.mesh_data.elements):
            node_points = self.node_points[element_index]

            ctheta_e = (
                integral_m(node_points, element_order)
                * self.material_manager.get_density(element)
                * self.material_manager.get_heat_capacity(element)
                * 2 * np.pi
            )
            ktheta_e = (
                integral_ktheta(
                    node_points,
                    element_order
                )
                * self.material_manager.get_thermal_conductivity(element)
                * 2 * np.pi
            )

            # Now assemble all element matrices
            for local_p, global_p in enumerate(element):
                for local_q, global_q in enumerate(element):
                    c[global_p, global_q] += ctheta_e[local_p, local_q]
                    k[global_p, global_q] += ktheta_e[local_p, local_q]

        self.c = c.tolil()
        self.k = k.tolil()

    def set_constant_volume_heat_source(
        self,
        mech_loss_density: npt.NDArray,
        number_of_time_steps: float
    ):
        """Sets the excitation for the heat conduction simulation.
        The mech_loss_density are set for every time step.

        Parameters:
            mech_loss_density: The mechanical losses for each element. They
                will be applied for every time step.
            number_of_time_steps: Total number of time steps
        """
        nodes = self.mesh_data.nodes
        element_order = self.mesh_data.element_order
        number_of_nodes = len(nodes)
        points_per_element = int(1/2*(element_order+1)*(element_order+2))

        f = np.zeros(number_of_nodes)
        for element_index, element in enumerate(self.mesh_data.elements):
            node_points = np.zeros(shape=(2, points_per_element))
            for node_index in range(points_per_element):
                node_points[:, node_index] = [
                    nodes[element[node_index]][0],
                    nodes[element[node_index]][1]
                ]

            f_theta_e = integral_theta_load(
                node_points,
                mech_loss_density[element_index],
                element_order
            ) * 2 * np.pi

            for local_p, global_p in enumerate(element):
                f[global_p] += f_theta_e[local_p]

        # Repeat it for n time steps
        self.f += np.tile(f.reshape(-1, 1), (1, number_of_time_steps))

    def _calculate_convection_bc(
        self,
        nodes: npt.NDArray,
        boundary_elements: npt.NDArray,
        theta: npt.NDArray,
        alpha: float,
        outer_temperature: float,
        element_order: int
    ) -> npt.NDArray:
        """Calculates the load vector for the convection boundary condition.
        The condition is calculated using the temeprature at the given
        boundary.
        The heat transfer coefficient alpha and the temeprature outside of the
        model.

        Parameters:
            nodes: All nodes in the mesh.
            elements: Boundary elements at which the convection boundary is
                added.
            theta: Current temperature field at all nodes.
            alpha: Heat transfer coefficient.
            outer_temperature: Temperature outside of the model.
        """
        points_per_element = int(1/2*(element_order+1)*(element_order+2))

        f = np.zeros(len(nodes))
        for element_index, element in enumerate(boundary_elements):
            node_points = np.zeros(shape=(2, points_per_element))
            for node_index in range(points_per_element):
                node_points[element_index, :, node_index] = [
                    nodes[element[node_index]][0],
                    nodes[element[node_index]][1]
                ]
            # TODO Only works for element_order > 1
            jacobian_det = np.sqrt(
                np.square(nodes[element[1]][0]-nodes[element[0]][0])
                + np.square(nodes[element[1]][1]-nodes[element[0]][1])
            )

            # Temperature at boundary
            boundary_nodes = []
            match element_order:
                case 1:
                    boundary_nodes = np.array([0, 1])
                case 2:
                    boundary_nodes = np.array([0, 3, 1])
                case 3:
                    boundary_nodes = np.array([0, 3, 4, 1])
                case _:
                    raise NotImplementedError(
                        "Convection boundary condition"
                        " not implemented for element order > 3"
                    )

            theta_e = np.zeros(boundary_nodes.shape[0])
            for index, bnode in enumerate(boundary_nodes):
                theta_e[index] = element[bnode]

            heat_flux = alpha*(
                theta_e - np.ones(boundary_nodes.shape[0])*outer_temperature
            )
            f_e = (
                integral_heat_flux(node_points, heat_flux, element_order)
                * 2 * np.pi * jacobian_det
            )

            for local_p, global_p in enumerate(element[:2]):
                f[global_p] += f_e[local_p]

        return f

    def set_convection_bc(
        self,
        convective_boundary_elements,
        alpha,
        outer_temperature
    ):
        """Adds a convection boundary condition to the current simulation.

        Parameters:
            convective_boundary_elements: List of element indicies at the
                boundary for which the convective bc is implemented.
            alpha: Heat transfer coefficient to the outer fluid.
            outer_temperature: Fixed temperature outside of the model.
        """
        self.convective_b_e = convective_boundary_elements
        self.convective_alpha = alpha
        self.convective_outer_temp = outer_temperature
        self.enable_convection_bc = True

    def set_dirichlet_bc_load_vector(
        self,
        f: npt.NDArray,
        nodes: npt.NDArray,
        values: npt.NDArray,
        time_step: int
    ) -> npt.NDArray:
        """Adds the dirichlet values for the given load vector.

        Parameters:
            f: Existing load vector. Could be empty or contain already set
                boundary conditions, which will be overwritten at
                dirichlet nodes.
            nodes: Nodes at which the dirichlet bc shall be set.
            values: Values which shall be set at the corresponding node.

        Returns:
            Right hand side vector for the simulation.
        """
        for node, value in zip(nodes, values):
            f[node] = value[time_step]

        return f

    def solve_time(
        self,
        theta_start = None
    ):
        """Runs the simulation using the assembled c and k matrices as well
        as the set excitation.
        Calculates the temperature field and saves it in theta.

        Parameters:
            theta_start: Sets a field for time step 0.
        """
        number_of_time_steps = self.simulation_data.number_of_time_steps
        gamma = self.simulation_data.gamma
        delta_t = self.simulation_data.delta_t

        c = self.c
        k = self.k

        # Init arrays
        theta = np.zeros((k.shape[0], number_of_time_steps), dtype=np.float64)
        if theta_start is not None:
            theta[:, 0] = theta_start
        # theta derived after t (du/dt)
        theta_dt = np.zeros(
            (k.shape[0], number_of_time_steps),
            dtype=np.float64)

        _, c, k = apply_dirichlet_bc(
            np.zeros(c.shape),
            c,
            k,
            self.dirichlet_nodes
        )

        k_star = (k+1/(gamma*delta_t)*c).tocsr()

        if self.enable_convection_bc:
            print("Convection bc is enabled")

        print("Starting heat conduction simulation")
        for time_index in range(number_of_time_steps-1):
            # Set dirichlet nodes for the load vector
            # The load vector could already contain e.g. energy sources
            # which are overwritten at the points where a dirichlet bc is set.
            f = self.set_dirichlet_bc_load_vector(
                self.f[:, time_index],
                self.dirichlet_nodes,
                self.dirichlet_values,
                time_index+1
            )

            # Add a convection boundary condition if it set
            if self.enable_convection_bc:
                f += self._calculate_convection_bc(
                    self.mesh_data.nodes,
                    self.convective_b_e,
                    theta[:, time_index],
                    self.convective_alpha,
                    self.convective_outer_temp,
                    self.mesh_data.element_order
                )

            # Perform Newmark method
            # Predictor step
            theta_tilde = (
                theta[:, time_index]
                + (1-gamma)*delta_t*theta_dt[:, time_index]
            )

            # Solve for next time step
            theta[:, time_index+1] = slin.spsolve(
                k_star, (
                    f
                    + 1/(gamma*delta_t)*c*theta_tilde
                )
            )

            # Perform corrector step
            theta_dt[:, time_index+1] = \
                (theta[:, time_index+1]-theta_tilde)/(gamma*delta_t)

            if (time_index + 1) % 100 == 0:
                print(f"Finished time step {time_index+1}")

        self.theta = theta

    def solve_until_material_parameters_change(
        self,
        initial_theta_field: npt.NDArray,
        initial_time_step: int
    ):
        """Runs the simulation using the assembled c and k matrices as well
        as the set excitation.
        Calculates the temperature field and saves it in theta.

        Parameters:
            theta_start: Field at time step start_index.
            start_index: Sets the starting time step.
        """
        number_of_time_steps = self.simulation_data.number_of_time_steps
        gamma = self.simulation_data.gamma
        delta_t = self.simulation_data.delta_t

        c = self.c
        k = self.k

        # Init arrays
        if initial_time_step == 0:
            self.theta = np.zeros(
                (k.shape[0], number_of_time_steps),
                dtype=np.float64
            )
            if initial_theta_field is not None:
                self.theta[:, initial_time_step] = initial_theta_field
            # theta derived after t (du/dt)
            self.theta_dt = np.zeros(
                (k.shape[0], number_of_time_steps),
                dtype=np.float64)

        _, c, k = apply_dirichlet_bc(
            np.zeros(c.shape),
            c,
            k,
            self.dirichlet_nodes
        )

        k_star = (k+1/(gamma*delta_t)*c).tocsr()

        print("Starting heat conduction simulation")
        for time_index in range(initial_time_step, number_of_time_steps-1):
            # Set dirichlet nodes for the load vector
            # The load vector could already contain e.g. energy sources
            # which are overwritten at the points where a dirichlet bc is set.
            f = self.set_dirichlet_bc_load_vector(
                self.f[:, time_index],
                self.dirichlet_nodes,
                self.dirichlet_values,
                time_index+1
            )

            # Add a convection boundary condition if it set
            if self.enable_convection_bc:
                f += self._calculate_convection_bc(
                    self.mesh_data.nodes,
                    self.convective_b_e,
                    self.theta[:, time_index],
                    self.convective_alpha,
                    self.convective_outer_temp
                )

            # Perform Newmark method
            # Predictor step
            theta_tilde = (
                self.theta[:, time_index]
                + (1-gamma)*delta_t*self.theta_dt[:, time_index]
            )

            # Solve for next time step
            self.theta[:, time_index+1] = slin.spsolve(
                k_star, (
                    f
                    + 1/(gamma*delta_t)*c*theta_tilde
                )
            )

            # Perform corrector step
            self.theta_dt[:, time_index+1] = \
                (self.theta[:, time_index+1]-theta_tilde)/(gamma*delta_t)

            if (time_index + 1) % 100 == 0:
                print(f"Finished time step {time_index+1}")

            # Check if material parameters would change
            temp_field_per_element = get_avg_temp_field_per_element(
                self.theta[:, -1],
                self.mesh_data.elements
            )
            if self.material_manager.update_temperature(
                    temp_field_per_element):
                return time_index+1
        return number_of_time_steps
