"""Extra module for the simulation of heat conduction over time."""

# Python standard libraries
import numpy as np
import numpy.typing as npt
import scipy.sparse.linalg as slin
from scipy import sparse

# Local libraries
from .base import MaterialData, SimulationData, MeshData, \
    local_shape_functions, gradient_local_shape_functions, \
    local_to_global_coordinates, integral_m, \
    quadratic_quadrature, line_quadrature


def integral_heat_flux(
        node_points: npt.NDArray,
        heat_flux: npt.NDArray) -> npt.NDArray:
    """Integrates the heat flux using the shape functions.

    Parameters:
        node_points: List of node points [[x1, x2, ..], [y1, y2, ..]]
        heat_flux: Heat flux at the points

    Returns:
        npt.NDArray heat flux integral on each point."""
    def inner(s, t):
        n = local_shape_functions(s, t)
        r = local_to_global_coordinates(node_points, s, t)[0]

        return n*heat_flux*r

    return line_quadrature(inner)

def integral_ktheta(
        node_points: npt.NDArray,
        jacobian_inverted_t: npt.NDArray) -> npt.NDArray:
    """Calculates the Ktheta integral.

    Parameters:
        node_points: List of node points [[x1, x2, ..], [y1, y2, ..]]
        jacobian_inverted_t: Jacobian matrix inverted and transposed, needed
            for calculation of global derivatives

    Returns:
        npt.NDArray: 3x3 Ktheta matrix for the given element.
    """
    def inner(s, t):
        dn = gradient_local_shape_functions()
        global_dn = np.dot(jacobian_inverted_t, dn)
        r = local_to_global_coordinates(node_points, s, t)[0]

        return np.dot(global_dn.T, global_dn)*r

    return quadratic_quadrature(inner)


def integral_theta_load(
        node_points: npt.NDArray,
        mech_loss: float) -> npt.NDArray:
    """Returns the load value for the temperature field (f) for the specific
    element.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        point_loss: Loss power on each node (heat source)
    Returns:
        npt.NDArray: f vector value at the specific ndoe
    """
    def inner(s, t):
        n = local_shape_functions(s, t)
        r = local_to_global_coordinates(node_points, s, t)[0]
        return n*mech_loss*r

    return quadratic_quadrature(inner)


class HeatConductionSim:
    """Simulates a heat conduction problem for the given mesh, material and
    simulation information.

    Attributes:
        mesh_data: Contains the mesh information.
        material_data: Contains the information about the materials.
        simulation_data: Contains the information about the simulation.
        c: Damping matrix.
        k: Stiffness matrix.
        theta: Temperature field defined for every node.
        f: Load vector for the temperature field contains the information
            from the given losses."""
    mesh_data: MeshData
    material_data: MaterialData
    simulation_data: SimulationData

    c: npt.NDArray
    k: npt.NDArray

    theta: npt.NDArray
    f: npt.NDArray

    def __init__(
            self,
            mesh_data: MeshData,
            material_data: MaterialData,
            simulation_data: SimulationData):
        self.mesh_data = mesh_data
        self.material_data = material_data
        self.simulation_data = simulation_data

    def assemble(self):
        """Assembles the FEM matrices based on the set mesh_data and
        material_data.
        The matrices are stored in self.c, self.k.
        """
        # TODO Assembly takes long rework this algorithm?
        # Maybe the 2x2 matrix slicing is not very fast
        nodes = self.mesh_data.nodes

        number_of_nodes = len(nodes)
        c = sparse.lil_matrix(
            (number_of_nodes, number_of_nodes),
            dtype=np.float64)
        k = sparse.lil_matrix(
            (number_of_nodes, number_of_nodes),
            dtype=np.float64)

        for element in self.mesh_data.elements:
            dn = gradient_local_shape_functions()
            # Get node points of element in format
            # [x1 x2 x3]
            # [y1 y2 y3] where (xi, yi) are the coordinates for Node i
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
            jacobian_det = np.linalg.det(jacobian)

            ctheta_e = (
                integral_m(node_points)
                * self.material_data.density
                * self.material_data.heat_capacity
                * jacobian_det * 2 * np.pi
            )
            ktheta_e = (
                integral_ktheta(
                    node_points,
                    jacobian_inverted_t)
                * self.material_data.thermal_conductivity
                * jacobian_det * 2 * np.pi
            )

            # Now assemble all element matrices
            for local_p, global_p in enumerate(element):
                for local_q, global_q in enumerate(element):
                    c[global_p, global_q] += ctheta_e[local_p, local_q]
                    k[global_p, global_q] += ktheta_e[local_p, local_q]

        self.c = c.tolil()
        self.k = k.tolil()

    def set_volume_heat_source(
            self,
            mech_loss_density,
            number_of_time_steps,
            delta_t):
        """Sets the excitation for the heat conduction simulation.
        The mech_losses are set for every time step.

        Parameters:
            mech_losses: The mechanical losses for each element.
        """
        time_values = np.arange(number_of_time_steps)*delta_t
        nodes = self.mesh_data.nodes
        number_of_nodes = len(nodes)
        f = np.zeros(shape=(number_of_nodes, number_of_time_steps))
        for time_step, _ in enumerate(time_values):
            for element_index, element in enumerate(self.mesh_data.elements):
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
                jacobian_det = np.linalg.det(jacobian)

                # TODO Maybe the outer time step loop can be removed
                f_theta_e = integral_theta_load(
                    node_points,
                    mech_loss_density[element_index, time_step])*2*np.pi*jacobian_det

                for local_p, global_p in enumerate(element):
                    f[global_p, time_step] += f_theta_e[local_p]

        self.f = f

    def set_constant_volume_heat_source(
            self,
            mech_loss_density,
            number_of_time_steps):
        """Sets the excitation for the heat conduction simulation.
        The mech_loss_density are set for every time step.

        Parameters:
            mech_loss_density: The mechanical losses for each element.
        """
        nodes = self.mesh_data.nodes
        number_of_nodes = len(nodes)
        f = np.zeros(number_of_nodes)

        for element_index, element in enumerate(self.mesh_data.elements):
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
            jacobian_det = np.linalg.det(jacobian)

            f_theta_e = integral_theta_load(
                node_points,
                mech_loss_density[element_index])*2*np.pi*jacobian_det

            for local_p, global_p in enumerate(element):
                f[global_p] += f_theta_e[local_p]

        # Repeat it for n time steps
        self.f = np.tile(f.reshape(-1, 1), (1, number_of_time_steps))

    def set_fixed_temperature(self, nodes, values):
        """Sets a fixed temperature of value at node (dirichlet boundary
        condition).
    
        Parameters:
            nodes: Nodes at which the values shall be set.
            values: Values which are set at the corresponding node (same index)
        """
        self.f = np.zeros(
            shape=(len(self.mesh_data.nodes),
                   self.simulation_data.number_of_time_steps))
        for node, value in zip(nodes, values):
            self.c[node, :] = 0
            self.k[node, :] = 0
            self.k[node, node] = 1
            self.f[node, :] = value

    def calculate_convection_bc(
            self,
            nodes,
            boundary_elements,
            theta,
            alpha,
            outer_temperature):
        f = np.zeros(len(nodes))

        for _, element in enumerate(boundary_elements):
            node_points = np.array([
                [nodes[element[0]][0],
                 nodes[element[1]][0],
                 nodes[element[2]][0]],
                [nodes[element[0]][1],
                 nodes[element[1]][1],
                 nodes[element[2]][1]]
            ])
            min_r = np.min([nodes[element[0]][0], nodes[element[1]][0]])
            max_r = np.max([nodes[element[0]][0], nodes[element[1]][0]])
            jacobian_det = (min_r-max_r)
            theta_e = np.array([
                theta[element[0]],
                theta[element[1]],
                theta[element[2]]
            ])

            heat_flux = alpha*(theta_e-np.ones(3)*outer_temperature)
            heat_flux[2] = 0  # No heat flux for the inner element
            f_e = (
                integral_heat_flux(node_points, heat_flux)
                * 2 * np.pi * jacobian_det
            )

            for local_p, global_p in enumerate(element):
                f[global_p] += f_e[local_p]

        return f

    def solve_time(self, theta_start=None, boundary_elements=[]):
        """Runs the simulation using the assembled c and k matrices as well
        as the set excitation.
        Calculates the temperature field and saves it in theta.
        """
        number_of_time_steps = self.simulation_data.number_of_time_steps
        gamma = self.simulation_data.gamma
        delta_t = self.simulation_data.delta_t

        c = self.c
        k = self.k

        # Init arrays
        # Temperature theta which is calculated
        theta = np.zeros((k.shape[0], number_of_time_steps), dtype=np.float64)
        if theta_start is not None:
            theta[:, 0] = theta_start
        # theta derived after t (du/dt)
        theta_dt = np.zeros(
            (k.shape[0], number_of_time_steps),
            dtype=np.float64)

        k_star = (k+1/(gamma*delta_t)*c).tocsr()

        print("Starting heat conduction simulation")
        for time_index in range(number_of_time_steps-1):
            f = self.f[:, time_index]+ self.calculate_convection_bc(
                self.mesh_data.nodes,
                boundary_elements,
                theta[:, time_index],
                80,
                20
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
