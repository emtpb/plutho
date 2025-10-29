"""Module for the simulation of a heat conduction problem over time.
Additonaly heat convection can be set."""


# Python standard libraries
from typing import List, Union

# Third party libraries
import numpy as np
import numpy.typing as npt
import scipy.sparse.linalg as slin
from scipy import sparse

# Local libraries
from .integrals import integral_heat_flux, integral_ktheta, \
    integral_theta_load, integral_m
from .solver import FEMSolver
from .helpers import mat_apply_dbcs, \
    get_avg_temp_field_per_element
from ..enums import SolverType
from ..mesh import Mesh


__all__ = [
    "ThermoTime"
]


class ThermoTime(FEMSolver):
    """Class for the simulation of time domain heat conduction problems.

    Attributes:
        c: Sparse damping matrix.
        k: Sparse stiffness matrix.
        theta: Resulting thermal field.
        enable_convection_bc: True if a convection boundary condition is set.
        convective_b_e: List of elements for which the convection boundary
            condition is applied.
        convective_alpha: Heat transfer coefficient of the convection.
        convective_outer_temp: Temperature of the material outside of the
            simulated material.
    """
    # FEM Matrices
    c: sparse.lil_array
    k: sparse.lil_array

    # Resulting arrays
    theta: npt.NDArray

    # Convective bc
    enable_convection_bc: bool
    convective_b_e: npt.NDArray
    convective_alpha: float
    convective_outer_temp: float

    def __init__(self, simulation_name: str, mesh: Mesh):
        super().__init__(simulation_name, mesh)

        self.solver_type = SolverType.ThermoTime
        self.enable_convection_bc = False
        self.f = None

    def assemble(self):
        """Assembles the FEM matrices based on the set mesh_data and
        material_data.
        The matrices are stored in self.c, self.k.
        """
        # TODO Assembly takes long rework this algorithm?
        # Maybe the 2x2 matrix slicing is not very fast
        self.material_manager.initialize_materials()
        nodes = self.mesh_data.nodes
        element_order = self.mesh_data.element_order

        number_of_nodes = len(nodes)
        c = sparse.lil_array(
            (number_of_nodes, number_of_nodes),
            dtype=np.float64)
        k = sparse.lil_array(
            (number_of_nodes, number_of_nodes),
            dtype=np.float64)

        for element_index, element in enumerate(self.mesh_data.elements):
            node_points = self.node_points[element_index]

            ctheta_e = (
                integral_m(node_points, element_order)
                * self.material_manager.get_density(element_index)
                * self.material_manager.get_heat_capacity(element_index)
                * 2 * np.pi
            )
            ktheta_e = (
                integral_ktheta(
                    node_points,
                    element_order
                )
                * self.material_manager.get_thermal_conductivity(element_index)
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
        The mech_loss_density is set for every time step.

        Parameters:
            mech_loss_density: The mechanical losses for each element. They
                will be applied for every time step.
            number_of_time_steps: Total number of time steps.
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
        if self.f is None:
            self.f = np.tile(f.reshape(-1, 1), (1, number_of_time_steps))
        else:
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
            boundary_elements: Boundary elements at which the convection
                boundary is added.
            theta: Current temperature field at all nodes.
            alpha: Heat transfer coefficient.
            outer_temperature: Temperature outside of the model.
            element_order: Element order of the mesh.
        """
        # This function is not yet tested
        points_per_element = int(1/2*(element_order+1)*(element_order+2))

        f = np.zeros(len(nodes))
        for _, element in enumerate(boundary_elements):
            node_points = np.zeros(shape=(2, points_per_element))
            for node_index in range(points_per_element):
                node_points[:, node_index] = [
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

            theta_e = np.zeros(points_per_element)
            theta_e[boundary_nodes] = theta[element[boundary_nodes]]

            outer_temps = np.zeros(points_per_element)
            outer_temps[boundary_nodes] = outer_temperature

            heat_flux = alpha*(
                theta_e - outer_temps
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
        convective_boundary_elements: npt.NDArray,
        alpha: float,
        outer_temperature: float
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

    def simulate(
        self,
        delta_t: float,
        number_of_time_steps: int,
        gamma: float,
        theta_start: Union[npt.NDArray, int, None] = None
    ):
        """Runs the time domain heat conduction simulation.
        The resulting temperature field is stored in self.theta.

        Parameters:
            delta_t: Distance between two time steps.
            number_of_time_steps: Total number of time steps of the simulation.
            gamma: Newmark integration parameter.
            theta_start: Sets an initial temperature field.
        """
        c = self.c
        k = self.k

        dirichlet_nodes = np.array(self.dirichlet_nodes)
        dirichlet_values = np.array(self.dirichlet_values)
        node_count = len(self.mesh_data.nodes)

        # Init arrays
        theta = np.zeros(
            (number_of_time_steps, node_count),
            dtype=np.float64
        )
        theta_dt = np.zeros(
            (number_of_time_steps, node_count),
            dtype=np.float64
        )

        if theta_start is not None:
            theta[0, :] = theta_start

        _, c, k = mat_apply_dbcs(
            None,
            c,
            k,
            dirichlet_nodes
        )

        k_star = (k+1/(gamma*delta_t)*c).tocsr()

        if self.enable_convection_bc:
            print("Convection bc is enabled")

        print("Starting heat conduction simulation")
        for time_index in range(number_of_time_steps-1):
            current_f = self.f[:, time_index+1]

            # Add a convection boundary condition if it set
            if self.enable_convection_bc:
                current_f += self._calculate_convection_bc(
                    self.mesh_data.nodes,
                    self.convective_b_e,
                    theta[time_index, :],
                    self.convective_alpha,
                    self.convective_outer_temp,
                    self.mesh_data.element_order
                )

            if len(dirichlet_nodes) > 0:
                current_f[dirichlet_nodes] = dirichlet_values

            # Perform Newmark method
            # Predictor step
            theta_tilde = (
                theta[time_index, :]
                + (1-gamma)*delta_t*theta_dt[time_index, :]
            )

            # Solve for next time step
            theta[time_index+1, :] = slin.spsolve(
                k_star, (
                    current_f
                    + 1/(gamma*delta_t)*c@theta_tilde
                )
            )

            # Perform corrector step
            theta_dt[time_index+1, :] = \
                (theta[time_index+1, :]-theta_tilde)/(gamma*delta_t)

            if (time_index + 1) % 100 == 0:
                print(f"Finished time step {time_index+1}")

        self.theta = theta

    def simulate_until_material_parameters_change(
        self,
        delta_t: float,
        number_of_time_steps: int,
        gamma: float,
        initial_theta_field: npt.NDArray,
        initial_time_step: int
    ) -> int:
        """Runs the time domain heat conduction simulation until the material
        parameters of the set material model are changed sufficiently.
        When the simulation is stopped due to a change of material parameters,
        the last simulated time step is returned.

        Parameters:
            delta_t: Difference btweeen two time steps.
            number_of_time_steps: Total number of time steps of the simulation.
            gamma: Newmark integration parameter.
            initial_theta_field: Initial temperature field at the first time
                step.
            initial_time_step: Sets the starting time step. This is needed, if
                the simulation was interruped due to changed materials and is
                now continued. The initial_thea_field must also be set
                accordingly.
        """
        c = self.c
        k = self.k

        dirichlet_nodes = np.array(self.dirichlet_nodes)
        dirichlet_values = np.array(self.dirichlet_values)
        node_count = len(self.mesh_data.nodes)

        # Init arrays
        if initial_time_step == 0:
            self.theta = np.zeros(
                (number_of_time_steps, node_count),
                dtype=np.float64
            )
            self.theta_dt = np.zeros(
                (number_of_time_steps, node_count),
                dtype=np.float64
            )
            if initial_theta_field is not None:
                self.theta[:, initial_time_step] = initial_theta_field

        _, c, k = mat_apply_dbcs(
            None,
            c,
            k,
            dirichlet_nodes
        )

        k_star = (k+1/(gamma*delta_t)*c).tocsr()

        print("Starting heat conduction simulation")
        for time_index in range(initial_time_step, number_of_time_steps-1):
            current_f = self.f[:, time_index+1]

            # Add a convection boundary condition if it set
            if self.enable_convection_bc:
                current_f += self._calculate_convection_bc(
                    self.mesh_data.nodes,
                    self.convective_b_e,
                    self.theta[time_index, :],
                    self.convective_alpha,
                    self.convective_outer_temp,
                    self.mesh_data.element_order
                )

            current_f[dirichlet_nodes] = dirichlet_values

            # Perform Newmark method
            # Predictor step
            theta_tilde = (
                self.theta[:, time_index]
                + (1-gamma)*delta_t*self.theta_dt[:, time_index]
            )

            # Solve for next time step
            self.theta[time_index+1, :] = slin.spsolve(
                k_star, (
                    current_f
                    + 1/(gamma*delta_t)*c@theta_tilde
                )
            )

            # Perform corrector step
            self.theta_dt[time_index+1, :] = \
                (self.theta[time_index+1, :]-theta_tilde)/(gamma*delta_t)

            if (time_index + 1) % 100 == 0:
                print(f"Finished time step {time_index+1}")

            # Check if material parameters would change
            temp_field_per_element = get_avg_temp_field_per_element(
                self.theta[-1, :],
                self.mesh_data.elements
            )
            if self.material_manager.update_temperature(
                    temp_field_per_element):
                return time_index+1

        return number_of_time_steps
