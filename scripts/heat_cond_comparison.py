# Python standard libraries
import os
import numpy as np
import numpy.typing as npt

# Third party libraries
import gmsh

# Local libraries
import piezo_fem as pfem
from piezo_fem.simulation.base import local_shape_functions, \
    local_to_global_coordinates, quadratic_quadrature, \
        gradient_local_shape_functions


def energy_integral_theta(
        node_points: npt.NDArray,
        theta: npt.NDArray):
    """Integrates the given element over the given theta field.

    Parameters:
        node_points: List of node points [[x1, x2, x3], [y1, y2, y3]] of
            one triangle.
        theta: List of the temperature field values of the points
            [theta1, theta2, theta3].
    """
    def inner(s, t):
        n = local_shape_functions(s, t)
        r = local_to_global_coordinates(node_points, s, t)[0]

        return np.dot(n.T, theta)*r

    return quadratic_quadrature(inner)


if __name__ == "__main__":
    #WD = "/home/jonash/uni/Masterarbeit/simulations/"
    CWD = "/upb/departments/emt/Student/jonasho/Masterarbeit/simulations/"
    SIM_NAME = "heat_cond_comparison"
    SIM_DIR = os.path.join(CWD, SIM_NAME)

    if not os.path.exists(SIM_DIR):
        os.makedirs(SIM_DIR)

    gmsh_handler = pfem.GmshHandler(
        os.path.join(SIM_DIR, "mesh.msh")
    )
    gmsh_handler.generate_rectangular_mesh(
        0.005, 0.001, 0.0001, 0
    )
    nodes, elements = gmsh_handler.get_mesh_nodes_and_elements()
    mesh_data = pfem.MeshData(
        nodes,
        elements
    )
    el = gmsh_handler.get_elements_by_physical_groups(["Electrode", "Ground"])
    boundary_elements = np.concatenate([el["Electrode"], el["Ground"]])

    delta_t = 0.001
    number_of_time_steps = 1000
    simulation_data = pfem.SimulationData(
        delta_t = delta_t,
        number_of_time_steps=number_of_time_steps,
        gamma=0.5,
        beta=0
    )

    # Create sim
    heat_sim = pfem.HeatConductionSim(
        mesh_data, pfem.pic255, simulation_data
    )
    # losses = np.zeros(len(elements))
    # heat_sim.set_volume_heat_source(
    #     losses
    # )
    heat_sim.assemble()
    electrode_nodes = gmsh_handler.get_nodes_by_physical_groups(["Electrode"])["Electrode"]
    #heat_sim.set_fixed_temperature(
    #    electrode_nodes,
    #    20*np.ones(len(electrode_nodes)))
    heat_sim.set_constant_volume_heat_source(
        1*np.ones(len(heat_sim.mesh_data.elements)),
        number_of_time_steps
    )
    heat_sim.solve_time(boundary_elements=boundary_elements)
    #pfem.create_scalar_field_as_csv(
    #    "theta",
    #    heat_sim.theta,
    #    nodes,
    #    os.path.join(SIM_DIR, "field")
    #)
    gmsh_handler.create_theta_post_processing_view(
        heat_sim.theta,
        number_of_time_steps,
        delta_t,
        False
    )
    np.save(os.path.join(SIM_DIR, "theta"), heat_sim.theta)
    print("Average temperature", np.mean(heat_sim.theta[:, -1]))
    
    # Check if the thermal energy integral is working
    temp_field_energy= pfem.postprocessing.calculate_stored_thermal_energy(
        heat_sim.theta[-1],
        nodes,
        elements,
        heat_sim.material_data.heat_capacity,
        heat_sim.material_data.density
    )
    print("Calculated temperature field energy", temp_field_energy)
