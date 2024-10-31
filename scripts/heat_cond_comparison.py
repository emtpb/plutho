# Python standard libraries
import os
import numpy as np
import numpy.typing as npt

# Third party libraries
from dotenv import load_dotenv

# Local libraries
import piezo_fem as pfem
from piezo_fem.simulation.base import local_shape_functions, \
    local_to_global_coordinates, quadratic_quadrature


if __name__ == "__main__":
    load_dotenv()

    CWD = os.getenv("piezo_fem_simulation_path")
    if CWD is None:
        print("Couldn't find simulation path.")
        exit(1)

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

    DELTA_T = 0.001
    NUMBER_OF_TIME_STEPS = 1000
    simulation_data = pfem.SimulationData(
        delta_t=DELTA_T,
        number_of_time_steps=NUMBER_OF_TIME_STEPS,
        gamma=0.5,
        beta=0
    )

    # Create sim
    heat_sim = pfem.HeatConductionSim(
        mesh_data, pfem.pic255, simulation_data
    )

    heat_sim.assemble()
    electrode_nodes = gmsh_handler.get_nodes_by_physical_groups(["Electrode"])["Electrode"]

    heat_sim.set_constant_volume_heat_source(
        0*np.ones(len(heat_sim.mesh_data.elements)),
        NUMBER_OF_TIME_STEPS
    )
    theta_start = 30*np.ones(len(nodes))
    heat_sim.solve_time(
        theta_start=theta_start,
        boundary_elements=boundary_elements
    )

    #pfem.create_scalar_field_as_csv(
    #    "theta",
    #    heat_sim.theta,
    #    nodes,
    #    os.path.join(SIM_DIR, "field")
    #)

    gmsh_handler.create_theta_post_processing_view(
        heat_sim.theta,
        NUMBER_OF_TIME_STEPS,
        DELTA_T,
        False
    )
    np.save(os.path.join(SIM_DIR, "theta"), heat_sim.theta)
    print("Average temperature", np.mean(heat_sim.theta[:, -1]))

    # Check if the thermal energy integral is working
    temp_field_energy = pfem.postprocessing.calculate_stored_thermal_energy(
        heat_sim.theta[:, -1],
        nodes,
        elements,
        heat_sim.material_data.heat_capacity,
        heat_sim.material_data.density
    )
    print("Calculated temperature field energy", temp_field_energy)

    import gmsh
    gmsh.fltk.run()
