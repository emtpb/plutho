# Python standard libraries
import os
import numpy as np
import numpy.typing as npt

# Third party libraries
import gmsh

# Local libraries
import piezo_fem as pfem


if __name__ == "__main__":
    CWD = "/home/jonash/uni/Masterarbeit/simulations/"
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

    delta_t = 1e-2
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
    heat_sim.set_fixed_temperature(
        electrode_nodes,
        20*np.ones(len(electrode_nodes)))
    heat_sim.solve_time()
    pfem.create_scalar_field_as_csv(
        "theta",
        heat_sim.theta,
        nodes,
        os.path.join(SIM_DIR, "field")
    )
    #gmsh_handler.create_theta_post_processing_view(
    #    heat_sim.theta,
    #    number_of_time_steps,
    #    delta_t,
    #    False
    #)
    #gmsh.fltk.run()