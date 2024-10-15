import os
import numpy as np
import matplotlib.pyplot as plt
import piezo_fem as pfem

if __name__ == "__main__":
    SIM_NAME = "energy_check"
    sim_dir = os.path.join(
        "/home/jonash/uni/Masterarbeit/simulations/",
        SIM_NAME
    )
    CONFIG_FILE_PATH = os.path.join(
        sim_dir,
        f"{SIM_NAME}.cfg"
    )
    MECH_LOSS_DENSITY_PATH = os.path.join(
        sim_dir,
        f"{SIM_NAME}_mech_loss.npy"
    )
    DISPLACEMENT_FILE_PATH = os.path.join(
        sim_dir,
        f"{SIM_NAME}_u.npy"
    )

    # Load piezo simulation
    piezo_sim = pfem.Simulation.load_simulation_settings(CONFIG_FILE_PATH)
    piezo_sim.solver.mech_loss = np.load(MECH_LOSS_DENSITY_PATH)
    piezo_sim.solver.u = np.load(DISPLACEMENT_FILE_PATH)

    # Run thermal sim
    heat_sim = pfem.HeatConductionSim(
        piezo_sim.mesh_data,
        pfem.pic255,
        piezo_sim.simulation_data
    )
    heat_sim.assemble()
    heat_sim.set_volume_heat_source(
        piezo_sim.solver.mech_loss,
        piezo_sim.simulation_data.number_of_time_steps,
        piezo_sim.simulation_data.delta_t
    )
    heat_sim.solve_time()

    # Save simulation results
    gmsh_handler = pfem.GmshHandler(
        os.path.join(sim_dir, "thermal.msh")
    )
    gmsh_handler.create_theta_post_processing_view(
        heat_sim.theta,
        piezo_sim.simulation_data.number_of_time_steps,
        piezo_sim.simulation_data.delta_t,
        False
    )
