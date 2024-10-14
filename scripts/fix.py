import os
import numpy as np
import piezo_fem as pfem

CWD = "/home/jonash/uni/Masterarbeit/simulations/"
#PIEZO_SIM_NAME = "double_sim"
PIEZO_SIM_NAME = "real_model_30k_sinus_volume"
WD = os.path.join(CWD, PIEZO_SIM_NAME)

sim = pfem.Simulation.load_simulation_settings(
    os.path.join(WD, f"{PIEZO_SIM_NAME}.cfg")
)
u = np.load(
    os.path.join(WD, f"{PIEZO_SIM_NAME}_u.npy")
)
mech_loss = np.load(
    os.path.join(WD, f"{PIEZO_SIM_NAME}_mech_loss.npy")
)

sim.gmsh_handler.create_u_default_post_processing_view(
    u,
    sim.simulation_data.number_of_time_steps,
    sim.simulation_data.delta_t,
    True,
    False
)
sim.gmsh_handler.create_element_post_processing_view(
    mech_loss,
    sim.simulation_data.number_of_time_steps,
    sim.simulation_data.delta_t,
    1,
    "Mech loss",
    True
)