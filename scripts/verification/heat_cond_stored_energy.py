
# Python standard libraries
import os

# Third party libraries
import numpy as np
from dotenv import load_dotenv

# Local libraries
import piezo_fem as pfem


if __name__ == "__main__":
    load_dotenv()
    MODEL_NAME = "impedance_pic255_thesis_fine_mesh"
    CWD = os.path.join(
        os.environ["piezo_fem_simulation_path"],
        MODEL_NAME
    )

    SIM_NAME = "heat_cond_test_new_model"
    mesh = pfem.Mesh(
        os.path.join(
            os.environ["piezo_fem_simulation_path"],
            "disc_mesh.msh"
        ),
        True
    )

    DELTA_T = 0.01
    NUMBER_OF_TIME_STEPS = 100

    # Create single simulation object
    sim = pfem.SingleSimulation(
        CWD,
        SIM_NAME,
        mesh
    )

    # Setup heat cond sim
    sim.setup_heat_conduction_time_domain(
        DELTA_T,
        NUMBER_OF_TIME_STEPS,
        0.5
    )

    # Add materials
    sim.add_material(
        "pic255",
        pfem.materials.pic255,
        ""
    )

    # Prevent mypy error message
    if not isinstance(sim.solver, pfem.HeatConductionSim):
        raise ValueError("Wrong simulation type.")

    # Add boundary condition
    input_power_density = 1
    sim.solver.set_constant_volume_heat_source(
        input_power_density * np.ones(len(sim.mesh_data.elements)),
        NUMBER_OF_TIME_STEPS
    )

    # Run simulation
    sim.simulate()

    # Check if the thermal energy integral is working
    temp_field_energy = pfem.postprocessing.calculate_stored_thermal_energy(
        sim.solver.theta[:, -1],
        sim.mesh_data.nodes,
        sim.mesh_data.elements,
        sim.material_manager.get_heat_capacity(0),
        sim.material_manager.get_density(0)
    )

    volume = np.sum(
        pfem.simulation.base.calculate_volumes(sim.solver.local_elements)
    )
    print("Calculated temperature field energy", temp_field_energy)
    print("Excpected field energy", input_power_density*volume)
