import os
import piezo_fem as pfem
import gmsh

if __name__ == "__main__":
    cwd = os.path.join(os.path.abspath(os.path.dirname(__file__)))
    working_directory = os.path.join(cwd, "test_sim")
    sim = pfem.Simulation(working_directory, pfem.pic255, "Test")
    sim.create_disc_mesh(0.005, 0.001, 0.00015)
    sim.set_simulation(
        delta_t=1e-8,
        number_of_time_steps=100,
        gamma=0.5,
        beta=0.25,
        simulation_type=pfem.SimulationType.PIEZOELECTRIC,
    )
    sim.set_triangle_pulse_excitation(1)
    sim.set_boundary_conditions()
    sim.save_simulation_settings("This is just a test.")
    sim.solve()
    sim.create_post_processing_views()
    gmsh.fltk.run()