# Python standard libraries
import os

# Third party libraries
import numpy as np

# Local libraries
import piezo_fem as pfem

### Helper functions
# ------------------

def compare_numpy_array(a, b):
    if a.shape != b.shape:
        return False

    return np.allclose(a, b, atol=1e-15)


### Test functions
# ----------------

def test_thermo_time(tmp_path):
    # Create mesh
    mesh_path = os.path.join(tmp_path, "default_mesh.msh")
    mesh = pfem.Mesh(mesh_path)
    # TODO Maybe use mesh with smaller element size?
    mesh.generate_rectangular_mesh()

    sim = pfem.SingleSimulation(
        tmp_path,
        "thermo_time_test",
        mesh
    )

    # Simulation parameters
    INPUT_POWER_DENSITY = 1
    number_of_time_steps = 1000
    delta_t = 0.001
    nodes, elements = mesh.get_mesh_nodes_and_elements()
    number_of_elements = len(elements)

    sim.setup_thermo_time_domain(
        delta_t,
        number_of_time_steps,
        0.5
    )

    sim.add_material(
        "pic255",
        pfem.pic255,
        None
    )

    # Set constant volume loss density
    sim.solver.set_constant_volume_heat_source(
        INPUT_POWER_DENSITY*np.ones(number_of_elements),
        number_of_time_steps
    )

    sim.simulate()

    # Calculated thermal stored energy
    temp_field_energy = pfem.postprocessing.\
        calculate_stored_thermal_energy(
            sim.solver.theta[:, -1],
            nodes,
            elements,
            sim.material_manager.get_heat_capacity(0),
            sim.material_manager.get_density(0)
    )

    volume = np.sum(
        pfem.simulation.base.calculate_volumes(sim.solver.local_elements)
    )

    input_energy = INPUT_POWER_DENSITY*volume

    MAX_ERROR = 1e-9 # TODO Is this enough?
    assert \
        np.abs(temp_field_energy-input_energy) < MAX_ERROR, \
        "Calculated therm field energy is not correct."


def test_piezo_time(tmp_path):
    # Create mesh
    mesh_path = os.path.join(tmp_path, "default_mesh.msh")
    mesh = pfem.Mesh(mesh_path)
    # TODO Maybe use mesh with smaller element size?
    mesh.generate_rectangular_mesh()

    sim = pfem.SingleSimulation(
        tmp_path,
        "piezo_time_test",
        mesh
    )

    sim.add_material(
        "pic255",
        pfem.pic255,
        None
    )

    NUMBER_OF_TIME_STEPS = 100

    sim.setup_piezo_time_domain(pfem.SimulationData(
        number_of_time_steps=NUMBER_OF_TIME_STEPS,
        delta_t=1e-8,
        gamma=0.5,
        beta=0.25
    ))

    # Set triangular excitation
    excitation = np.zeros(NUMBER_OF_TIME_STEPS)
    excitation[1:10] = (
        1 * np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    )

    # Set boundary conditions
    sim.add_dirichlet_bc(
        pfem.FieldType.PHI,
        "Electrode",
        excitation
    )
    sim.add_dirichlet_bc(
        pfem.FieldType.PHI,
        "Ground",
        np.zeros(NUMBER_OF_TIME_STEPS)
    )
    sim.add_dirichlet_bc(
        pfem.FieldType.U_R,
        "Symaxis",
        np.zeros(NUMBER_OF_TIME_STEPS)
    )

    electrode_elements = mesh.get_elements_by_physical_groups(
        ["Electrode"]
    )["Electrode"]

    sim.simulate(electrode_elements=electrode_elements)

    # Compare results with test data
    # Load test data
    test_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "piezo_time"
    )
    test_q = np.load(os.path.join(test_folder, "q.npy"))
    test_u = np.load(os.path.join(test_folder, "u.npy"))

    uut_q = sim.solver.q
    uut_u = sim.solver.u

    # Compare arrays
    # For displacement just take last time step. TODO Is this sufficient?
    assert compare_numpy_array(uut_q, test_q), "Charge is not equal"
    assert compare_numpy_array(uut_u[:, -1], test_u[:, -1]), \
           "Displacement u is not equal"


if __name__ == "__main__":
    dir = "tests/data"
    test_piezo_time(dir)



