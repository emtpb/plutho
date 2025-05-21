# Python standard libraries
import os

# Third party libraries
import numpy as np
import pytest

# Local libraries
import piezo_fem as pfem

# -------- Global variables --------

NUMBER_OF_TIME_STEPS = 1000
MAX_ERROR = 1e-15


# -------- Helper functions --------

def compare_numpy_array(a, b):
    if a.shape != b.shape:
        return False

    return np.allclose(a, b, atol=MAX_ERROR)


# -------- Test functions --------


def test_thermo_time(tmp_path):
    """Test for the thermal simulation. Calculates the total stored energy in
    the simulation and compares it with the input energy.
    """
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
    delta_t = 0.001
    nodes, elements = mesh.get_mesh_nodes_and_elements()
    number_of_elements = len(elements)

    sim.setup_thermo_time_domain(
        delta_t,
        NUMBER_OF_TIME_STEPS,
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
        NUMBER_OF_TIME_STEPS
    )

    sim.simulate()

    # Calculated thermal stored energy
    temp_field_energy = pfem.calculate_stored_thermal_energy(
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

    MAX_ERROR = 1e-9  # TODO Is this enough?
    assert \
        np.abs(temp_field_energy-input_energy) < MAX_ERROR, \
        "Calculated therm field energy is not correct."


def test_piezo_time(tmp_path):
    """Test function for the piezo time domain simulation.
    Tests the simulation for a triangular excitation."""
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

    sim.setup_piezo_time_domain(
        number_of_time_steps=NUMBER_OF_TIME_STEPS,
        delta_t=1e-8,
        gamma=0.5,
        beta=0.25
    )

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

    uut_q = sim.solver.q[-1]
    uut_u = sim.solver.u[:, -1]

    # Compare arrays
    # For displacement just take last time step. TODO Is this sufficient?
    assert compare_numpy_array(uut_q, test_q), "Charge is not equal"
    assert compare_numpy_array(uut_u, test_u), \
        "Displacement u is not equal"


def test_piezo_freq(tmp_path):
    """Test function for the piezo frequency domain simulation. Tests the
    displacement field and charge for a sinusoidal signal.
    """
    # Create mesh
    mesh_path = os.path.join(tmp_path, "default_mesh.msh")
    mesh = pfem.Mesh(mesh_path)
    # TODO Maybe use mesh with smaller element size?
    mesh.generate_rectangular_mesh()

    sim = pfem.SingleSimulation(
        tmp_path,
        "piezo_freq_test",
        mesh
    )

    sim.add_material(
        "pic255",
        pfem.pic255,
        None
    )

    sim.setup_piezo_freq_domain(np.array([2e6]))

    # Set boundary conditions
    sim.add_dirichlet_bc(
        pfem.FieldType.PHI,
        "Electrode",
        np.ones(1)
    )
    sim.add_dirichlet_bc(
        pfem.FieldType.PHI,
        "Ground",
        np.zeros(1)
    )
    sim.add_dirichlet_bc(
        pfem.FieldType.U_R,
        "Symaxis",
        np.zeros(1)
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
        "piezo_freq"
    )

    test_q = np.load(os.path.join(test_folder, "q.npy"))
    test_u = np.load(os.path.join(test_folder, "u.npy"))

    uut_q = sim.solver.q
    uut_u = sim.solver.u

    # Compare arrays
    # For displacement just take last time step. TODO Is this sufficient?
    assert compare_numpy_array(uut_q, test_q), "Charge is not equal"
    assert compare_numpy_array(uut_u, test_u), \
        "Displacement u is not equal"


def test_thermo_piezo_time(tmp_path):
    """Test for the thermo piezo simulation using a triangular excitation.
    The input energy is compared with the total loss energy as well as the
    stored energy in the thermal field. Additionaly the simulation results at
    last time step are compared with fixed results.
    """
    # Create mesh
    mesh_path = os.path.join(tmp_path, "default_mesh.msh")
    mesh = pfem.Mesh(mesh_path)
    # TODO Maybe use mesh with smaller element size?
    mesh.generate_rectangular_mesh()
    nodes, elements = mesh.get_mesh_nodes_and_elements()

    sim = pfem.SingleSimulation(
        tmp_path,
        "thermo_piezo_time",
        mesh
    )

    sim.add_material(
        "pic255",
        pfem.pic255,
        None
    )

    number_of_nodes = len(nodes)
    DELTA_T = 1e-8
    NUMBER_OF_TIME_STEPS = 10000

    sim.setup_thermo_piezo_time_domain(
        number_of_time_steps=NUMBER_OF_TIME_STEPS,
        delta_t=DELTA_T,
        gamma=0.5,
        beta=0.25
    )

    # Set sinusoidal excitation
    AMPLITUDE = 1
    FREQUENCY = 2e6
    time_values = np.arange(NUMBER_OF_TIME_STEPS)*DELTA_T
    excitation = AMPLITUDE*np.sin(2*np.pi*FREQUENCY*time_values)

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

    # np.save("tests/data/thermo_piezo_time/u.npy", sim.solver.u[:, -1])
    # np.save("tests/data/thermo_piezo_time/q.npy", sim.solver.q[-1])
    # np.save("tests/data/thermo_piezo_time/mech_loss.npy", sim.solver.mech_loss[:, -1])

    test_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "thermo_piezo_time"
    )

    # Calculate input energy
    input_energy = pfem.calculate_electrical_input_energy(
        excitation,
        sim.solver.q,
        DELTA_T
    )

    # Calculate total loss energy
    volumes = pfem.simulation.base.calculate_volumes(
        sim.solver.local_elements
    )
    power = np.zeros(NUMBER_OF_TIME_STEPS)
    for time_step in range(NUMBER_OF_TIME_STEPS):
        power[time_step] = np.dot(sim.solver.mech_loss[:, time_step], volumes)

    total_loss_energy = np.trapezoid(
        power,
        None,
        DELTA_T
    )

    # Calculate stored thermal energy
    theta = sim.solver.u[3*number_of_nodes:]
    stored_thermal_energy = pfem.calculate_stored_thermal_energy(
        theta[:, -1],
        nodes,
        elements,
        sim.material_manager.get_heat_capacity(0),
        sim.material_manager.get_density(0)
    )

    # The input_energy has a bigger difference to the two other values
    # TODO Is this due to wrong simulation? Too short simulation time?
    # Maybe the signals are not fully dissipated?
    assert np.abs(input_energy - total_loss_energy) < 1e-11, \
        "Input energy does not equal total loss energy"
    assert np.abs(total_loss_energy - stored_thermal_energy) < MAX_ERROR, \
        "Total loss energy does not equal stored thermal energy"

    # Compare results with test data
    # Load test data
    test_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "thermo_piezo_time"
    )
    test_q = np.load(os.path.join(test_folder, "q.npy"))
    test_u = np.load(os.path.join(test_folder, "u.npy"))
    test_mech_loss = np.load(os.path.join(test_folder, "mech_loss.npy"))

    uut_q = sim.solver.q[-1]
    uut_u = sim.solver.u[:, -1]
    uut_mech_loss = sim.solver.mech_loss[:, -1]

    print("Test q:", test_q)
    print("UUT q:", uut_q)
    print("Test u:", test_u)
    print("UUT u:", uut_u)
    print("Test mech loss", test_mech_loss)
    print("UUT mech loss", uut_mech_loss)

    # Compare arrays
    assert compare_numpy_array(uut_q, test_q), "Charge is not equal"
    assert compare_numpy_array(uut_u, test_u), \
        "Displacement u is not equal"
    assert compare_numpy_array(uut_mech_loss, test_mech_loss), \
        "Mech loss is not equal"


def generate_data():
    """Generates the data for the tests. Assumes the current implementation of
    piezo_fem is correct."""
    dir = "tests/data"
    # test_piezo_time(dir)
    # test_piezo_time(dir)
    test_thermo_piezo_time(dir)


if __name__ == "__main__":
    generate_data()
