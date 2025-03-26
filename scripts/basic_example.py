"""Implements examples on how to use the piezo_fem package."""

# Python standard libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# Local libraries
import piezo_fem as pfem


def load_mesh(mesh_file_path):
    """Loads a mesh file. Creates a default disc mesh it if it does not exists

    Parameters:
        mesh_file_path: Path of the mesh file.
    """
    if os.path.exists(mesh_file_path):
        mesh = pfem.Mesh(mesh_file_path, load=True)
    else:
        mesh = pfem.Mesh(mesh_file_path, load=False)

        # When there is no mesh file a new mesh is generated.
        # mesh_size is the maximum possible distance betweeen two points of the
        # same triangle in the mesh.
        mesh.generate_rectangular_mesh(
            width=0.005,
            height=0.001,
            mesh_size=0.00004,
            x_offset=0
        )

    return mesh


def simulate_piezo_impedance(base_directory, show_results):
    """Example function on how to run a piezo simulation which calculates
    the charge and calculates the impedance of the model. The simulation is
    done in the frequency domain.

    Parameters:
        base_directory: Directory in which a new simulation directory is
            created in which the simulation files are added.
        show_results: Set to True of the resulting impedance shall be plotted.
    """
    # First load or create a mesh.
    # Since the mesh is used in multiple simulations it is saved in the base
    # directory.
    mesh = load_mesh(os.path.join(base_directory, "disc_mesh.msh"))

    # Create the simulation
    sim = pfem.SingleSimulation(
        # Folder IN which the simulation folder is created
        working_directory=base_directory,
        # Name of the simulation & sim folder name
        simulation_name="pic255_impedance_example",
        # Mesh which is used in the simulation
        mesh=mesh
    )

    # Frequencies for which the simulation is done
    frequencies = np.linspace(0, 1e7, 1000)[1:]
    # Set a frequency domain piezo simulation
    sim.setup_piezo_freq_domain(frequencies)

    # Add a material to the model and specify for which elements this element
    # is used.
    sim.add_material(
        # Name of the material
        material_name="pic255",
        # This is a predefined material
        material_data=pfem.pic255,
        # Since this is None the material will be applied everywhere
        physical_group_name=None
    )

    # Setup boundary conditions
    # First two boundary conditions for PHI (electrical potenzial) are added
    # those implement ground (0 V potential) on the bottom of the ceramic
    # as well as an arbitrary excitation potential at the top.
    sim.add_dirichlet_bc(
        # Name of the field for which the bc is
        field_type=pfem.FieldType.PHI,
        # Name of the physical group on which this bc is applied.
        # In this case this is a line segment of the model.
        physical_group_name="Electrode",
        # Values for each simulation step which are equally applied along the
        # model segment.
        values=np.ones(len(frequencies))
    )
    sim.add_dirichlet_bc(
        field_type=pfem.FieldType.PHI,
        physical_group_name="Ground",
        values=np.zeros(len(frequencies))
    )

    # Additionaly since a disc is simulated which left side is at r=0 an
    # symmetry boundary condition is needed.
    sim.add_dirichlet_bc(
        field_type=pfem.FieldType.U_R,
        physical_group_name="Symaxis",
        values=np.zeros(len(frequencies))
    )

    # Since the charge shall be simulated in order to calculate the impedance
    # it is necesarry to determine the elements for which a charge is
    # calculated
    electrode_elements = mesh.get_elements_by_physical_groups(
        ["Electrode"]
    )["Electrode"]

    # Now the simulation can be done
    sim.simulate(
        electrode_elements=electrode_elements
    )

    # The simulation settings and results are saved in the simulation folder
    sim.save_simulation_settings()
    sim.save_simulation_results()

    # Calculate impedance
    # The charge can be accessed using sim.solver
    charge = sim.solver.q

    # Since the simulation is already in the frequency domain the impedance
    # can be calculated directly
    impedance = np.abs(1/(1j*2*np.pi*frequencies*charge))

    if show_results:
        # Plot results
        plt.plot(
            frequencies/1e6,
            impedance,
            label="Impedance PIC255"
        )
        plt.xlabel("Frequency $f$ / MHz")
        plt.ylabel("Impedance $|Z|$ / $\\Omega$")
        plt.legend()
        plt.grid()
        plt.show()


def simulate_thermo_piezo(base_directory):
    """Runs a thermo-piezoeletric simulation and calculates the thermal
    field after 20 microseconds.

    Parameters:
        base_directory: Directory in which the simulation folder is created.
    """
    mesh = load_mesh(os.path.join(base_directory, "disc_mesh.msh"))

    sim = pfem.SingleSimulation(
        base_directory,
        "thermo_piezo_sim_20k",
        mesh
    )
    # Simulation parameters
    NUMBER_OF_TIME_STEPS = 20000
    DELTA_T = 1e-8

    # Time integration parameters: Those values make sure that the simulation
    # is unconditionally stable.
    GAMMA = 0.5
    BETA = 0.25

    sim.setup_thermo_piezo_time_domain(
        NUMBER_OF_TIME_STEPS,
        DELTA_T,
        GAMMA,
        BETA
    )

    sim.add_material(
        "pic255",
        pfem.pic255,
        None
    )

    # Triangular excitation
    excitation = np.zeros(NUMBER_OF_TIME_STEPS)
    excitation[1:10] = (
        1 * np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    )

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

    sim.simulate(calculate_mech_loss=True)
    sim.save_simulation_settings()
    sim.save_simulation_results()


def simulate_coupled_thermo_time(base_directory):
    """This function runs a thermo-piezoelectric simulation with a coupled
    thermo simulation. First the thermo-piezoelectric simulation is run until
    the mech losses are stationary. Then the constant mech losses are
    calculated and used as a source in a thermo simulation which is run
    afterwards

    Parameters:
        base_directory: Directory in which the simulation folder is created
    """
    mesh = load_mesh(os.path.join(base_directory, "disc_mesh.msh"))

    PIEZO_DELTA_T = 1e-8
    PIEZO_NUMBER_OF_TIME_STEPS = 20000

    THERMO_DELTA_T = 0.001
    THERMO_NUMBER_OF_TIME_STEPS = 1000

    piezo_sim_data = pfem.SimulationData(
        PIEZO_DELTA_T,
        PIEZO_NUMBER_OF_TIME_STEPS,
        0.5,
        0.25
    )
    thermo_sim_data = pfem.SimulationData(
        THERMO_DELTA_T,
        THERMO_NUMBER_OF_TIME_STEPS,
        0.5,
        0  # Not needed in thermo simulation
    )

    coupled_sim = pfem.CoupledThermoPiezoThermoSim(
        base_directory,
        "CoupledThermoPiezoelectricSim",
        piezo_sim_data,
        thermo_sim_data,
        mesh
    )

    coupled_sim.add_material(
        "pic255",
        pfem.pic255,
        None
    )

    # Excitation for piezoelectric simulation
    AMPLITUDE = 20
    FREQUENCY = 2e6
    time_values = np.arange(PIEZO_NUMBER_OF_TIME_STEPS)*PIEZO_DELTA_T
    coupled_sim.set_excitation(
        excitation=AMPLITUDE*np.sin(2*np.pi*FREQUENCY*time_values),
        is_disc=True
    )

    coupled_sim.simulate()
    coupled_sim.save_simulation_settings()
    coupled_sim.save_simulation_results()


def simulate_coupled_thermo_freq(base_directory):
    """This function implements a thermo-piezoelectric simulation in the
    frequency domain. The calculated mech losses are embed in a time domain
    thermo simulation.

    Parameters:
        base_directory: Directory in which the simulation folder is created
    """
    mesh = load_mesh(os.path.join(base_directory, "disc_mesh.msh"))

    THERMO_DELTA_T = 0.001
    THERMO_NUMBER_OF_TIME_STEPS = 1000
    FREQUENCY = 2.07e6
    AMPLITUDE = 1.3

    thermo_sim_data = pfem.SimulationData(
        THERMO_DELTA_T,
        THERMO_NUMBER_OF_TIME_STEPS,
        0.5,
        0  # Not needed in thermo simulation
    )

    coupled_sim = pfem.CoupledFreqPiezoTherm(
        base_directory,
        "CoupledThermopiezoelectricFreqSim",
        FREQUENCY,
        thermo_sim_data,
        mesh
    )

    coupled_sim.add_material(
        "pic184",
        pfem.materials.pic184_25,
        None
    )

    coupled_sim.set_excitation(
        excitation=AMPLITUDE,
        is_disc=True
    )

    coupled_sim.simulate(
        material_starting_temperature=25,
        temperature_dependent=False
    )
    coupled_sim.save_simulation_settings()
    coupled_sim.save_simulation_results()


def simulate_thermo_time(base_directory):
    mesh = load_mesh(os.path.join(base_directory, "disc_mesh.msh"))

    sim = pfem.SingleSimulation(base_directory, "ThermoTimeSim", mesh)

    DELTA_T = 0.001
    NUMBER_OF_TIME_STEPS = 1000

    _, elements = mesh.get_mesh_nodes_and_elements()
    number_of_elements = len(elements)

    sim.setup_thermo_time_domain(
        DELTA_T,
        NUMBER_OF_TIME_STEPS,
        0.5
    )

    # As an example set a constant volume heat source
    sim.solver.set_constant_volume_heat_source(
        np.ones(number_of_elements),
        NUMBER_OF_TIME_STEPS
    )

    # Set convective boundarz condition for the elements on the boundaries
    elements = mesh.get_elements_by_physical_groups(
        ["Electrode", "Ground", "RightBoundary"]
    )
    convective_boundary_elements = np.vstack(
        [
            elements["Electrode"],
            elements["Ground"],
            elements["RightBoundary"]
        ]
    )
    sim.solver.set_convection_bc(
        convective_boundary_elements,
        80,  # Heat transfer coefficient
        20  # Outer temperature
    )

    sim.simulate(initial_theta_field=25)
    sim.save_simulation_settings()
    sim.save_simulation_results()


if __name__ == "__main__":
    CWD = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "simulations"
    )

    if not os.path.isdir(CWD):
        os.makedirs(CWD)

    simulate_piezo_impedance(CWD, False)
    simulate_thermo_piezo(CWD)
    simulate_coupled_thermo_time(CWD)
    simulate_coupled_thermo_freq(CWD)
