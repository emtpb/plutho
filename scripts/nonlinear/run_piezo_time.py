
# Python standard libraries
import os

# Third party libraries
import numpy as np
from dotenv import load_dotenv

# Local libraries
import piezo_fem as pfem


if __name__ == "__main__":
    load_dotenv()

    CWD = os.getenv("piezo_fem_simulation_path")
    if CWD is None:
        print("Couldn't find simulation path.")
        exit(1)

    # Run simulation
    SIM_NAME = "nonlinear_time_test"
    sim_directory = os.path.join(CWD, SIM_NAME)

    if not os.path.isdir(sim_directory):
        os.makedirs(sim_directory)

    # Load mesh
    mesh = pfem.Mesh(
        os.path.join(CWD, "disc_mesh_0DOT0001.msh"),
        True
    )
    # mesh.generate_rectangular_mesh(mesh_size=0.001)
    nodes, elements = mesh.get_mesh_nodes_and_elements()
    number_of_nodes = len(nodes)
    number_of_elements = len(elements)

    # Basic simulation settings
    delta_t = 1e-8
    number_of_time_steps = 20
    excitation = np.zeros(number_of_time_steps)
    excitation[1:10] = (
        1 * np.array([0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2])
    )

    # Load material manager
    material_manager = pfem.MaterialManager(
        number_of_elements
    )
    material_manager.add_material(
        "pic255",
        pfem.pic255,
        None,
        np.arange(number_of_elements)
    )
    material_manager.initialize_materials()

    sim = pfem.NonlinearPiezoSim(
        pfem.MeshData(
            nodes,
            elements
        ),
        material_manager,
        pfem.SimulationData(
            delta_t=delta_t,
            number_of_time_steps=number_of_time_steps,
            gamma=None,
            beta=None
        )
    )

    # Setup boundary conditions
    dirichlet_nodes = []
    dirichlet_values = []

    node_indices = mesh.get_nodes_by_physical_groups(
        ["Electrode", "Ground", "Symaxis"]
    )

    for index in node_indices["Electrode"]:
        dirichlet_nodes.append(index+2*number_of_nodes)
        dirichlet_values.append(excitation)

    for index in node_indices["Ground"]:
        dirichlet_nodes.append(index+2*number_of_nodes)
        dirichlet_values.append(np.zeros(number_of_time_steps))

    for index in node_indices["Symaxis"]:
        dirichlet_nodes.append(2*index)
        dirichlet_values.append(np.zeros(number_of_time_steps))

    sim.dirichlet_nodes = np.array(dirichlet_nodes)
    sim.dirichlet_values = np.array(dirichlet_values)

    electrode_elements = mesh.get_elements_by_physical_groups(
        ["Electrode"]
    )["Electrode"]

    nonlinear_ela_matrix = np.zeros((4, 4))
    sim.assemble(nonlinear_ela_matrix)
    sim.solve_time(electrode_elements)

    print(sim.q)

    np.save(
        os.path.join(sim_directory, "u.npy"),
        sim.u
    )
    np.save(
        os.path.join(sim_directory, "q.npy"),
        sim.q
    )

