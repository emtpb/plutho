
# Python standard libraries
import os
import time

# Thid party libraries
import numpy as np
import plutho
import json
import matplotlib.pyplot as plt


pic181 = plutho.MaterialData(
    **{
        "c11": 141218192791.12772,
        "c12": 82292914124.0279,
        "c13": 80362329625.75212,
        "c33": 137188538228.6311,
        "c44": 29848846049.816402,
        "e15": 13.216444117013664,
        "e31": -4.979419636068149,
        "e33": 14.149818966822629,
        "eps11": 1.3327347064648263e-08,
        "eps33": 5.380490373139249e-09,
        "alpha_m": 12758.833098284977,
        "alpha_k": 1.289813815258054e-10,
        "thermal_conductivity": 1.1,
        "heat_capacity": 350,
        "temperatures": 25,
        "density": 7850
    }
)


def save_mesh_data(
    file_path: str,
    mesh_data: plutho.MeshData
):
    node_indices = mesh.get_nodes_by_physical_groups([
        "Electrode", "Ground"
    ])
    dirichlet_nodes_electrode = node_indices["Electrode"]
    dirichlet_nodes_ground = node_indices["Ground"]

    content = ""
    content = {
        "nodes": mesh_data.nodes.tolist(),
        "elements": mesh_data.elements.tolist(),
        "electrode_nodes": dirichlet_nodes_electrode.tolist(),
        "ground_nodes": dirichlet_nodes_ground.tolist()
    }

    with open(file_path, "w", encoding="UTF-8") as fd:
        json.dump(content, fd, indent=2)


def simulate_hb(mesh, frequencies):
    # Simulation parameters
    HB_ORDER = 1

    nonlinearity = plutho.Nonlinearity()
    nonlinearity.set_cubic_rayleigh(0)

    # Create simulation
    sim = plutho.NLPiezoHB(
        "HB_Test",
        mesh,
        nonlinearity,
        HB_ORDER
    )

    # Set materials
    sim.add_material(
        material_name="pic181",
        material_data=pic181,
        physical_group_name=""  # Means all elements
    )

    sim.add_dirichlet_bc(
        plutho.FieldType.PHI,
        "Electrode",
        np.ones(len(frequencies))
    )
    sim.add_dirichlet_bc(
        plutho.FieldType.PHI,
        "Ground",
        np.zeros(len(frequencies))
    )

    # Run simulation
    sim.assemble()
    sim.simulate(frequencies)
    sim.calculate_charge("Electrode", is_complex=True)

    return sim.q


def simulate_plutho(mesh, frequencies):
    sim = plutho.PiezoFreq(
        simulation_name="pincic_test",
        mesh=mesh
    )

    sim.add_material(
        material_name="pic181",
        material_data=pic181,
        physical_group_name=""
    )

    sim.add_dirichlet_bc(
        field_type=plutho.FieldType.PHI,
        physical_group_name="Electrode",
        values=np.ones(len(frequencies))
    )
    sim.add_dirichlet_bc(
        field_type=plutho.FieldType.PHI,
        physical_group_name="Ground",
        values=np.zeros(len(frequencies))
    )

    sim.assemble()
    sim.simulate(
        frequencies,
        False
    )
    sim.calculate_charge("Electrode", is_complex=True)

    return sim.q


def plot_impedances(charge_hb, charge_linear, frequencies):
    impedance_hb = np.abs(1/(1j*2*np.pi*frequencies*charge_hb[0, :]))
    impedance_linear = np.abs(1/(1j*2*np.pi*frequencies*charge_linear))
    plt.plot(frequencies/1e6, impedance_hb, label="NLPiezoHB")
    plt.plot(frequencies/1e6, impedance_linear, "--", label="PiezoFreq")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    CWD = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "simulations"
    )

    if not os.path.isdir(CWD):
        os.makedirs(CWD)

    # Settings
    element_order = 1
    frequencies = np.linspace(0, 1e7, 1000)[1:]

    # Load/create ring mesh
    mesh_file = os.path.join(CWD, "mesh.msh")
    plutho.Mesh.generate_rectangular_mesh(
        mesh_file,
        width=0.00635,
        height=0.001,
        x_offset=0.0026,
        mesh_size=0.0001,
        element_order=element_order
    )

    mesh = plutho.Mesh(mesh_file, element_order)

    start_time = time.time()
    charge_hb = simulate_hb(mesh, frequencies)
    hb_time = time.time() - start_time

    start_time = time.time()
    charge_piezo_freq = simulate_plutho(mesh, frequencies)
    plutho_time = time.time() - start_time

    print(f"harmonic balancing took {hb_time}s to simulate.")
    print(f"piezo freq took {plutho_time}s to simulate.")

    plot_impedances(charge_hb, charge_piezo_freq, frequencies)   plt.show()
