import plutho

if __name__ == "__main__":
    path = "test_disc.msh"
    plutho.Mesh.generate_rectangular_mesh(
        path,
        width=0.005,
        height=0.001,
        x_offset=0.0,
        mesh_size=0.00015,
        element_order=1
    )
