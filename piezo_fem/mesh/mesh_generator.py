"""Module to generate different gmsh mesh files."""

# Third party libraries
import gmsh

def generate_rectangular_mesh(mesh_file_path, width=0.005, height=0.001, mesh_size=0.00015):
    if not gmsh.isInitialized():
        gmsh.initialize()

    corner_points = [[0, 0], [width, 0], [width, height], [0, height]]

    gmsh_point_indices = []
    for point in corner_points:
        gmsh_point_indices.append(gmsh.model.geo.addPoint(point[0], point[1], 0, mesh_size)) # 0.00005

    bottom_line = gmsh.model.geo.addLine(gmsh_point_indices[0], gmsh_point_indices[1])
    right_line = gmsh.model.geo.addLine(gmsh_point_indices[1], gmsh_point_indices[2])
    top_line = gmsh.model.geo.addLine(gmsh_point_indices[2], gmsh_point_indices[3])
    left_line = gmsh.model.geo.addLine(gmsh_point_indices[3], gmsh_point_indices[0])

    curve_loop = gmsh.model.geo.addCurveLoop([bottom_line, right_line, top_line, left_line])
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    boundary_top = gmsh.model.geo.addPhysicalGroup(1, [top_line])
    gmsh.model.setPhysicalName(1, boundary_top, "Electrode")
    boundary_left = gmsh.model.geo.addPhysicalGroup(1, [left_line])
    gmsh.model.setPhysicalName(1, boundary_left, "Symaxis")
    boundary_bottom = gmsh.model.geo.addPhysicalGroup(1, [bottom_line])
    gmsh.model.setPhysicalName(1, boundary_bottom, "Ground")

    all = gmsh.model.geo.addPhysicalGroup(2, [surface])
    gmsh.model.setPhysicalName(2, all, "Surface")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write(mesh_file_path)