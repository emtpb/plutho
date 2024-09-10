import os
import gmsh
import numpy as np

import piezo_fem as pfem

class GmshHandler:
    def __init__(self, mesh_file_path):
        if not gmsh.isInitialized():
            gmsh.initialize()

        self.mesh_file_path = mesh_file_path
        self.output_file_path = os.path.splitext(mesh_file_path)[0] + "_results.msh"

    @staticmethod
    def _get_nodes(node_tags, node_coords, _ = None):
        # Returns a list of node coords: [[x1, y1], [x2, y2], ..]
        nodes = np.zeros(shape=(len(node_tags), 2))
        for i, _ in enumerate(node_tags):
            nodes[i] = (node_coords[3*i], node_coords[3*i+1])

        return nodes

    @staticmethod
    def _get_elements(element_types, element_tags, element_node_tags):
        # Get all triangle elements
        elements = np.zeros(shape=(len(element_tags[1]), 3), dtype=int)
        for i, element_type in enumerate(element_types):
            if element_type == 2:
                # Only looking for 3-node triangle elements
                current_element_tags = element_tags[i]
                current_node_tags = element_node_tags[i]
                for j, _ in enumerate(current_element_tags):
                    elements[j] = current_node_tags[3*j:3*(j+1)] - np.ones(3)

        return elements

    def get_mesh_nodes_and_elements(self):
        nodes = GmshHandler._get_nodes(*gmsh.model.mesh.getNodes())
        elements = GmshHandler._get_elements(*gmsh.model.mesh.getElements())

        return nodes, elements

    def get_nodes_by_physical_groups(self, needed_pg_names):
        # Get all possible physical group names
        possible_pg_tags = gmsh.model.getPhysicalGroups(1)
        possible_pg = {}
        for dim, tag in possible_pg_tags:
            physical_name = gmsh.model.getPhysicalName(1, tag)
            possible_pg[physical_name] = tag

        # Get physical group tags according to the given names
        pg_tags = {}
        for name in needed_pg_names:
            if name in possible_pg.keys():
                node_tags, node_coords = gmsh.model.mesh.getNodesForPhysicalGroup(1, possible_pg[name])
                pg_tags[name] = (node_tags - np.ones(len(node_tags))).astype(int)

        return pg_tags

    def get_elements_by_physical_groups(self, needed_pg_names):
        pg_tags = self.get_nodes_by_physical_groups(needed_pg_names)

        triangle_elements = {}
        for pg_name, elements in pg_tags.items():
            current_triangle_elements = []
            for check_element in GmshHandler._get_elements(*gmsh.model.mesh.getElements()):
                found_count = 0

                if check_element[0] in elements:
                    found_count += 1
                if check_element[1] in elements:
                    found_count += 1
                if check_element[2] in elements:
                    found_count += 1

                if found_count > 1:
                    current_triangle_elements.append(check_element)
        
            triangle_elements[pg_name] = np.array(current_triangle_elements, dtype=int)

        return triangle_elements

    def create_power_loss_post_processing_view(self, power_loss, number_of_time_steps, delta_t):
        element_tags = gmsh.model.mesh.getElements(2)[1][0]
        number_of_elements = len(element_tags)
        view_tag = gmsh.view.add("Power Loss in W")

        for time_index in range(number_of_time_steps//10):
            current_power_loss = power_loss[:, time_index].reshape(number_of_elements, 1)
            gmsh.view.addModelData(view_tag, time_index, "", "ElementData", element_tags, current_power_loss, time_index*delta_t, 1)

        gmsh.view.write(view_tag, self.output_file_path, True)

    def create_post_processing_views(self, field, number_of_time_steps, delta_t):
        node_tags, _, _ = gmsh.model.mesh.getNodes() 
        number_of_nodes = len(node_tags)

        # Views
        u_view_tag = gmsh.view.add(f"u")
        #u_r_view_tag = gmsh.view.add(f"u_r")
        #u_z_view_tag = gmsh.view.add(f"u_z")
        v_view_tag = gmsh.view.add(f"Voltage in V")
        theta_view_tag = gmsh.view.add(f"Temperature in °C")

        for time_index in range(number_of_time_steps):
            current_u = [[field[2*i, time_index], field[2*i+1, time_index], 0] for i in range(number_of_nodes)]
            #current_u_r = field[:2*number_of_nodes:2, time_index].reshape(number_of_nodes, 1)
            #current_u_z = field[1:2*number_of_nodes:2, time_index].reshape(number_of_nodes, 1)t
            current_v = field[2*number_of_nodes:3*number_of_nodes, time_index].reshape(number_of_nodes, 1)
            current_theta = field[3*number_of_nodes:, time_index].reshape(number_of_nodes, 1)

            gmsh.view.addModelData(u_view_tag, time_index, "", "NodeData", node_tags, current_u, time_index*delta_t, 3)
            #gmsh.view.addModelData(u_r_view_tag, time_index, "", "NodeData", node_tags, current_u_r, time_index*delta_t, 1)
            #gmsh.view.addModelData(u_z_view_tag, time_index, "", "NodeData", node_tags, current_u_z, time_index*delta_t, 1)
            gmsh.view.addModelData(v_view_tag, time_index, "", "NodeData", node_tags, current_v, time_index*delta_t, 1)
            gmsh.view.addModelData(theta_view_tag, time_index, "", "NodeData", node_tags, current_theta, time_index*delta_t, 1)

        gmsh.write(self.output_file_path)
        gmsh.view.write(u_view_tag, self.output_file_path, True)
        #gmsh.view.write(u_r_view_tag, self.output_file_path, True)
        #gmsh.view.write(u_z_view_tag, self.output_file_path, True)
        gmsh.view.write(v_view_tag, self.output_file_path, True)
        gmsh.view.write(theta_view_tag, self.output_file_path, True)    

    @staticmethod
    def get_electrode_triangles(electrode_elements, all_elements):
        # TODO Can this be removed?
        triangle_elements = []
        for element in electrode_elements:
            for check_element in all_elements:
                if element[0] in check_element and element[1] in check_element:
                    triangle_elements.append(check_element)
                    break

        return triangle_elements

    def generate_rectangular_mesh(self, width=0.005, height=0.001, mesh_size=0.00015):
        gmsh.clear()
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
        gmsh.write(self.mesh_file_path)

if __name__ == "__main__":
    data_directory = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "data")
    gmsh_file = os.path.join(data_directory, "mesh.msh")
    """
    nodes, elements = get_mesh_nodes_and_elements(gmsh_file)
    phyiscal_groups = get_elements_by_physical_groups(gmsh_file, ["Electrode", "Ground", "Symaxis"])
    electrode_triangles = phyiscal_groups["Electrode"]

    parser = pfem.mesh.GmshParser(gmsh_file)

    # Get nodes and elements from parser
    nodes_old = parser.nodes.copy()
    elements_old = parser.getTriangleElements()
    electrode_elements_old = parser.getElementsInPhysicalGroup("Electrode")
    electrode_triangles_old = get_electrode_triangles(electrode_elements_old, elements_old)
    """
    TIME_STEP_COUNT = 100
    DELTA_T = 1e-8
    u = np.load(os.path.join(data_directory, "displacement.npy"))
    #q = np.load(os.path.join(data_directory, "charge.npy"))
    number_of_nodes = 318
    print(u.shape)
    #u_r = u[:2*number_of_nodes:2]
    create_post_processing_views(gmsh_file, u, TIME_STEP_COUNT, DELTA_T)