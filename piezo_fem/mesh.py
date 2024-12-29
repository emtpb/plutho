
# Python standard libraries
import os
from typing import List, Dict, Tuple

# Third party libraries
import numpy as np
import numpy.typing as npt
import gmsh


class Mesh:

    mesh_file_path: str

    def __init__(
            self,
            file_path: str,
            load: bool = False):
        if not gmsh.is_initialized():
            gmsh.initialize()

        self.mesh_file_path = file_path
        if os.path.isfile(file_path):
            if load:
                gmsh.open(file_path)
            else:
                raise IOError(
                    "Mesh file already exists. If the file shall be loaded,"
                    "set the load parameter to True."
                )

    @staticmethod
    def _get_nodes(node_tags, node_coords, _=None) -> npt.NDArray:
        """Internal function to return a list of the node coordinates based
        on the node tags and node coords lists given by the gmsh api.

        Parameters:
            node_tags: Node tags given from gmsh api.
            node_coords: Node coords given from gmsh api.
            _: Unnecessary parameter but when calling gmsh...getNodes()
                there are 3 parameters given this parameter makes it possible
                to just unwarp getNodes() and give it to this function.
        Returns:
            One list used in the simulations containing the node coords like
            [[x1, y1], [x2, y2], ...]
        """
        nodes = np.zeros(shape=(len(node_tags), 2))
        for i, _ in enumerate(node_tags):
            nodes[i] = (node_coords[3*i], node_coords[3*i+1])

        return nodes

    @staticmethod
    def _get_elements(
            element_types,
            element_tags,
            element_node_tags) -> npt.NDArray:
        """Returns a list of elements used in the simulation from the gmsh
        getElements() api function.

        Parameters:
            element_types: Element types from gmsh api. Similar to the element
                dimension so 2 equals triangles.
            element_tags: Tags of the element per element type.
            element_node_tags: Tags of the nodes per element tag.

        Returns:
            A list of lists where each inner list contains the node indices for
            the triangle at the outer list index.
        """
        elements = np.zeros(shape=(len(element_tags[1]), 3), dtype=int)
        for i, element_type in enumerate(element_types):
            if element_type == 2:
                # Only looking for 3-node triangle elements
                current_element_tags = element_tags[i]
                current_node_tags = element_node_tags[i]
                for j, _ in enumerate(current_element_tags):
                    # 1 is subtracted because the indices in gmsh start with 1.
                    elements[j] = current_node_tags[3*j:3*(j+1)] - np.ones(3)

        return elements

    def generate_rectangular_mesh(
            self,
            width: float = 0.005,
            height: float = 0.001,
            mesh_size: float = 0.00015,
            x_offset: float = 0):
        """Creates a gmsh rectangular mesh given the width, height, the mesh
        size and the x_offset.

        Parameters:
            width: With of the rect in m.
            height: Height of the rect in m.
            mesh_size: Mesh size of the mesh. Equal to the maximum distance
                between two point in the mesh.
            x_offset: Moves the rect anlong the x-direction. Default value is
                0. For 0 the left side of the rect is on the y-axis.
        """
        gmsh.clear()
        corner_points = [[x_offset, 0],
                         [width+x_offset, 0],
                         [width+x_offset, height],
                         [x_offset, height]]

        gmsh_point_indices = []
        for point in corner_points:
            gmsh_point_indices.append(gmsh.model.geo.addPoint(
                point[0], point[1], 0, mesh_size))

        bottom_line = gmsh.model.geo.addLine(
            gmsh_point_indices[0],
            gmsh_point_indices[1])
        right_line = gmsh.model.geo.addLine(
            gmsh_point_indices[1],
            gmsh_point_indices[2])
        top_line = gmsh.model.geo.addLine(
            gmsh_point_indices[2],
            gmsh_point_indices[3])
        left_line = gmsh.model.geo.addLine(
            gmsh_point_indices[3],
            gmsh_point_indices[0])

        curve_loop = gmsh.model.geo.addCurveLoop(
            [bottom_line, right_line, top_line, left_line])
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])

        boundary_top = gmsh.model.geo.addPhysicalGroup(1, [top_line])
        gmsh.model.setPhysicalName(1, boundary_top, "Electrode")
        boundary_left = gmsh.model.geo.addPhysicalGroup(1, [left_line])
        gmsh.model.setPhysicalName(1, boundary_left, "Symaxis")
        boundary_bottom = gmsh.model.geo.addPhysicalGroup(1, [bottom_line])
        gmsh.model.setPhysicalName(1, boundary_bottom, "Ground")

        model = gmsh.model.geo.addPhysicalGroup(2, [surface])
        gmsh.model.setPhysicalName(2, model, "Surface")

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write(self.mesh_file_path)

    def get_mesh_nodes_and_elements(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """Creates the nodes and elements lists as used in the simulation.

        Returns:
            List of nodes and elements"""
        nodes = Mesh._get_nodes(*gmsh.model.mesh.getNodes())
        elements = Mesh._get_elements(*gmsh.model.mesh.getElements())

        return nodes, elements

    def get_nodes_by_physical_groups(
            self,
            needed_pg_names: List[str]) -> Dict[str, List]:
        """Returns the node indices of each physical group given by
        needed_pg_names list.

        Parameters:
            needed_pg_names: List of names of physical groups for which the
                node indices shall be returned

        Returns:
            Dictionary containing the pg names as keys and the node index list
            as values."""
        # Get all possible physical group names
        possible_pg_tags = gmsh.model.getPhysicalGroups(1)
        possible_pg = {}
        for _, tag in possible_pg_tags:
            physical_name = gmsh.model.getPhysicalName(1, tag)
            possible_pg[physical_name] = tag

        # Get physical group tags according to the given names
        pg_tags = {}
        for name in needed_pg_names:
            if name in possible_pg:
                node_tags, _ = \
                    gmsh.model.mesh.getNodesForPhysicalGroup(
                        1,
                        possible_pg[name])
                pg_tags[name] = \
                    (node_tags - np.ones(len(node_tags))).astype(int)

        return pg_tags

    def get_elements_by_physical_groups(
            self,
            needed_pg_names: List[str]) -> Dict[str, npt.NDArray]:
        """Returns the elements inside the given physical groups.

        Parameters:
            needed_pg_names: List of names of physical groups for which the
                triangles are returned.

        Returns:
            Dictionary where keys are the pg names and the values are a list
            of triangles of this physical group."""
        pg_tags = self.get_nodes_by_physical_groups(needed_pg_names)
        elements = self._get_elements(*gmsh.model.mesh.getElements())
        triangle_elements = {}
        for pg_name, nodes in pg_tags.items():
            current_triangle_elements = []
            for check_element in elements:
                # If at least 2 nodes of the check_element are inside of the
                # nodes list of the current physical group, then the element
                # is also part of the physical group.
                found_count = 0

                if check_element[0] in nodes:
                    found_count += 1
                if check_element[1] in nodes:
                    found_count += 1
                if check_element[2] in nodes:
                    found_count += 1

                if found_count > 1:
                    current_triangle_elements.append(check_element)

            triangle_elements[pg_name] = np.array(
                current_triangle_elements,
                dtype=int)

        return triangle_elements
