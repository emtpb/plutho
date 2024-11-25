"""Module for the mesh generation using gmsh."""

# Python standard libraries
from typing import Tuple, List, Dict
import os
import gmsh
import numpy as np
import numpy.typing as npt

from .simulation.base import ModelType


class GmshHandler:
    """Class to handle the mesh generation and the accessing of the nodes
    and elements of the mesh.

    Attributes:
        width: Width of the model.
        height: Height of the model.
        mesh_size: Mesh size of the mesh.
        x_offset: Also called inner radius of the model. How far the model
            is offset along the x direction.
        model_type: Type of the mesh model.
    """

    width: float
    height: float
    mesh_size: float
    x_offset: float
    model_type: ModelType

    def __init__(self, mesh_file_path: str, load_mesh: bool = False):
        """Constructor for the GmshHandler. Given the mesh file path a *.msh
        file is created. If post processing views are added the results are
        saved in a different file with the same path as mesh file path but
        with _results.msh appended.

        Parameters:
            mesh_file_path: Path where the mesh file will be saved.
            load_mesh: Set to true if there is already a mesh and it shall be
                opened in gmsh.
        """
        if not gmsh.isInitialized():
            gmsh.initialize()

        self.mesh_file_path = mesh_file_path
        self.output_file_path = os.path.splitext(mesh_file_path)[0] + \
            "_results.msh"

        if load_mesh:
            gmsh.open(self.mesh_file_path)

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

    def get_mesh_nodes_and_elements(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """Creates the nodes and elements lists as used in the simulation.

        Returns:
            List of nodes and elements"""
        nodes = GmshHandler._get_nodes(*gmsh.model.mesh.getNodes())
        elements = GmshHandler._get_elements(*gmsh.model.mesh.getElements())

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
        elements = GmshHandler._get_elements(*gmsh.model.mesh.getElements())
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

    def create_element_post_processing_view(
            self,
            field: npt.NDArray,
            number_of_time_steps: int,
            delta_t: float,
            field_dimension: int,
            field_name: str,
            append: bool):
        """Appends a post processing view to the output mesh file.
        The given field must be scalar and defined over the elements.

        Parameters:
            field: Field quantity [element_index, time_index].
            number_of_time_steps: Total number of time steps.
            delta_t: Time difference between each time step.
            field_dimension: 1 for scalar fields and 2 for vector fields
                with 2 components.
            field_name: Description of the field in gmsh.
            append: Set to true if this field should be appended to the file.
        """
        element_tags = gmsh.model.mesh.getElements(2)[1][0]
        number_of_elements = len(element_tags)
        view_tag = gmsh.view.add(field_name)

        for time_index in range(number_of_time_steps):
            if field_dimension != 2:
                current_field_value = field[:, time_index].reshape(
                    number_of_elements, field_dimension)
            else:
                # Since the components have 2 components but gmsh can only take
                # 3.
                current_field_value = np.array(
                    [
                        [field[2*i, time_index],
                         field[2*i+1, time_index], 0] for i in range(
                         number_of_elements)
                    ]
                )
            gmsh.view.addModelData(
                view_tag,
                time_index,
                "",
                "ElementData",
                element_tags,
                current_field_value,
                time_index*delta_t,
                field_dimension if field_dimension != 2 else 3)

        if not os.path.exists(self.output_file_path):
            gmsh.write(self.output_file_path)
        gmsh.view.write(view_tag, self.output_file_path, append)

    def create_u_default_post_processing_view(
            self,
            u: npt.NDArray,
            number_of_time_steps: int,
            delta_t: float,
            temperature_field: bool,
            append: bool):
        """Appends a post processing view to the output mesh file.
        The given field must be scalar and defined over the nodes.

        Parameters:
            u: u field given from the simulation.
            number_of_time_steps: Total number of time steps.
            delta_t: Time difference between each time step.
            temperature_field: True of u contains temperature field,
                else false.
            append: Set to true if this field should be appended to the file.
        """
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        number_of_nodes = len(node_tags)

        # Views
        u_view_tag = gmsh.view.add("Displacement")
        v_view_tag = gmsh.view.add("Potential")
        if temperature_field:
            theta_view_tag = gmsh.view.add("Temperature")

        for time_index in range(number_of_time_steps):
            current_u = [
                [u[2*i, time_index],
                 u[2*i+1, time_index], 0] for i in range(number_of_nodes)]
            current_v = u[2*number_of_nodes:3*number_of_nodes, time_index] \
                .reshape(number_of_nodes, 1)
            gmsh.view.addModelData(
                u_view_tag,
                time_index,
                "",
                "NodeData",
                node_tags,
                current_u,
                time_index*delta_t,
                3)
            gmsh.view.addModelData(
                v_view_tag,
                time_index,
                "",
                "NodeData",
                node_tags,
                current_v,
                time_index*delta_t,
                1)

            if temperature_field:
                current_theta = u[3*number_of_nodes:, time_index] \
                    .reshape(number_of_nodes, 1)
                gmsh.view.addModelData(
                    theta_view_tag,
                    time_index,
                    "",
                    "NodeData",
                    node_tags,
                    current_theta,
                    time_index*delta_t,
                    1)
        if not os.path.exists(self.output_file_path):
            gmsh.write(self.output_file_path)
        gmsh.view.write(u_view_tag, self.output_file_path, append)
        gmsh.view.write(v_view_tag, self.output_file_path, True)
        if temperature_field:
            gmsh.view.write(theta_view_tag, self.output_file_path, True)

    def create_theta_post_processing_view(
            self,
            theta: npt.NDArray,
            number_of_time_steps: int,
            delta_t: float,
            append: bool):
        """Appends a post processing view to the output mesh file.
        The given field must be scalar and defined over the nodes.

        Parameters:
            theta: theta field given from the simulation.
            number_of_time_steps: Total number of time steps.
            delta_t: Time difference between each time step.
            append: Set to true if this field should be appended to the file.
        """
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        number_of_nodes = len(node_tags)

        # Views
        theta_view_tag = gmsh.view.add("Temperature")

        for time_index in range(number_of_time_steps):
            current_theta = theta[:, time_index].reshape(number_of_nodes, 1)
            gmsh.view.addModelData(
                theta_view_tag,
                time_index,
                "",
                "NodeData",
                node_tags,
                current_theta,
                time_index*delta_t,
                1)
        if not os.path.exists(self.output_file_path):
            gmsh.write(self.output_file_path)
        gmsh.view.write(theta_view_tag, self.output_file_path, append)

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
        # Saves values for serialization and deserialization
        self.width = width
        self.height = height
        self.mesh_size = mesh_size
        self.x_offset = x_offset

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

    def as_dict(self) -> Dict[str, str]:
        """Returns the important mesh parameters as dictionary in order
        to save it to a file.

        Returns:
            The dictionary containing the values."""
        return {
            "width": str(self.width),
            "height": str(self.height),
            "mesh_size": str(self.mesh_size),
            "x_offset": str(self.x_offset),
            "model_type": self.model_type.value
        }
