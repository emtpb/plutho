"""Module for loading the meshes using the gmsh python interface."""

# Python standard libraries
import os
from typing import Tuple, Dict, List

# Third party libraries
import gmsh
import numpy as np
import numpy.typing as npt

class GmshParser:
    """Class to use the gmsh python interface to read the mesh files.
    """

    def __init__(self, file_path):
        if not gmsh.is_initialized():
            gmsh.initialize()

        if os.path.isfile(file_path):
            gmsh.open(file_path)
        else:
            raise IOError(f"Mesh file {file_path} not found.")

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
        for i, node_tag in enumerate(node_tags):
            nodes[node_tag-1] = (node_coords[3*i], node_coords[3*i+1])

        return nodes

    @staticmethod
    def _get_elements(
        element_types,
        element_tags,
        element_node_tags
    ) -> npt.NDArray:
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
        nodes = self._get_nodes(*gmsh.model.mesh.getNodes())
        elements = self._get_elements(*gmsh.model.mesh.getElements())

        return nodes, elements

    def get_nodes_by_physical_groups(
        self,
        needed_pg_names: List[str]
    ) -> Dict[str, npt.NDArray]:
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
        needed_pg_names: List[str]
    ) -> Dict[str, npt.NDArray]:
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

    def create_element_post_processing_view(
        self,
        output_file_path: str,
        field: npt.NDArray,
        number_of_time_steps: int,
        delta_t: float,
        field_dimension: int,
        field_name: str,
        append: bool
    ):
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

        if not os.path.exists(output_file_path):
            gmsh.write(output_file_path)
        gmsh.view.write(view_tag, output_file_path, append)

    def create_u_default_post_processing_view(
        self,
        output_file_path: str,
        u: npt.NDArray,
        number_of_time_steps: int,
        delta_t: float,
        temperature_field: bool,
        append: bool
    ):
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
        if not os.path.exists(output_file_path):
            gmsh.write(output_file_path)
        gmsh.view.write(u_view_tag, output_file_path, append)
        gmsh.view.write(v_view_tag, output_file_path, True)
        if temperature_field:
            gmsh.view.write(theta_view_tag, output_file_path, True)

    def create_theta_post_processing_view(
        self,
        output_file_path: str,
        theta: npt.NDArray,
        number_of_time_steps: int,
        delta_t: float,
        append: bool
    ):
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
        if not os.path.exists(output_file_path):
            gmsh.write(output_file_path)
        gmsh.view.write(theta_view_tag, output_file_path, append)
