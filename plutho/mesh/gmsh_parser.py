"""Module for loading the meshes using the gmsh python interface."""

# Python standard libraries
import os
from typing import Tuple, Dict, List

# Third party libraries
import gmsh
import numpy as np
import numpy.typing as npt


# First index represents the element type:
# 1: Line
# 2: Triangle
# Second index represents the element order
element_to_nodes_map = np.array([
    [-1, -1, -1, -1],
    [-1, 2, 3, 4],
    [-1, 3, 6, 10]
])

# Since its the same for lines and triangles, this map only has one row
# and the index represents the element order
element_to_boundary_nodes_map = np.array([-1, 2, 3, 4])


class GmshParser:
    """Class to use the gmsh python interface to read the mesh files.
    """
    element_order: int

    def __init__(self, file_path, element_order):
        if not gmsh.is_initialized():
            gmsh.initialize()

        if os.path.isfile(file_path):
            gmsh.open(file_path)
        else:
            raise IOError(f"Mesh file {file_path} not found.")

        self.element_order = element_order

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

    def _get_elements(
        self,
        element_types,
        element_tags,
        element_node_tags
    ) -> Dict[int, npt.NDArray]:
        """
        For each given element type a list of elements is returned.

        Parameters:
            element_types: List of element types for which the elements lists
                are extracted.
            element_tags: Tags of the elements per element type.
            element_node_tags: Tags of the nodes of the elements per element
                type.

        Returns:
            A list of elements for each given element type:
            List of elements [e1, e2, e3, ...] where each element consists of
            its node tags: e1: [n1, n2, ...].
        """
        elements_per_type = {}

        # Types of the given elements: 2D -> Triangle or 1D -> Line
        # Iterate over all given types
        for i, element_type in enumerate(element_types):
            elements = []
            element_indices = []

            # For each type get the order to calculate the number of nodes
            # it consists of
            _, dim, order, _, _, _ = \
                gmsh.model.mesh.getElementProperties(
                    element_type
                )

            match dim:
                case 0:
                    nodes_per_element = 1
                case 1:
                    # Line element
                    nodes_per_element = order+1
                case 2:
                    # Triangle element
                    nodes_per_element = int(1/2*(order+1)*(order+2))
                case _:
                    raise ValueError("Only supporting 2D meshes")

            # The element tags and element_node_tags lists are nested lists
            # which contain the element tags and node tags of the elements of
            # the current type at the respective index
            current_element_tags = element_tags[i]
            current_node_tags = element_node_tags[i]

            # Now iterate over every element of the current type and extract
            # the node indices
            for j, _ in enumerate(current_element_tags):
                # 1 is subtracted because the indices in gmsh start with 1.
                elements.append(current_node_tags[
                        nodes_per_element*j:nodes_per_element*(j+1)
                    ] - np.ones(nodes_per_element)
                )
                element_indices.append(j)

            if len(elements) == 0:
                raise ValueError(
                    "Couldn't return elements because the list is empty."
                )

            elements_np = np.zeros(
                shape=(len(current_element_tags), nodes_per_element),
                dtype=int
            )
            for index, element in zip(element_indices, elements):
                elements_np[index] = element

            elements_per_type[element_type] = elements_np

        return elements_per_type

    def get_mesh_nodes_and_elements(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """Creates the nodes and elements lists as used in the simulation.

        Returns:
            List of nodes and elements"""
        nodes = self._get_nodes(*gmsh.model.mesh.getNodes())

        el_types, el_tags, el_node_tags = gmsh.model.mesh.getElements(dim=2)
        elements_per_type = self._get_elements(
            el_types, el_tags, el_node_tags
        )

        if len(elements_per_type) != 1:
            raise ValueError(
                "The given mesh as more (or less) than one element type for "
                "dim=2"
            )

        # There should be only one element type for dim=2
        elements = elements_per_type[el_types[0]]

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
        """Get all triangle elements for the given physical group.

        Parameters:
            needed_pg_names: List of names of physical groups for which the
                triangle elements shall be returned.

        Returns:
            dictionary where the key is the physical group name and the value
            is a list of elements for this pg.
        """
        _, elements = self.get_mesh_nodes_and_elements()
        pg_nodes = self.get_nodes_by_physical_groups(needed_pg_names)
        element_order = self.element_order
        nodes_per_line = element_order+1
        pg_elements = {}

        for pg_name in needed_pg_names:
            nodes = pg_nodes[pg_name]

            # For every possible element, check if it contains 'enought' nodes
            # from the physical group -> Then add it to the elements list
            # TODO Rework this: Check if its also possible to return
            # element lists containing line elements.
            current_elements = []
            for element in elements:
                count = 0
                for node_index in element:
                    if node_index in nodes:
                        count += 1

                if count >= nodes_per_line:
                    current_elements.append(element)

            pg_elements[pg_name] = np.array(current_elements)

        return pg_elements

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
