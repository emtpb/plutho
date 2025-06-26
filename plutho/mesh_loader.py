"""Simple parser for version 2 gmsh files"""

# Python standard libraries
import os
from enum import Enum
from typing import Dict, Tuple, Union, List
from dataclasses import dataclass

# Third party libraries
import numpy as np
import numpy.typing as npt


class TokenType(Enum):
    SectionStart = "SectionStart",
    SectionEnd = "SectionEnd",
    TokenFloat = "TokenFloat",
    TokenString = "TokenString",
    TokenInt = "TokenInt"


@dataclass
class Token:
    token_type: TokenType
    value: Union[str, float, int]
    line: int


@dataclass
class MeshFormat:
    version: float
    file_type: int
    data_size: int


@dataclass
class Element:
    node_indices: List[int]
    type: int
    tags: List[int]


class MeshData:
    nodes: npt.NDArray
    elements: npt.NDArray


def tokenize(mesh_text: str) -> List[Token]:
    tokens = []
    current_index = 0
    buffer = ""
    current_line = 0

    start_section_names = [
        "$MeshFormat", "$PhysicalNames", "$Nodes",
        "$Elements"
    ]

    end_section_names = [
        "$EndMeshFormat", "$EndPhysicalNames", "$EndNodes",
        "$EndElements"
    ]

    for current_index, current_char in enumerate(mesh_text):
        current_char = mesh_text[current_index]

        if current_char.isspace():
            if current_char == "\n":
                current_line += 1

            if buffer.startswith('$'):
                # SectionStart or SectionEnd
                if buffer in start_section_names:
                    tokens.append(Token(
                        TokenType.SectionStart,
                        buffer[1:],
                        current_line
                    ))
                elif buffer in end_section_names:
                    tokens.append(
                        Token(TokenType.SectionEnd, buffer[1:], current_line)
                    )
                else:
                    raise ValueError(
                        f"Line {current_line}: Error during tokenization: "
                        "Expected SectionStart or SectionEnd but got "
                        f"{buffer}"
                    )
            elif buffer.startswith('"'):
                # String
                tokens.append(
                    Token(TokenType.TokenString, buffer[1:-1], current_line)
                )
            else:
                # Float or int
                if "." in buffer:
                    # Float
                    tokens.append(Token(
                        TokenType.TokenFloat,
                        float(buffer),
                        current_line
                    ))
                else:
                    # Int
                    tokens.append(Token(
                        TokenType.TokenInt,
                        int(buffer),
                        current_line
                    ))

            buffer = ""
        else:
            buffer += current_char

        current_index += 1

    return tokens


def expect_token(given_token, expected_type):
    if given_token.token_type is not expected_type:
        raise ValueError(
            f"Line {given_token.line}: Expected {expected_type.value} but"
            f" got {given_token.token_type.value}"
        )


class GmshData:
    mesh_format: MeshFormat
    physical_names: Dict
    nodes: npt.NDArray
    elements: List[Element]

    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise IOError(
                f"Cannot read mesh since file does not exist {file_path}"
            )

        mesh_text = ""
        with open(file_path, "r", encoding="UTF-8") as fd:
            mesh_text = fd.read()

        tokens = tokenize(mesh_text)

        # Sets the class attribues
        parser = MeshParser(tokens, self)
        parser.parse_tokens()

    def get_mesh_nodes_and_elements(self) -> Tuple[npt.NDArray, npt.NDArray]:
        # The nodes array is properly assigned but the z coordinate can be
        # dispensed
        # The elements array can be assembled by searching for all elements
        # with 3 nodes
        nodes = self.nodes[:, :2]

        triangle_elements = []
        for element in self.elements:
            if element.type == 2:
                triangle_elements.append(element)

        number_of_triangle_elements = len(triangle_elements)
        elements = np.zeros(shape=(number_of_triangle_elements, 3))

        for index, triangle_element in enumerate(triangle_elements):
            elements[index] = np.array(triangle_element.node_indices)

        return nodes, elements

    def get_nodes_by_physical_group(
        self,
        physical_group_name: str
    ) -> npt.NDArray:
        # First get the corresponding elements
        # Then extract the nodes from the elements
        elements = self.get_elements_by_physical_group(physical_group_name)

        node_indices = []
        for element in elements:
            for node_index in element:
                node_indices.append(node_index)

        # Remove duplicates  TODO Make it a set directly?
        return np.array(list(set(node_indices)), dtype=np.int64)

    def get_elements_by_physical_group(
        self,
        physical_group_name: str
    ) -> npt.NDArray:
        # Check if physical group exists
        if physical_group_name not in self.physical_names.keys():
            raise ValueError(
                f"Physical group {physical_group_name} not found in mesh."
            )

        physical_group = self.physical_names[physical_group_name]
        dimension = physical_group["dimension"]
        tag = physical_group["tag"]

        # Find elements with the corresponding dimension and tag
        found_elements = []
        for element in self.elements:
            if dimension != element.type:
                continue

            if tag in element.tags:
                found_elements.append(element.node_indices)

        if not found_elements:
            raise ValueError(
                "No elements found with given physical group "
                f"{physical_group_name}"
            )

        return np.array(found_elements)


class MeshParser:

    tokens: List[Token]
    token_index: int
    gmsh_data: GmshData

    def __init__(self, tokens: List[Token], gmsh_data: GmshData):
        self.tokens = tokens
        self.token_index = 0
        self.gmsh_data = gmsh_data

    def next_token(self):
        self.token_index += 1
        return self.tokens[self.token_index-1]

    def parse_mesh_format(self):
        token = self.next_token()
        expect_token(token, TokenType.TokenFloat)
        version = float(token.value)

        token = self.next_token()
        expect_token(token, TokenType.TokenInt)
        file_type = int(token.value)

        token = self.next_token()
        expect_token(token, TokenType.TokenInt)
        data_size = int(token.value)

        # SectionEnd token
        token = self.next_token()
        expect_token(token, TokenType.SectionEnd)

        # Write mesh format to gmsh_data object
        self.gmsh_data.mesh_format = MeshFormat(
            version,
            file_type,
            data_size
        )

    def parse_physical_names(self):
        # Get number of physical groups
        token = self.next_token()
        expect_token(token, TokenType.TokenInt)
        physical_name_count = int(token.value)

        physical_names = {}

        # For each physical group get
        for _ in range(physical_name_count):
            # Dimension
            token = self.next_token()
            expect_token(token, TokenType.TokenInt)
            dimension = int(token.value)

            # Tag
            token = self.next_token()
            expect_token(token, TokenType.TokenInt)
            tag = int(token.value)

            # Name
            token = self.next_token()
            expect_token(token, TokenType.TokenString)
            name = str(token.value)

            physical_names[name] = {
                "dimension": dimension,
                "tag": tag
            }

        # SectionEnd token
        token = self.next_token()
        expect_token(token, TokenType.SectionEnd)

        # Add to gmsh data
        self.gmsh_data.physical_names = physical_names

    def parse_nodes(self):
        # Get number of nodes
        token = self.next_token()
        expect_token(token, TokenType.TokenInt)
        number_of_nodes = int(token.value)

        nodes = np.zeros(shape=(number_of_nodes, 3))

        # Iterate over all nodes
        for _ in range(number_of_nodes):
            # Get node index
            token = self.next_token()
            expect_token(token, TokenType.TokenInt)
            # Since in gmsh the indices start at 1
            token_index = int(token.value)-1

            # Get node points
            # Can be int or float in the mesh file, but are converted to
            # float anyway
            for j in range(3):  # for x, y, z
                token = self.next_token()
                if (token.token_type is TokenType.TokenInt or
                        token.token_type is TokenType.TokenFloat):
                    nodes[token_index, j] = float(token.value)
                else:
                    raise ValueError(
                        f"Line {token.line}: Expected int or float but got"
                        f" {token.token_type.value}"
                    )

        # SectionEnd token
        token = self.next_token()
        expect_token(token, TokenType.SectionEnd)

        # Set nodes
        self.gmsh_data.nodes = nodes

    def parse_elements(self):
        # Get number of elements
        token = self.next_token()
        expect_token(token, TokenType.TokenInt)
        number_of_elements = int(token.value)

        # Dynamic list since the elements can have various amounts of nodes
        # Preallocate so the element_indices from mesh file can be used
        elements = [Element([], 0, [])]*number_of_elements

        # Now parse the elements one by one
        for _ in range(number_of_elements):
            # Get index
            token = self.next_token()
            expect_token(token, TokenType.TokenInt)
            # Since in gmsh the indices start at 1
            element_index = int(token.value)-1

            # Get type
            token = self.next_token()
            expect_token(token, TokenType.TokenInt)
            element_type = int(token.value)

            # Get number of tags
            token = self.next_token()
            expect_token(token, TokenType.TokenInt)
            number_of_tags = int(token.value)

            # Parse each tag
            tags = []
            for _ in range(number_of_tags):
                token = self.next_token()
                expect_token(token, TokenType.TokenInt)
                tags.append(int(token.value))

            # Parse nodes
            # element_type = 1 -> Line -> 2 Points
            # element_type = 2 -> Triangle -> 3 Points
            nodes = []
            for _ in range(element_type+1):
                token = self.next_token()
                expect_token(token, TokenType.TokenInt)
                # Subtract 1 because in gmsh the indices start at 1
                nodes.append(int(token.value)-1)

            elements[element_index] = Element(
                nodes,
                element_type,
                tags
            )

        # SectionEnd token
        token = self.next_token()
        expect_token(token, TokenType.SectionEnd)

        self.gmsh_data.elements = elements

    def parse_tokens(self):
        self.token_index = 0

        while self.token_index < len(self.tokens):
            current_token = self.next_token()
            expect_token(current_token, TokenType.SectionStart)

            if current_token.value == "MeshFormat":
                self.parse_mesh_format()
            elif current_token.value == "PhysicalNames":
                self.parse_physical_names()
            elif current_token.value == "Nodes":
                self.parse_nodes()
            elif current_token.value == "Elements":
                self.parse_elements()
            else:
                raise ValueError(
                    f"Line {current_token.line}: Unknown section "
                    f"type: {current_token.value}"
                )

if __name__ == "__main__":
    mesh_file_v2 = "mesh_v2.msh"
    mesh_file_v4 = "mesh_v4.msh"

    physical_groups = ["load", "ground", "axis"]

    # GMSH Parser
    import plutho
    mesh_v4 = plutho.Mesh(mesh_file_v4, True)
    nodes_v4, elements_v4 = mesh_v4.get_mesh_nodes_and_elements()
    pg_nodes_v4 = mesh_v4.get_nodes_by_physical_groups(physical_groups)
    pg_elements_v4 = mesh_v4.get_elements_by_physical_groups(physical_groups)

    # Mesh loader
    mesh_v2 = GmshData(mesh_file_v2)
    nodes_v2, elements_v2 = mesh_v2.get_mesh_nodes_and_elements()
    pg_nodes_v2 = {}
    pg_elements_v2 = {}
    for pg in physical_groups:
        pg_nodes_v2[pg] = mesh_v2.get_nodes_by_physical_group(pg)
        pg_elements_v2[pg] = mesh_v2.get_elements_by_physical_group(pg)

    if not np.array_equal(nodes_v2, nodes_v4):
        print("Nodes not equal")
    if not np.array_equal(elements_v2, elements_v4):
        print("Elements not equal")

    for pg in physical_groups:
        nodes_v2 = pg_nodes_v2[pg]
        nodes_v4 = pg_nodes_v4[pg]
        elements_v2 = pg_elements_v2[pg]
        elements_v4 = pg_elements_v4[pg]

        print("Elements v2", elements_v2)
        print("Elements v4", elements_v4)

        if not np.array_equal(nodes_v2, nodes_v4):
            print(f"({pg}) Physical group nodes not equal")
        if not np.array_equal(elements_v2, elements_v4):
            print(f"({pg}) Physical group elements not equal")
