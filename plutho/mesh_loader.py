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
    SectionStart = "$SectionStart",
    SectionEnd = "$SectionEnd",
    TokenFloat = "TokenFloat",
    TokenString = "TokenString",
    TokenInt = "TokenInt"


@dataclass
class Token:
    token_type: TokenType
    value: Union[str, float, int]
    line: int


def tokenize(mesh_text: str) -> List[Token]:
    tokens = []
    current_index = 0
    buffer = ""
    current_line = 0

    for current_index, current_char in enumerate(mesh_text):
        current_char = mesh_text[current_index]

        if current_char.isspace():
            if current_char == "\n":
                current_line += 1

            if buffer.startswith('$'):
                # SectionStart or SectionEnd
                if buffer == TokenType.SectionStart.value:
                    tokens.append(Token(
                        TokenType.SectionStart,
                        buffer[1:],
                        current_line
                    ))
                elif buffer == TokenType.SectionEnd.value:
                    tokens.append(
                        Token(TokenType.SectionEnd, buffer[1:], current_line)
                    )
                else:
                    raise ValueError(
                        "Error during tokenization: Expected "
                        f"SectionStart or SectionEnd but got {buffer}"
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
    if not given_token.token_type is expected_type:
        raise ValueError(
            f"Line {given_token.line}: Expected {expected_type.value} but"
            f" got {given_token.token_type.value}"
        )


class MeshData:
    nodes: npt.NDArray
    elements: npt.NDarray


class GmshData:
    mesh_format: Dict
    physical_names: Dict
    nodes: npt.NDArray
    elements: npt.NDArray

    def _parse_tokens(self, tokens: List[Token]):
        for token in tokens:
            expect_token(token, TokenType.SectionStart)

            if token.value == "MeshFormat":
                _parse_mesh_format()
            elif token.value == "PhysicalNames":
                _parse_physical_names()
            elif token.value == "Nodes":
                _parse_nodes()
            elif token.value == "Elements":
                _parse_elements()
            else:
                raise ValueError(
                    f"Line {token.line}: Unknown section type: {token.value}"
                )

    @staticmethod
    def load_mesh_from_file(file_path: str) -> GmshData:
        if not os.path.exists(file_path):
            raise IOError(
                f"Cannot read mesh since file does not exist {file_path}"
            )

        mesh_text = ""
        with open(file_path, "r", encoding="UTF-8") as fd:
            mesh_text = fd.read()

        tokens = tokenize(mesh_text)

        gmsh_data = GmshData()
        gmsh_data._parse_tokens(tokens)

        return gmsh_data

    def get_mesh_nodes_and_elements(self) -> Tuple[npt.NDArray, npt.NDArray]:
        pass

    def get_nodes_by_physical_group(
        self,
        physical_group_name: str
    ) -> npt.NDArray:
        pass

    def get_elements_by_physical_group(
        self,
        physical_group_name: str
    ) -> npt.NDArray:
        pass
