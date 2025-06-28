"""Module for handling the mesh generation and loading."""

# Python standard libraries
import os
from typing import Union, Tuple, Dict, List

# Third party libraries
import gmsh
import numpy.typing as npt

# Local libraries
from .gmsh_parser import GmshParser
from .custom_mesh_parser import CustomParser


class Mesh:
    """Class to generate and read meshes.

    Attributes:
        mesh_file_path: Path to the mesh. File must exist.
    """
    mesh_file_path: str
    file_version: str
    parser: Union[GmshParser, CustomParser]

    def __init__(
        self,
        file_path: str,
    ):
        if not os.path.isfile(file_path):
            raise IOError(f"Mesh file {file_path} not found.")

        self.mesh_file_path = file_path

        # Check gmsh version
        # If version2  -> custom gmsh parser
        # Else -> default gmsh parser
        with open(file_path, "r", encoding="UTF-8") as fd:
            lines = fd.readlines()
            if len(lines) < 2:
                raise IOError(
                    "Given gmsh mesh file {file_path} is not valid"
                )
            second_line = lines[1]
            version, _, _ = second_line.split(" ")
            self.file_version = version

        if version.startswith("2"):
            self.parser = CustomParser(file_path)
        else:
            self.parser = GmshParser(file_path)

    def get_mesh_nodes_and_elements(self) -> Tuple[npt.NDArray, npt.NDArray]:
        return self.parser.get_mesh_nodes_and_elements()

    def get_nodes_by_physical_groups(
        self,
        physical_group_names: List[str]
    ) -> Dict[str, npt.NDArray]:
        if isinstance(self.parser, CustomParser):
            pg_nodes = {}
            for pg in physical_group_names:
                # TODO Maybe let get nodes function run with multple pg names
                pg_nodes[pg] = self.parser.get_nodes_by_physical_group(pg)

            return pg_nodes
        elif isinstance(self.parser, GmshParser):
            return self.parser.get_nodes_by_physical_groups(
                physical_group_names
            )
        else:
            raise NotImplementedError(
                "Cannot read nodes for parser "
                f"{type(self.parser)}"
            )

    def get_elements_by_physical_groups(
        self,
        physical_group_names
    ) -> Dict[str, npt.NDArray]:
        if isinstance(self.parser, CustomParser):
            pg_nodes = {}
            for pg in physical_group_names:
                # TODO Maybe let get nodes function run with multple pg names
                pg_nodes[pg] = self.parser. \
                    get_triangle_elements_by_physical_group(pg)

            return pg_nodes
        elif isinstance(self.parser, GmshParser):
            return self.parser.get_elements_by_physical_groups(
                physical_group_names
            )
        else:
            raise NotImplementedError(
                "Cannot read nodes for parser "
                f"{type(self.parser)}"
            )

    @staticmethod
    def generate_rectangular_mesh(
        file_path: str,
        width: float = 0.005,
        height: float = 0.001,
        mesh_size: float = 0.00015,
        x_offset: float = 0
    ):
        """Creates a gmsh rectangular mesh given the width, height, the mesh
        size and the x_offset.

        Parameters:
            width: With of the rect in m.
            height: Height of the rect in m.
            mesh_size: Mesh size of the mesh. Equal to the maximum distance
                between two point in the mesh.
            x_offset: Moves the rect along the x-direction. Default value is
                0. For 0 the left side of the rect is on the y-axis.
        """

        if not gmsh.is_initialized():
            gmsh.initialize()

        gmsh.open(file_path)
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
        boundary_right = gmsh.model.geo.addPhysicalGroup(1, [right_line])
        gmsh.model.setPhysicalName(1, boundary_right, "RightBoundary")

        model = gmsh.model.geo.addPhysicalGroup(2, [surface])
        gmsh.model.setPhysicalName(2, model, "Surface")

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write(file_path)
