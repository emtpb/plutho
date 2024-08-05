import os
import numpy as np
from typing import List
from tkinter import *
import time

class PhysicalGroup:
    dimension = int
    physical_tag = int
    name = str

    def __init__(self, dim, physical_tag, name):
        self.dimension = dim
        self.physical_tag = physical_tag
        self.name = name

class Entity:
    physical_tags: List[int]
    node_indices: List[int]
    element_indices: List[int]

    def __init__(self):
        self.physical_tags = []
        self.node_indices = []
        self.element_indices = []

class PointEntity(Entity):
    x: float
    y: float
    z: float

    def __init__(self, x, y, z, physical_tags):
        super().__init__()
        self.x = x
        self.y = y
        self.z = z
        self.physical_tags = physical_tags

class CurveEntity(Entity):
    bounding_point_tags: List[int]
    orientation: int

    def __init__(self, physical_tags, bounding_point_tags, orientation):
        super().__init__()
        self.physical_tags = physical_tags
        self.orientation = orientation
        self.bounding_point_tags = bounding_point_tags

class SurfaceEntity(Entity):
    bounding_curve_tags: List[int]

    def __init__(self, physical_tags, bounding_curve_tags):
        super().__init__()
        self.physical_tags = physical_tags
        self.bounding_curve_tags = bounding_curve_tags

class VolumeEntity(Entity):
    bounding_surface_tags: List[int]

    def __init__(self, physical_tags, bounding_surface_tags):
        super().__init__()
        self.physical_tags = physical_tags
        self.bounding_surface_tags = bounding_surface_tags


class MeshParser():
    file_path: str
    version_number: int
    physical_groups = List[PhysicalGroup]
    entities = List[List[Entity]]
    nodes = List[List[float]]
    elements = List[List[int]]

    def _parse_mesh_format(self, lines):
        contents = lines[0].split(" ")
        self.version_number = contents[0]
        # TODO Version check

    def _parse_physical_names(self, lines):
        self.physical_groups = []

        for line in lines[1:]:
            dimension, tag, name = line.split(" ")
            self.physical_groups.append(PhysicalGroup(int(dimension), int(tag)-1, name[1:-1]))

    def _parse_entities(self, lines):
        # Parse first line
        contents = lines[0].split(" ")
        num_points = int(contents[0])
        num_curves = int(contents[1])
        num_surfaces = int(contents[2])
        num_volumes = int(contents[3])

        # Create empty lists
        point_entities = [0] * num_points
        curve_entities = [0] * num_curves
        surface_entities = [0] * num_surfaces
        volume_entities = [0] * num_volumes

        offset = 1
        for line in lines[offset:offset+num_points]:
            contents = line.split(" ")
            point_tag = int(contents[0])-1
            x, y, z = contents[1:4]
            number_physical_tags = int(contents[4])
            physical_tags = [int(x)-1 for x  in contents[5:5+number_physical_tags]]
            point_entities[point_tag] = PointEntity(float(x), float(y), float(z), physical_tags)

        offset += num_points
        for line in lines[offset:offset+num_curves]:
            contents = line.split(" ")
            curve_tag = int(contents[0])-1
            number_physical_tags = int(contents[7])
            physical_tags = [int(x)-1 for x in contents[8:8+number_physical_tags]]
            bounding_point_tags = [abs(int(x))-1 for x in contents[8+number_physical_tags+1:-1]]
            orientation = 1 if int(contents[-2]) > 0 else -1
            curve_entities[curve_tag] = CurveEntity(physical_tags, bounding_point_tags, orientation)

        offset += num_curves
        for line in lines[offset:offset+num_surfaces]:
            # Surfaces
            contents = line.split(" ")
            surface_tag = int(contents[0])-1
            number_physical_tags = int(contents[7])
            physical_tags = [int(x)-1 for x in contents[8:8+number_physical_tags]]
            bounding_curve_tags = [abs(int(x))-1 for x in contents[8+number_physical_tags+1:-1]]
            surface_entities[surface_tag] = SurfaceEntity(physical_tags, bounding_curve_tags)

        offset += num_surfaces
        for line in lines[offset:offset+num_volumes]:
            # Volumes
            contents = line.split(" ")
            volume_tag = int(contents[0])-1
            number_physical_tags = int(contents[7])
            physical_tags = [int(x)-1 for x in contents[8:8+number_physical_tags]]
            bounding_surface_tags = [abs(int(x))-1 for x in contents[8+number_physical_tags+1:-1]]
            volume_entities[volume_tag] = VolumeEntity(physical_tags, bounding_surface_tags)

        self.entities = [point_entities, curve_entities, surface_entities, volume_entities]

    def _parse_nodes(self, lines):
        # Parse first line
        contents = lines[0].split(" ")
        num_entity_blocks = int(contents[0])
        num_nodes = int(contents[1])
        self.nodes = [0] * num_nodes

        offset = 1
        for _ in range(num_entity_blocks):
            # Parse first line of entity block
            block_contents = lines[offset].split(" ")
            entity_dim = int(block_contents[0])
            entity_tag = int(block_contents[1])-1
            num_nodes_block = int(block_contents[3])
            
            # Get the entity based of the dim and tag for the upcoming nodes
            entity = self.entities[entity_dim][entity_tag]

            for i in range(num_nodes_block):
                # Get node index
                node_contents = lines[offset+i+1].split(" ")
                node_tag = int(node_contents[0])-1

                # Get node coordinates
                x, y, z = lines[offset+i+1+num_nodes_block].split(" ")

                # Add the node index to the corresponding entity
                entity.node_indices.append(node_tag)
                # Add node in general nodes list
                self.nodes[node_tag] = [float(x), float(y), float(z)]

            offset += 2*num_nodes_block + 1

    def _parse_elements(self, lines):
        contents = lines[0].split(" ")
        num_entity_blocks = int(contents[0])
        num_elements = int(contents[1])
        self.elements = [0] * num_elements

        offset = 1
        for _ in range(num_entity_blocks):
            # Parse first line of entity block
            block_contents = lines[offset].split(" ")
            entity_dim = int(block_contents[0])
            entity_tag = int(block_contents[1])-1
            element_type = int(block_contents[2])
            num_elements_block = int(block_contents[3])

            # Get the entity based of the dim and tag for the upcoming nodes
            entity = self.entities[entity_dim][entity_tag]

            for i in range(num_elements_block):
                # Get element tag and node tags of the element
                element_contents = lines[offset+i+1].split(" ")
                element_tag = int(element_contents[0])-1
                node_tags = [int(x)-1 for x in element_contents[1:-1]]

                # Add element to entity
                entity.element_indices.append(element_tag)
                # Add element to list of elements
                self.elements[element_tag] = node_tags

            offset += num_elements_block + 1

    def parse(self, file_path):
        self.file_path = file_path

        if not os.path.exists(file_path):
            raise FileExistsError(f"File {file_path} does not exist.")

        lines = None
        with open(file_path, "r", encoding="UTF-8") as fd:
            lines = fd.read().splitlines()

        if not lines:
            raise Exception("File {file_path} is empty.")
    
        line_history = []
        for line in lines:
            if line.startswith("$End"):
                if line == "$EndMeshFormat":
                    self._parse_mesh_format(line_history)
                elif line == "$EndPhysicalNames":
                    self._parse_physical_names(line_history)
                elif line == "$EndEntities":
                    self._parse_entities(line_history)
                elif line == "$EndNodes":
                    self._parse_nodes(line_history)
                elif line == "$EndElements":
                    self._parse_elements(line_history)
                line_history = []
            else:
                if line.startswith("$"):
                    continue
                line_history.append(line)

    def _getNodesInEntity(self, dimension, entity):
        # Find points corresponding to the entity
        nodes = entity.node_indices

        # Since the nodes on the border of the entity shall also be returned it is necessary
        # to go through the different lower-dimensional entities to get the nodes if the current entity
        # is higher dimensional (dim > 1)
        if dimension == 1:
            tag1, tag2 = entity.bounding_point_tags
            nodes.append(tag1)
            nodes.append(tag2)
        elif dimension == 2:
            curves = entity.bounding_curve_tags
            for curve_tag in curves:
                curve_nodes = self._getNodesInEntity(1, self.entities[1][curve_tag])
                nodes.extend(curve_nodes)
        elif dimension == 3:
            surfaces = entity.bounding_surface_tags
            for surface_tag in surfaces:
                surface_nodes = self._getNodesInEntity(2, self.entities[2][surface_tag])
                nodes.extend(surface_nodes)
                
        return nodes
    
    def getElementsInPhysicalGroup(self, physical_group_name):
        # Find physical group with given name
        physical_tag = -1
        dimension = -1
        for physical_group in self.physical_groups:
            if physical_group.name == physical_group_name:
                physical_tag = physical_group.physical_tag
                dimension = physical_group.dimension
                break

        # Find entity which contains the physical tag in the appropriate dimension
        entity = None
        for current_entity in self.entities[dimension]:
            if physical_tag in current_entity.physical_tags:
                entity = current_entity
                break

        if entity is None:
            raise Exception(f"No entity found which is connected to the physical group {physical_group_name}")

        elements = []
        for element_index in entity.element_indices:
            elements.append(self.elements[element_index])

        return elements

    def getNodesInPhysicalGroup(self, physical_group_name):
        elements = self.getElementsInPhysicalGroup(physical_group_name)

        # Get unique nodes from elements
        nodes = []
        for element in elements:
            for node in element:
                if node not in nodes:
                    nodes.append(node)

        return nodes

    """
    def getNodesInPhysicalGroup(self, physical_group_name):
        # Find physical group with given name
        physical_tag = -1
        dimension = -1
        for physical_group in self.physical_groups:
            if physical_group.name == physical_group_name:
                physical_tag = physical_group.physical_tag
                dimension = physical_group.dimension
                break

        if physical_tag == -1 or dimension == -1:
            raise Exception(f"Physical group with name {physical_group_name} not found.")
        
        # Find entity which contains the physical tag in the appropriate dimension
        entity = None
        for current_entity in self.entities[dimension]:
            if physical_tag in current_entity.physical_tags:
                entity = current_entity
                break
        
        if entity is None:
            raise Exception(f"No entity found which is connected to the physical group {physical_group_name}")

        nodes_in_entity = self._getNodesInEntity(dimension, entity)

        # Get node coordinates from node list
        nodes = []
        for node in nodes_in_entity:
            nodes.append(self.nodes[node])

        return nodes
    """

    def getTriangleElements(self):
        current_elements = []
        for element in self.elements:
            if len(element) == 3:
                current_elements.append(element)

        return current_elements

    def __init__(self, file_path):
        self.parse(file_path)

def draw_mesh(nodes, elements):
    top = Tk()
    canvas_width = 1100
    canvas_height = 300
    top.geometry(f"{canvas_width+10}x{canvas_height+10}")
    c = Canvas(top, bg="white", height=str(canvas_height), width=str(canvas_width))

    point_thickness = 5
    offset_x = 10
    offset_y = 10

    transform_coords = lambda x: [2e5*x[0]+offset_x,
                                  -2e5*x[1]+(canvas_height-2*offset_y)+offset_y]

    for element in elements:
        for index in range(len(element)):
            x1, y1 = transform_coords(nodes[element[index]])
            x2, y2 = transform_coords(nodes[element[(index+1)%len(element)]])
            c.create_line(x1, y1, x2, y2)
        #if len(element) == 3:
        #    x1, y1 = transform_coords(nodes[element[0]])
        #    x2, y2 = transform_coords(nodes[element[1]])
        #    x3, y3 = transform_coords(nodes[element[2]])
        #    c.create_line(x1, y1, x2, y2)
        #    c.create_line(x1, y1, x3, y3)
        #    c.create_line(x2, y2, x3, y3)

    for node in nodes:
        x, y = transform_coords(node)
        c.create_oval(x-point_thickness/2,
                      y-point_thickness/2,
                      x+point_thickness/2,
                      y+point_thickness/2, fill="black")

    c.pack()
    top.mainloop()

if __name__ == "__main__":
    mesh_file = "piezo.msh"
    start_time = time.time()
    parser = MeshParser(mesh_file)
    print("Parse time", time.time()-start_time)

    nodes = parser.nodes
    elements = parser.getTriangleElements()

    print("Total number of nodes", len(nodes))
    print("Total number of triangles", len(elements))

    draw_mesh(nodes, elements)


    
