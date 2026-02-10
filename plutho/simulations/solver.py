"""Base module for all linear simulations."""

# Python standard libraries
import os
import configparser
from abc import ABC, abstractmethod
from typing import Dict, Union, List

# Third party libraries
import numpy as np
import numpy.typing as npt


# Local libraries
from ..enums import SolverType, FieldType
from ..materials import MaterialManager, MaterialData
from ..mesh.mesh import Mesh, MeshData
from .helpers import calculate_charge, create_node_points


__all__ = [
    "FEMSolver"
]


class FEMSolver(ABC):
    """Base class for all linear solvers. The assemble() and simulation()
    functions must be implemented in the child class

    Attributes:
        simulation_name: Name of the simulation.
        mesh: Mesh object, stores all the mesh information.
        node_points: Lists of node points per element.
        material_manager: MaterialManager object, stores all the material
            informations.
        dirichlet_nodes: List of nodes for which a dirichlet bc is applied.
        dirichlet_values: Values to which the fields are set at the respective
            nodes
        boundary_conditions: List of all implemented boundary conditions. Is
            used when the simulation settings are saved.
        solver_type: Type of the current solver.

    Parameters:
        simulation_name: Name of the simulation.
        mesh: mesh object, stores all the mesh information.
    """
    simulation_name: str
    mesh: Mesh
    node_points: npt.NDArray
    material_manager: MaterialManager
    dirichlet_nodes: List
    dirichlet_values: List
    boundary_conditions: List
    solver_type: Union[SolverType, None]

    def __init__(self, simulation_name: str, mesh: Mesh):
        self.simulation_name = simulation_name
        self.mesh = mesh
        self.solver_type = None

        nodes, elements = mesh.get_mesh_nodes_and_elements()
        element_order = mesh.element_order
        self.mesh_data = MeshData(nodes, elements, mesh.element_order)
        self.material_manager = MaterialManager(len(elements))
        self.node_points = create_node_points(nodes, elements, element_order)

        self.dirichlet_nodes = []
        self.dirichlet_values = []
        self.boundary_conditions = []

        self.u = None
        self.q = None

    def add_material(
        self,
        material_name: str,
        material_data: MaterialData,
        physical_group_name: str
    ):
        """Adds a material to the simulation

        Parameters:
            material_name: Name of the material.
            material_data: Material data.
            physical_group_name: Name of the physical group for which the
                material shall be set. If this is None or "" the material will
                be set for the whole model.
        """
        if physical_group_name is None or physical_group_name == "":
            element_indices = np.arange(len(self.mesh_data.elements))
        else:
            element_indices = self.mesh.get_elements_by_physical_groups(
                [physical_group_name]
            )[physical_group_name]

        self.material_manager.add_material(
            material_name,
            material_data,
            physical_group_name,
            element_indices
        )

    def add_dirichlet_bc(
        self,
        field_type: FieldType,
        physical_group_name: str,
        values: npt.NDArray
    ):
        """Adds a dirichlet boundary condition (dirichlet bc) to the
        simulation. The given values are for each time step. Therefore each
        value is applied to every node from the given physical_group.

        Parameters:
            field_type: The type of field for which the bc shall be set.
            physical_group_name: Name of the physical group for which the
                boundary condition shall be set.
            values: List of values per time step. Value of the bc for each time
                step. The value for one time step is applied to every element
                equally. Length: number_of_time_step.
        """
        if self.solver_type is None:
            raise ValueError("Solver type is not set")

        match self.solver_type:
            case SolverType.ThermoTime:
                if (field_type is FieldType.U_R or
                        field_type is FieldType.U_Z or
                        field_type is FieldType.PHI):
                    raise ValueError(
                        f"Unknown variable type {field_type} given for"
                        f"simulation type {self.solver_type}"
                    )
            case SolverType.PiezoFreq:
                if field_type is FieldType.THETA:
                    raise ValueError(
                        f"Unknown variable type {field_type} given for"
                        f"simulation type {self.solver_type}"
                    )

        # Save boundary condition for serialization
        self.boundary_conditions.append({
            "field_type": field_type.name,
            "physical_group_name": physical_group_name,
            "values": values.tolist()
        })

        # Apply the given values for all nodes from the given physical_group
        node_indices = self.mesh.get_nodes_by_physical_groups(
            [physical_group_name]
        )[physical_group_name]

        number_of_nodes = len(self.mesh_data.nodes)
        for node_index in node_indices:
            real_index = 0

            # Depending on the variable type and the simulation types
            # the corresponding field variable may be found at different
            # positions of the solution vector
            if field_type is FieldType.PHI:
                real_index = 2*number_of_nodes+node_index
            elif field_type is FieldType.THETA:
                if self.solver_type is SolverType.ThermoTime:
                    real_index = node_index
                else:
                    real_index = 3*number_of_nodes+node_index
            elif field_type is FieldType.U_R:
                real_index = 2*node_index
            elif field_type is FieldType.U_Z:
                real_index = 2*node_index+1

            # TODO: Flip the indices in dirichlet_values: [freq, nodes]
            self.dirichlet_nodes.append(real_index)
            self.dirichlet_values.append(values)

    def clear_dirichlet_bcs(self):
        """Resets the dirichlet boundary conditions."""
        self.boundary_conditions = []
        self.dirichlet_nodes = []
        self.dirichlet_values = []

    def save_simulation_settings(self, directory: str, prefix: str = ""):
        """Creates a folder 'simulation_name' in the given directory and saves
        the current simulation settings in it.

        Parameters:
            directory: Path to the directory in which the simulation folder is
                created.
            prefix: Possible prefix, which is prepend to the file names.
        """
        if self.solver_type is None:
            raise ValueError("Simulation settings can not be saved, since no \
                simulation is configured yet.")

        wd = os.path.join(directory, self.simulation_name)

        if not os.path.isdir(wd):
            os.makedirs(wd)

        settings = configparser.ConfigParser()
        settings["general"] = {
            "simulation_name": self.simulation_name,
            "solver_type": self.solver_type.value,
            "mesh_file_path": self.mesh.mesh_file_path,
            "mesh_element_order": str(self.mesh_data.element_order)
        }
        settings["materials"] = self._get_materials_settings()
        settings["boundary_conditions"] = self._get_bc_settings()

        # Save config file
        if prefix != "":
            prefix += "_"

        config_file = os.path.join(wd, f"{prefix}settings.cfg")
        with open(config_file, "w", encoding="UTF-8") as fd:
            settings.write(fd)

    def save_simulation_results(self, directory: str, prefix: str = ""):
        """Creates a folder 'simulation_name' in the given directory and saves
        the current simulation results in it.

        Parameters:
            directory: Path to the directory in which the simulation folder is
                created.
            prefix: Possible prefix, which is prepend to the file names.
        """
        wd = os.path.join(directory, self.simulation_name)

        if not os.path.isdir(wd):
            os.makedirs(wd)

        if prefix != "":
            prefix += "_"

        if self.u is not None:
            u_path = os.path.join(wd, f"{prefix}u.npy")
            np.save(u_path, self.u)

        if self.q is not None:
            q_path = os.path.join(wd, f"{prefix}q.npy")
            np.save(q_path, self.q)

        match self.solver_type:
            case SolverType.ThermoPiezoTime:
                mech_loss_path = os.path.join(wd, f"{prefix}mech_loss.npy")
                np.save(mech_loss_path, self.mech_loss)
            case SolverType.ThermoTime:
                theta_path = os.path.join(wd, f"{prefix}theta.npy")
                np.save(theta_path, self.theta)
            case SolverType.PiezoFreq:
                if self.frequencies is not None:
                    f_path = os.path.join(wd, f"{prefix}frequencies.npy")
                    np.save(f_path, self.frequencies)

    @classmethod
    def load_simulation_settings(cls, simulation_folder: str):
        """Loads the simulation settings from the given folder.

        Parameters:
            simulation_folder: Path to the simulation folder, which shall be
                loaded (must contain a *.cfg file).

        Returns:
            The loaded simulation class object (some child of the FEMSolver
            class).
        """
        simulation_name = os.path.basename(simulation_folder)
        config_path = os.path.join(simulation_folder, f"{simulation_name}.cfg")
        settings = configparser.ConfigParser()
        settings.read(config_path)
        simulation_name = settings["general"]["simulation_name"]
        mesh_file_path = settings["general"]["mesh_file_path"]
        mesh_element_order = int(settings["general"]["mesh_element_order"])
        mesh = Mesh(mesh_file_path, mesh_element_order)

        # Check for solver type
        solver_type = None
        try:
            solver_type = SolverType(settings["general"]["solver_type"])
        except ValueError:
            raise ValueError(
                f"Unknown solver type: {settings['general']['solver_type']}"
            )

        # Create simulation
        sim = cls(simulation_name, mesh)

        if solver_type is not None and sim.solver_type is not solver_type:
            raise ValueError(
                "Wrong class used to instantiate a solver of type "
                f"{solver_type}"
            )

        # Load materials
        material_settings = settings["materials"]
        for material_name, material in material_settings.items():
            material_data = MaterialData.from_dict(material["material_data"])
            physical_group_name = material["physical_group_name"]

            sim.add_material(
                material_name,
                material_data,
                physical_group_name
            )

        # Load boundary conditions
        bc_settings = settings["boundary_conditions"]
        for bc in bc_settings:
            field_type = FieldType(bc["field_type"])
            physical_group_name = bc["physical_group_name"]
            values = np.array(list(bc["values"]))

            sim.add_dirichlet_bc(field_type, physical_group_name, values)

        return sim

    def _get_materials_settings(self) -> Dict:
        """Local function. Is used to parse the material parameters to a
            dictionary.
        """
        material_settings = {}

        for material in self.material_manager.materials:
            material_settings[material.material_name] = {
                "material_data": material.to_dict(),
                "physical_group_name": material.physical_group_name
            }

        return material_settings

    def _get_bc_settings(self) -> Dict:
        """Local function. Is used to parse the boundary conditions to a
            dirctionary
        """
        bc_settings = {}

        for index, bc in enumerate(self.boundary_conditions):
            bc_settings[index] = bc

        return bc_settings

    def calculate_charge(
        self,
        electrode_name: str,
        is_complex: bool = False
    ):
        """Calculates the charge for the displacement u stored in the current
        object -> self.u. Therefore the simulate() function must be called
        beforehand or a self.u is set manually.

        Parameters:
            electrode_name: Name of the boundary on which the charge is
                calculated, typically the electrode.
            is_complex: Set to true if the displacement and thus the charge
                can be of a complex type. E.g. in a frequency domain
                simulation.
        """
        if self.u is None:
            raise ValueError("Cannot calculate charge since no simulation \
                has been done")

        # Get electrode elements and electrode normals
        electrode_elements = self.mesh.get_elements_by_physical_groups(
            [electrode_name]
        )[electrode_name]
        # TODO This must be calculated dynamically depending on the direction
        # of the normal vector
        electrode_normals = np.tile([0, 1], (len(electrode_elements), 1))

        iteration_count, _ = self.u.shape

        if is_complex:
            q = np.zeros(iteration_count, dtype=np.complex128)
        else:
            q = np.zeros(iteration_count)

        for index in range(iteration_count):
            q[index] = calculate_charge(
                self.u[index, :],
                self.material_manager,
                electrode_elements,
                electrode_normals,
                self.mesh_data.nodes,
                self.mesh_data.element_order,
                is_complex
            )

        self.q = q

    @abstractmethod
    def assemble(self, *args, **kwargs):
        """Needs to be implemented by any child class. Implements the matrix
        assembly.
        """
        raise NotImplementedError("'assembly' function not implemented.")

    @abstractmethod
    def simulate(self, *args, **kwargs):
        """Needs to be implemented by any child class. Implements the
        simulation process
        """
        raise NotImplementedError("'simulate' function not implemented.")
