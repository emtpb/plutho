"""Module for coupled simulations."""

# Python stanard libraries
import numpy as np

# Third party libraries

# Local libraries
from piezo_fem import SingleSimulation, SimulationData, Mesh, MaterialData, \
    FieldType


class CoupledThermPiezoHeatCond:

    # Basic settings
    working_directory: str
    simulation_name: str

    # Simulations
    piezo_sim: SingleSimulation
    heat_cond_sim: SingleSimulation

    def __init__(
            self,
            working_directory: str,
            simulation_name: str,
            thermo_piezo_simulation_data: SimulationData,
            heat_cond_simulation_data: SimulationData,
            mesh: Mesh):
        self.working_directory = working_directory
        self.simulation_name = simulation_name

        # Init simulations
        piezo_sim = SingleSimulation(
            working_directory,
            simulation_name,
            mesh
        )
        piezo_sim.setup_thermal_piezo_time_domain(
            thermo_piezo_simulation_data
        )

        heat_cond_sim = SingleSimulation(
            working_directory,
            simulation_name,
            mesh
        )
        heat_cond_sim.setup_heat_conduction_time_domain(
            heat_cond_simulation_data.delta_t,
            heat_cond_simulation_data.number_of_time_steps,
            heat_cond_simulation_data.gamma
        )

        self.piezo_sim = piezo_sim
        self.heat_cond_sim = heat_cond_sim

    def add_material(
            self,
            material_name: str,
            material_data: MaterialData,
            physical_group_name: str):
        self.piezo_sim.add_material(
            material_name,
            material_data,
            physical_group_name
        )
        self.heat_cond_sim.add_material(
            material_name,
            material_data,
            physical_group_name
        )

    def set_excitation(
            self,
            excitation):
        self.piezo_sim.add_dirichlet_bc(
            FieldType.PHI,
            "Electrode",
            excitation
        )
        self.piezo_sim.add_dirichlet_bc(
            FieldType.PHI,
            "Ground",
            np.zeros(len(excitation))
        )
        self.piezo_sim.add_dirichlet_bc(
            FieldType.U_R,
            "Symaxis",
            np.zeros(len(excitation))
        )

    def simulate(self, **kwargs):
        # Run thermo_piezoelectric simulation first
        self.piezo_sim.simulate(**kwargs)
        self.piezo_sim.save_simulation_results("thermo_piezo")

        # Calculate mech losses which are sources in heat cond sim
        avg_mech_loss = np.mean(
            self.piezo_sim.solver.mech_loss[:, -100:],
            axis=1
        )
        self.heat_cond_sim.solver.set_constant_volume_heat_source(
            avg_mech_loss,
            self.heat_cond_sim.solver.simulation_data.number_of_time_steps
        )

        # Set the result temp field of piezo sim to start field of heat cond
        # sim
        number_of_nodes = len(self.piezo_sim.mesh_data.nodes)
        theta_start = self.piezo_sim.solver.u[3*number_of_nodes:, -1]

        self.heat_cond_sim.simulate(theta_start=theta_start)
        self.heat_cond_sim.save_simulation_results("heat_cond")


class CoupledFreqPiezoHeatCond:
    pass