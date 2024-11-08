""""""

# Python stanard libraries
from typing import Dict
import os
import numpy as np

# Third party libraries

# Local libraries
from piezo_fem import PiezoSimulation, SimulationData, HeatConductionSim, \
                        PiezoSimTherm
from piezo_fem.piezo_fem import SimulationException

class CoupledSimulation:
    """Creates a coupled simulation of a thermo-piezoelectric simulation
    with a heat conduction simulation. The mechanical losses calculated in th
    thermo-piezoelectric simulation are extrapolated and embed into the heat
    conduction simulation. Therefore it is possible to predict temperature
    distributions at longer time intervals.
    In order to work properly it is necessary that the mechanical losses from 
    the thermo-piezoelectric are stationary.

    Attributes:
        piezo_simulation: Thermo-piezoelectric simulation.
        heat_conduction_simulation: Heat conduction simulation.
    """

    piezo_simulation: PiezoSimulation
    heat_conduction_sim: HeatConductionSim

    def __init__(
            self,
            piezo_simulation: PiezoSimulation,
            heat_conduction_sim_data: SimulationData):
        if not isinstance(piezo_simulation.solver, PiezoSimTherm):
            raise SimulationException(
                "Thermo-piezoelectric simulation must be given"
            )

        self.piezo_simulation = piezo_simulation

        # Create heat conduction sim
        self.heat_conduction_sim = HeatConductionSim(
            piezo_simulation.mesh_data,
            piezo_simulation.material_data,
            heat_conduction_sim_data
        )

    def run_stationary_heat_conduction_sim(
            self,
            averaging_count: int,
            convection_bc_settings: Dict = None):
        """Runs a heat conduction simulation which uses the result of the
        given piezo simulation. The results are saved in the heat conduction
        simulation object.

        Parameters:
            averaging_count: Number of time steps over which the mechanical
                losses per element are averaged. The last values are taken such
                as mech_loss[averaging_count:].
            convection_bc_settings: Dictionary which contains the data needed
                to use convection boundary condition. If it is None no
                convection boundary is used.
                The dictionary must contain the following values:
                    "boundary_elements": List of elements at the boundary which
                        are used in the bc.
                    "alpha": Heat transfer coefficient of the material.
                    "outer_temperature": Temperature outside of the material.
        """
        # Check if simulation results are present
        if not np.any(self.piezo_simulation.solver.mech_loss) or \
                not np.any(self.piezo_simulation.solver.u) or \
                not np.any(self.piezo_simulation.q):
            raise SimulationException("Please run the piezo simulation first.")

        heat_delta_t = self.heat_conduction_sim.simulation_data.delta_t
        piezo_delta_t = self.piezo_simulation.simulation_data.delta_t
        heat_number_of_time_steps = \
            self.heat_conduction_sim.simulation_data.number_of_time_steps
        print(
            "Number of skipped time steps:",
            heat_delta_t/piezo_delta_t
        )

        avg_mech_loss_density = np.mean(
            self.piezo_simulation.solver.mech_loss[:, -averaging_count:],
            axis=1
        )
        self.heat_conduction_sim.set_constant_volume_heat_source(
            avg_mech_loss_density,
            heat_number_of_time_steps
        )

        # Get the starting field for heat conduction simulation from the
        # ending field of the piezo sim
        number_of_nodes = len(self.piezo_simulation.mesh_data.nodes)
        theta_start = self.piezo_simulation.solver.u[3*number_of_nodes:, -1]

        self.heat_conduction_sim.assemble()

        if convection_bc_settings:
            self.heat_conduction_sim.solve_time(
                theta_start,
                convection_bc_settings["boundary_elements"],
                convection_bc_settings["alpha"],
                convection_bc_settings["outer_temperature"]
            )
        else:
            self.heat_conduction_sim.solve_time(theta_start)

    def save_heat_conduction_results(self):
        results_folder = self.piezo_simulation.workspace_directory
        sim_name = self.piezo_simulation.simulation_name

        np.save(
            os.path.join(
                results_folder,
                f"{sim_name}_heat_cond_theta.npy"),
            self.heat_conduction_sim.theta
        )

        # TODO Should the predicted mech loss density also be saved?