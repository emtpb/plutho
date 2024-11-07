""""""

# Python stanard libraries
import os
import numpy as np

# Third party libraries

# Local libraries
from piezo_fem import PiezoSimulation, SimulationData, HeatConductionSim
from piezo_fem.piezo_fem import SimulationException

class CoupledSimulation:

    piezo_simulation: PiezoSimulation
    heat_conduction_sim: HeatConductionSim

    def __init__(
            self,
            piezo_simulation: PiezoSimulation,
            heat_conduction_sim_data: SimulationData):
        self.piezo_simulation = piezo_simulation

        # Create heat conduction sim
        self.heat_conduction_sim = HeatConductionSim(
            piezo_simulation.mesh_data,
            piezo_simulation.material_data,
            heat_conduction_sim_data
        )


    def run_stationary_heat_conduction_sim(
            self,
            number_of_time_steps,
            delta_t):
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

        # Since the mech losses are stationary the excitation frequency
        # can be used
        averaging_count = 1/ \
            (self.piezo_simulation.excitation_info.frequency*piezo_delta_t)
        print(averaging_count)

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
        self.heat_conduction_sim.solve_time(theta_start)

    def save_heat_conduction_results(self):
        