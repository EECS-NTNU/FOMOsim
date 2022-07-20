"""
This file contains all the policies used in the thesis.
"""
import copy

from policies import Policy
import sim
import settings
from policies.gleditsch_hagen.heuristic_manager import *


class GleditschHagenPolicy(Policy):
    def __init__(self, variant='PatternBased', planning_horizon=25):

        self.variant = variant #Exact, RouteBased and PatternBased 
        self.planning_horizon = planning_horizon

        
        

        super().__init__()  

    def get_best_action(self, simul, vehicle):
        if self.variant = 'PatternBased':
            return self.PB_solve(simul,vehicle)

    def PB_solve(self, simul, vehicle):

        #UPDATE THIS ONE

        heuristic_man = HeuristicManager(simul, vehicle, simul.state.locations,
                                         planning_horizon=self.planning_horizon,
                                         
                                         #what else:
                                         init_branching=self.branching,
                                         , handling_time=self.handling_time, flexibility=self.flexibility,
                                         average_handling_time=self.average_handling_time, seed_scenarios_subproblems=simul.state.rng.integers(10000), 
                                         criticality=self.criticality, weights=self.weights, crit_weights=self.crit_weights)


        # Index of vehicle that triggered event
        next_station, num_loading, num_unloading = heuristic_man.return_solution(vehicle_index=vehicle.id)
        
        scooters_to_swap = []
        scooters_to_pickup = vehicle.location.scooters[range(num_loading)]
        scooters_to_deliver = []
        for i in range(num_unloading):
            scooters_to_deliver.append(vehicle.get_scooter_inventory()[i]) #ust pick the first ones

        return sim.Action(
            scooters_to_swap,
            scooters_to_pickup,
            scooters_to_deliver,
            next_station.station_id,
        )


