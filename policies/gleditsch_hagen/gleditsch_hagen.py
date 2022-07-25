import copy
from policies import Policy
import sim

from policies.gleditsch_hagen.pattern_based_cgh.pattern_based_main import PatternBasedCGH


class GleditschHagenPolicy(Policy):
    def __init__(self, variant='PatternBased'):
        self.variant = variant #Exact, RouteBased and PatternBased 
        super().__init__()  

    def get_best_action(self, simul, vehicle):
        if self.variant == 'PatternBased':
            return self.PB_solve(simul,vehicle)
        elif self.variant == 'ColumnBased':
            print('not yet implemented')
            

    def PB_solve(self, simul, vehicle):
        PBCGH = PatternBasedCGH(simul, vehicle)

        next_station, num_loading, num_unloading = PBCGH.return_solution(vehicle_index_input=vehicle.id)        
        bikes_to_swap = []
        bikes_to_pickup =  []
        for i in range(int(num_loading)):
            bikes_to_pickup.append(vehicle.location.bikes[i])
        bikes_to_deliver = []
        for i in range(int(num_unloading)):
            bikes_to_deliver.append(vehicle.get_bike_inventory()[i]) #just pick the first ones

        return sim.Action(
            bikes_to_swap,
            bikes_to_pickup,
            bikes_to_deliver,
            next_station, #this is the id
        )


