import copy
from policies import Policy
import sim

from policies.gleditsch_hagen.pattern_based_cgh.pattern_based_main import PatternBasedCGH


class GleditschHagenPolicy(Policy):
    def __init__(self, variant='PatternBased'):
        self.variant = variant #Exact, RouteBased and PatternBased 
        super().__init__()  
        self.num_times_called = 0
        # TO DO:
        # - ADD THE WEIGHTS and other PARAMETERS HERE
        
    def get_best_action(self, simul, vehicle):
        if self.variant == 'PatternBased':
            return self.PB_solve(simul,vehicle)
        elif self.variant == 'ColumnBased':
            print('not yet implemented')
            
    def PB_solve(self, simul, vehicle):
        
        vehicle_same_location = False #this can happen in the beginning of the simulation
        #Alternatively, the vehicles can be randomly distributed.
        for vehicle_other in simul.state.vehicles:
            if vehicle is not vehicle_other:
                if vehicle_other.location == vehicle.location:
                    vehicle_same_location = True   #
        
        PBCGH = PatternBasedCGH(simul, vehicle, vehicle_same_location)
        self.num_times_called+=1
        
        next_station, num_loading, num_unloading = PBCGH.return_solution(vehicle_index_input=vehicle.id)        
        bikes_to_swap = []
        bikes_to_pickup =  []
        bikes_to_pickup = [bike.id for bike in vehicle.location.get_bikes()][0:int(num_loading)]
        #for i in range(int(num_loading)):
        #    vehicle.location.bikes[i]
        #    bikes_to_pickup.append(i)  #using vehicle.location.bikes[i] causes type error
        bikes_to_deliver = []
        bikes_to_deliver = [bike.id for bike in vehicle.get_bike_inventory()][0:int(num_unloading)]
        #for i in range(int(num_unloading)):
        #    bikes_to_deliver.append(i) #just pick the first ones

        return sim.Action(
            bikes_to_swap,
            bikes_to_pickup,
            bikes_to_deliver,
            next_station, #this is the id
        )


