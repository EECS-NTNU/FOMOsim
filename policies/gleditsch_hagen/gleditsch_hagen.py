import copy
from policies import Policy
import sim

from policies.gleditsch_hagen.pattern_based_cgh.pattern_based_main import PatternBasedCGH


class GleditschHagenPolicy(Policy):
    def __init__(self, variant='PatternBased',
                #  weights = (time to violation, net demand , driving time, deviation from target state)
                    parameters = dict(crit_weights=[0.1,0.5,0.1,0.3], #criticality weights used when generating initial routes
                                    vd_weights = [0.8,0.2],     # violation/deviation weights in the objective function of the master problem
                                    planning_horizon = 50,      #in minutes.
                                    branching_constant= 6,      #for the route extension algorithm. How many candidate stations to consider in the branching
                                    threshold_service_vehicle = 0.5) #for the filtering procedure. WHen to go pickup/delivery stations..
                ):
        self.variant = variant #Exact, RouteBased and PatternBased 
        super().__init__()  
        self.num_times_called = 0
        # TO DO:
        # - ADD THE WEIGHTS and other PARAMETERS HERE
        self.parameters = parameters 

    def get_best_action(self, state, vehicle):
        if self.variant == 'PatternBased':
            return self.PB_solve(state,vehicle)
        elif self.variant == 'ColumnBased':
            print('not yet implemented')
            
    def PB_solve(self, state, vehicle):
        
        other_vehicle_at_same_location = False #this can happen in the beginning of the simulation
        #Alternatively, the vehicles can be randomly distributed.
        for vehicle2 in state.vehicles:
            if vehicle is not vehicle2:
                if vehicle2.location == vehicle.location:
                    other_vehicle_at_same_location = True   #
        
        PBCGH = PatternBasedCGH(state, self.parameters, vehicle, other_vehicle_at_same_location)
        self.num_times_called+=1
        
        next_station, num_loading, num_unloading = PBCGH.return_solution(vehicle_index_input=vehicle.id)        
        batteries_to_swap = []
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
            batteries_to_swap,
            bikes_to_pickup,
            bikes_to_deliver,
            next_station, #this is the id
        )


