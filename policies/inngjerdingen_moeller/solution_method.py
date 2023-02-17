'''rename this module and class eg., PILOT, InngjerdingenMoellerPolicy_2, heuristic or something similar '''
from policies import Policy
import sim
from criticality_score_neighbor import calculate_criticality

class SolutionMethod(Policy):
    def __init__(self):
        super().__init__()

    def get_best_action(self, simul, vehicle):
        next_station = None
        bikes_to_pickup = []
        bikes_to_deliver = []  
        
        #########################################
        #               WHAT TO DO              #
        #########################################


        ##########################################
        #               WHERE TO GO              #
        ##########################################

        
        
        return sim.Action(
            [],               # batteries to swap
            bikes_to_pickup, #list of bike id's
            bikes_to_deliver, #list of bike id's
            next_station, #id
        )   


    def PILOT_function():
        #calls the greedy function recursively + smart filtering, branching and depth
        return None
    
    def greedy_with_neighbor():
        #calls calculate_criticality 
        #should calculate_criticality recieve potential stations or receive
        # all stations and make the filtering inside?
        return None