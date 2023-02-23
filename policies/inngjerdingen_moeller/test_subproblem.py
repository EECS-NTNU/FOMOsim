from parameters_MILP import MILP_data
from mathematical_model import run_model
from inngjerdingen_moeller_policy import InngjerdingenMoellerPolicy
# import inngjerdingen_moeller

import sim
import demand
from init_state.wrapper import read_initial_state
import target_state
from helpers import timeInMinutes
from visualize_subproblem import Visualizer


def test_subproblems(filename, start_day, start_hour, t_state, time_horizon, tau, duration, number_of_runs, number_of_vehicles, roaming):
        results = dict()
        start_stations = [0,5,10,15,20,25,30,35,40] #use this for Trondheim
        # start_stations = [4,5,10,15,20,25,30,35,40] #use this for Oslo
        # start_stations = [0,1,2,3,4,5,6,7,0,  1,2,3,4,5,6,7,0,1, 2,3,4,5,6,7,0,1,2] #use this for Edinburgh
        for test_number in range(0, number_of_runs):
                if test_number > 2 and test_number < 6:
                        start_hour = 12
                elif test_number > 5:
                        start_hour = 16
                test_state = read_initial_state(filename)
                test_state.set_seed(1)
                test_demand = demand.Demand()
                test_demand.update_demands(test_state, start_day, start_hour)
                start_time = timeInMinutes(hours=start_hour)
                t_state.update_target_state(test_state, start_day, start_hour)
                policy = InngjerdingenMoellerPolicy()
                test_state.set_vehicles([policy for _ in range(0,number_of_vehicles)])
                for vehicle in range(0,number_of_vehicles):
                        test_state.vehicles[vehicle].location = test_state.locations[start_stations[test_number] + 5*vehicle]  #use this for Trondheim and Oslo
                        # test_state.vehicles[vehicle].location = test_state.locations[start_stations[test_number+9*vehicle]] #use this for Edinburgh
                test_simul = sim.Simulator(
                        initial_state = test_state,
                        target_state = t_state,
                        demand = test_demand,
                        start_time = start_time,
                        duration = duration,
                        verbose = True,
                )
                d = MILP_data(test_simul, time_horizon, tau)
                d.initalize_parameters()
                m=run_model(d, roaming)
                # sol = policy.return_solution(m,test_state.vehicles[0])
                # print(sol)
                results[test_number] = [round(m.Runtime,2), m.MIPGap]
                print("\n----Test run number:", str(test_number+1), "----")
                print("Start hour:", str(start_hour))
                print("Runtime of experiment was", str(round(m.Runtime,2)))
                print("MIP gap was ", str(m.MIPGap))
                # v= inngjerdingen_moeller.Visualizer(m,d)
                # v.visualize_route()
        total_runtime = 0
        total_MIPGap = 0
        for run in range(0, number_of_runs):
                total_runtime += results[run][0]
                total_MIPGap += results[run][1]
        avg_runtime = total_runtime/number_of_runs
        avg_MIPGap = total_MIPGap/number_of_runs
        print("\n--------TESTING COMPLETE--------")
        print("Average runtime:", str(round(avg_runtime, 2)))
        print("Average MIP-gap", str(avg_MIPGap))


def test_single_subproblems(filename, start_day, start_hour, t_state, time_horizon, tau, duration, number_of_vehicles, roaming):
        test_state = read_initial_state(filename)
        test_state.set_seed(1)
        test_demand = demand.Demand()
        test_demand.update_demands(test_state, start_day, start_hour)
        start_time = timeInMinutes(hours=start_hour)
        t_state.update_target_state(test_state, start_day, start_hour)
        policy = InngjerdingenMoellerPolicy()
        test_state.set_vehicles([policy for _ in range(0, number_of_vehicles)])
        for vehicle in range(0,number_of_vehicles):
                        test_state.vehicles[vehicle].location = test_state.locations[1 + 5*vehicle]
        test_simul = sim.Simulator(
                initial_state = test_state,
                target_state = t_state,
                demand = test_demand,
                start_time = start_time,
                duration = duration,
                verbose = True,
        )
        d = MILP_data(test_simul, time_horizon, tau)
        d.initalize_parameters()
        # d.print_neighbor_info(16)
        # d.deep_dive_test_2()
        m=run_model(d, roaming)
        
        # m.printAttr("X")
        print("Runtime of experiment was", str(round(m.Runtime,2)))
        # print("MIP gap was ", str(m.MIPGap)) #this doesn't work when using variable relaxation 
        # v=Visualizer(m,d)
        # v.visualize_route()
        # v.visualize_map_and_route()
        # v.visualize_stations()

if __name__ == "__main__":
# ------------ TESTING DATA MANUALLY ---------------
        # filename = "instances/EH_W31"
        # filename = "instances/TD_W34"
        filename = "instances/TD_W34_old"
        # filename = "instances/OS_W31"

        # filename = "instances/BG_W35"

        START_DAY = 0 #0 -> monday ,days other than 0 results in target inventory = 0 for all stations
        START_HOUR = 8 #8 -> 08:00 am
        START_TIME = timeInMinutes(hours=START_HOUR)
        DURATION = timeInMinutes(hours=1)
        time_horizon = 25
        tau = 5
        number_of_runs = 9
        number_of_vehicles = 1
        roaming = True
 
        # tstate = target_state.EvenlyDistributedTargetState()
        # tstate = target_state.OutflowTargetState()
        # tstate = target_state.EqualProbTargetState()
        tstate = target_state.USTargetState()
        # tstate = target_state.HalfCapacityTargetState()
        
        # test_subproblems(filename, START_DAY, START_HOUR, tstate, time_horizon, tau, DURATION, number_of_runs, number_of_vehicles, roaming)
        test_single_subproblems(filename, START_DAY, START_HOUR, tstate, time_horizon, tau, DURATION, number_of_vehicles, roaming)
        
# ----------------------------------------------------