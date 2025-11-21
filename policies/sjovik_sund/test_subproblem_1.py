import os
import sys

# Ensure repository root is on sys.path so imports work when running this test directly
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from Sub_problem import subproblem_parameters as MILP_data
from sjovik_sund_policy import SjovikSundPolicy
from Sub_problem.sjovik_sund_subproblem import run_subproblem_model

import sim
import demand
from init_state.wrapper import read_initial_state
import target_state
from helpers import timeInMinutes
from visualize_subproblem import Visualizer
import time


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

                # Create vehicles for this test (defensive: clear any existing vehicles)
                policy = SjovikSundPolicy()
                if hasattr(test_state, 'vehicles'):
                        test_state.vehicles = {}
                test_state.set_sb_vehicles([policy for _ in range(0, number_of_vehicles)])

                # Assign starting locations using safe list indexing
                locs = test_state.get_locations()
                vehicles = test_state.get_vehicles()
                for vidx in range(0, number_of_vehicles):
                        desired_idx = start_stations[test_number] + 5 * vidx
                        if len(locs) == 0:
                                raise RuntimeError("No locations available in test state")
                        chosen_loc = locs[desired_idx % len(locs)]
                        vehicles[vidx].location = chosen_loc

                test_simul = sim.Simulator(
                        initial_state=test_state,
                        target_state=t_state,
                        demand=test_demand,
                        start_time=start_time,
                        duration=duration,
                        verbose=True,
                )

                # Instantiate MILP parameters and run subproblem
                d = MILP_data.MILP_parameters(test_simul, time_horizon=time_horizon, tau=tau)
                d.initalize_parameters()
                m = run_subproblem_model(d.to_dict())

                results[test_number] = [round(m.Runtime, 2), m.MIPGap]
                print("\n----Test run number:", str(test_number + 1), "----")
                print("Start hour:", str(start_hour))
                print("Runtime of experiment was", str(round(m.Runtime, 2)))
                print("MIP gap was ", str(m.MIPGap))
                # v= inngjerdingen_moeller.Visualizer(m,d)
                # v.visualize_route()
        total_runtime = 0
        total_MIPGap = 0
        for run in range(0, number_of_runs):
                total_runtime += results[run][0]
                total_MIPGap += results[run][1]
        avg_runtime = total_runtime / number_of_runs
        avg_MIPGap = total_MIPGap / number_of_runs
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

        # Create vehicles (defensive: clear any existing vehicles)
        policy = SjovikSundPolicy()
        if hasattr(test_state, 'vehicles'):
                test_state.vehicles = {}
        test_state.set_sb_vehicles([policy for _ in range(0, number_of_vehicles)])

        # Assign starting locations
        locs = test_state.get_locations()
        vehicles = test_state.get_vehicles()
        for idx, vehicle in enumerate(vehicles[:number_of_vehicles]):
                vehicle.location = locs[(1 + 5 * idx) % len(locs)]

        test_simul = sim.Simulator(
                initial_state=test_state,
                target_state=t_state,
                demand=test_demand,
                start_time=start_time,
                duration=duration,
                verbose=True,
        )

        # Instantiate MILP parameters and run subproblem
        if hasattr(MILP_data, 'MILP_parameters'):
                d = MILP_data.MILP_parameters(test_simul, time_horizon=time_horizon, tau=tau)
        elif callable(MILP_data):
                d = MILP_data(test_simul, time_horizon=time_horizon, tau=tau)
        else:
                raise TypeError("Unsupported MILP_data import; expected module with MILP_parameters or a callable class")
        d.initalize_parameters()
        m = run_subproblem_model(d.to_dict())

        print("Runtime of experiment was", str(round(m.Runtime, 2)))

def test_policy(filename, number_of_runs, start_day, start_hour, t_state, policy, duration, number_of_vehicles):
        test_state = read_initial_state(filename)
        test_state.set_seed(1)
        test_demand = demand.Demand()
        test_demand.update_demands(test_state, start_day, start_hour)
        start_time = timeInMinutes(hours=start_hour)
        t_state.update_target_state(test_state, start_day, start_hour)
        # Defensive: clear any existing vehicles before creating new ones
        if hasattr(test_state, 'vehicles'):
                test_state.vehicles = {}
        test_state.set_sb_vehicles([policy for _ in range(0, number_of_vehicles)])
        solution_times = []
        locs = test_state.get_locations()
        vehicles = test_state.get_vehicles()
        if len(locs) == 0:
                raise RuntimeError("No locations available in test state")
        for run in range(number_of_runs):
                for vidx, vehicle in enumerate(vehicles[:number_of_vehicles]):
                        # safe wrap-around indexing into the locations list
                        desired_idx = run * 4 + 1 + 5 * vidx
                        vehicle.location = locs[desired_idx % len(locs)]
                test_simul = sim.Simulator(
                        initial_state = test_state,
                        target_state = t_state,
                        demand = test_demand,
                        start_time = start_time,
                        duration = duration,
                        verbose = True,
                )
                start_solve = time.time()
                # Use the safe vehicles list (not dict-indexing) to get the first vehicle
                action = policy.get_best_action(test_simul, vehicles[0])
                print(action)
                solution_times.append(time.time()-start_solve)
        avg_sol_time = sum(solution_times)/len(solution_times)
        print("Average solution time:", round(avg_sol_time, 3))

if __name__ == "__main__":
        # ------------ TESTING DATA MANUALLY ---------------
        # filename = "instances/EH_W31"
        filename = "instances/TD_W34"
        # filename = "instances/TD_W34_old"
        # filename = "instances/NY_W31"
        # filename = "instances/OS_W34"
        # filename = "instances/BG_W25"
        #filename = "instances/BO_W31"

        START_DAY = 0  # 0 -> monday ,days other than 0 results in target inventory = 0 for all stations
        START_HOUR = 8  # 8 -> 08:00 am
        START_TIME = timeInMinutes(hours=START_HOUR)
        DURATION = timeInMinutes(hours=6)
        time_horizon = 6
        tau = 5
        number_of_runs = 9
        number_of_vehicles = 2
        roaming = False
 
        # tstate = target_state.EvenlyDistributedTargetState()
        # tstate = target_state.OutflowTargetState()
        # tstate = target_state.EqualProbTargetState()
        tstate = target_state.USTargetState()
        # tstate = target_state.HalfCapacityTargetState()

        criticality_weights_sets = [[0.45, 0.45, 0.1, 0.01]]
        #evaluation_weights = [0.4, 0.3, 0.3] #[avoided_viol, neighbor_roaming, improved deviation]
        policy = SjovikSundPolicy(roaming=False, time_horizon=6, tau=5, weights=[0.45,0.45,0.2,0.01])
        
        #test_subproblems(filename, START_DAY, START_HOUR, tstate, time_horizon, tau, DURATION, number_of_runs, number_of_vehicles, roaming)
        #test_single_subproblems(filename, START_DAY, START_HOUR, tstate, time_horizon, tau, DURATION, number_of_vehicles, roaming)
        test_policy(filename, 10, START_DAY, START_HOUR, tstate, policy, DURATION, number_of_vehicles)