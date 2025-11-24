import os
import sys 
from pathlib import Path

path = Path(__file__).parents[2]
os.chdir(path)
sys.path.insert(0, '')

from policies.sjovik_sund.Sub_problem.subproblem_parameters import MILP_parameters
from policies.sjovik_sund.Sub_problem.sjovik_sund_subproblem import run_subproblem_model

import time
from gurobipy import GRB

################################################################################
# 1. Define the Test Data Dictionary (Based on N=5, V=1, T=1 Scenario)
################################################################################

def create_test_data():
    """
    Creates a dictionary containing the parameters for the N=5, V=2 test
    instance, using T=6 and tau=5 min from the main simulation setup.
    """
    # --- Parameters matched to run_simulation.py policy setup ---
    T = 6  # Time horizon (periods)
    tau = 5  # Period length (minutes)
    
    # --- Network and Initial Conditions (Kept minimal for debugging) ---
    N = [0, 1, 2, 3, 4]  # 5 Stations
    s = -1  # Source
    d = -2  # Sink
    V = [0, 1] # 2 Vehicles <--- Now two vehicles
    
    # Objective Weights: [w_S, w_C, w_D, r_M]
    w_S, w_C, w_D = 0.45, 0.45, 0.1
    r_M = 0.01

    # Travel Time Parameters
    T_L = 0.5
    T_D = {}  # Travel time in minutes
    T_DD = {} # Discretized travel time in periods

    # Setup T_D and T_DD for all arcs
    
    short_time = 4.0      # T_DD = 1
    medium_time = 15.0    # T_DD = 4
    long_time = 25.0      # T_DD = 6 (Max feasible T_DD for T=6)
    
    all_nodes = N + [s, d]
    
    for i in all_nodes:
        for j in all_nodes:
            if i == d or j == s:
                continue

            if i == j and i in N:
                T_D[(i, j)] = 0.0
                T_DD[(i, j)] = 1
            elif i in N and j in N:
                # Use medium hops to force intermediate steps within T=6
                if (i, j) in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]:
                    T_D[(i, j)] = medium_time
                    T_DD[(i, j)] = 4
                # Long hop to test T_DD = T (arrival at T=6+T_DD=12 - requires t=0 departure)
                elif (i, j) == (0, 4):
                    T_D[(i, j)] = long_time
                    T_DD[(i, j)] = 6
                else: # Default unlisted arcs
                    T_D[(i, j)] = short_time
                    T_DD[(i, j)] = 1
            elif i == s and j in N:
                # V0 starts at S0, V1 starts at S4
                if j in [0, 4]: 
                    T_D[(i, j)] = 0.0
                    T_DD[(i, j)] = 1
            elif i in N and j == d:
                T_D[(i, j)] = 0.0
                T_DD[(i, j)] = 1

    # Vehicle Parameters <-- CORRECTED for 2 vehicles
    Q_V = {0: 3, 1: 2}    
    Q_V0 = {0: 1, 1: 0}   
    eta = {0: 0, 1: 4}    

    # Station Parameters
    Q_S = {i: 10 for i in N} 
    I_N0 = {0: 8, 1: 5, 2: 5, 3: 1, 4: 8} # Initial inventory
    I_T = {i: 5 for i in N}  # Target inventory at T=6

    # Demand (D[(station_idx, period)]). T_pos = {1, 2, ..., 6}.
    D = {}
    for i in N:
        D[(i, 0)] = 0.0 
    
    # Set demand over the horizon to stress S0 (source) and S3/S4 (destination)
    for t in range(1, T + 1):
        D[(0, t)] = -0.5  # Consistent net departures at S0
        D[(3, t)] = -0.5  # Consistent net departures at S3 (high starvation)
        D[(4, t)] = +0.5  # Consistent net arrivals at S4 (high congestion)
        D[(1, t)] = 0.0
        D[(2, t)] = 0.0

    # Maintenance Parameters
    T_M_min = {i: 0 for i in N}
    T_M_max = {i: 0 for i in N}
    # Enforce maintenance at Station 4 (T_M_min = 5 min = 1 period)
    T_M_min[4] = 5.0
    T_M_max[4] = 5.0

    # Final Data Dictionary
    data = {
        "T": T, "tau": tau, "N": N, "s": s, "d": d, "V": V,
        "T_D": T_D, "T_DD": T_DD, "T_L": T_L,
        "T_M_min": T_M_min, "T_M_max": T_M_max,
        "Q_V": Q_V, "Q_V0": Q_V0, "eta": eta,
        "Q_S": Q_S, "I_N0": I_N0, "I_T": I_T,
        "D": D,
        "w_S": w_S, "w_C": w_C, "w_D": w_D, "r_M": r_M
    }
    
    return data

################################################################################
# 2. Execute the Model and Report Results 
################################################################################

if __name__ == "__main__":
    print("---  Generating Test Data for N=5, V=2, T=6 (Policy Config) ---")
    test_data = create_test_data()
    
    T_val = test_data["T"] 
    
    print("\n---  Running Gurobi Model ---")
    
    try:
        start_time = time.time()
        model = run_subproblem_model(test_data)
        end_time = time.time()
        
        if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
            print("\n---  Optimization Success ---")
            print(f"Objective Value (Minimize Penalty): {model.ObjVal:.4f}")
            print(f"Time Taken: {end_time - start_time:.2f} seconds")
            
            # --- Extract Key Solution Variables for Validation ---
            
            print("\n---  ROUTING DECISIONS (Vehicle Path and Flow) ---")
            
            # 1. Routing (x - binary)
            for v in test_data["V"]:
                print(f"\n   **Vehicle {v} (Starts at S{test_data['eta'][v]})**")
                
                # Dictionary to store all departures: {departure_time: (from, to, arrival_time)}
                route_decisions = {}
                
                # Step 1: Find the initial move from Source (s) at t=0
                initial_station = test_data['eta'][v]
                var_name_s = f"x[{test_data['s']},{initial_station},{v},0]"
                x_var_s = model.getVarByName(var_name_s)
                
                if x_var_s and x_var_s.X > 0.5:
                    arr_t = 0 + test_data['T_DD'].get((test_data['s'], initial_station), 1)
                    route_decisions[0] = (test_data['s'], initial_station, arr_t)
                
                # Step 2: Find all subsequent moves (i -> j) starting from any station i
                for i in test_data["N"]:
                    for j in test_data["N"] + [test_data["d"]]:
                        for t_dep in range(0, T_val + 1):
                            var_name = f"x[{i},{j},{v},{t_dep}]"
                            x_var = model.getVarByName(var_name)
                            
                            if x_var and x_var.X > 0.5:
                                arrival_time = t_dep + test_data['T_DD'].get((i, j), 1)
                                if t_dep not in route_decisions or arrival_time < route_decisions[t_dep][2]:
                                    route_decisions[t_dep] = (i, j, arrival_time)
                
                # Print the final, ordered route
                if not route_decisions:
                    print("    No complete route found.")
                    continue
                
                ordered_path = []
                # Ensure the path is printed in time order (sorted by departure time t_dep)
                for t_dep in sorted(route_decisions.keys()):
                    frm, to, arr_t = route_decisions[t_dep]
                    
                    if frm == test_data['s']:
                        frm_str = 'Source (s)'
                    elif frm == test_data['d']:
                        frm_str = 'Sink (d)'
                    else:
                        frm_str = f'Station {frm}'

                    if to == test_data['s']:
                        to_str = 'Source (s)'
                    elif to == test_data['d']:
                        to_str = 'Sink (d)'
                    else:
                        to_str = f'Station {to}'
                        
                    ordered_path.append(f"{frm_str} -> {to_str} (Dep t={t_dep}, Arr t={arr_t})")
                
                print(f"    Full Route: {' -> '.join(ordered_path)}")

                # 2. Vehicle Load (qV - continuous) on selected arcs
                print("    **Bike Load (qV) on Route Arcs:**")
                
                for t_dep in sorted(route_decisions.keys()):
                    frm, to, arr_t = route_decisions[t_dep]
                    
                    var_name = f'qV[{frm},{to},{v},{t_dep}]'
                    qV_var = model.getVarByName(var_name)
                    
                    if qV_var and qV_var.X > 0.01:
                        print(f"      Load on Arc ({frm}->{to}, t={t_dep}): {qV_var.X:.2f} bikes")


            # 3. Loading/Unloading/Maintenance
            print("\n---  SERVICE DECISIONS (Load, Unload, Maintenance) ---")
            for i in test_data["N"]:
                I_N0 = test_data['I_N0'][i]
                I_T = test_data['I_T'][i]
                print(f"##  Station {i} | Initial: {I_N0}, Target: {I_T}")
                
                station_net_change = 0.0 # Track net change for the station
                
                for v in test_data["V"]:
                    # Calculate totals for V(v) at Station i
                    total_qL = sum(model.getVarByName(f"qL[{i},{v},{t}]").X for t in range(1, T_val+1) if model.getVarByName(f"qL[{i},{v},{t}]"))
                    total_qU = sum(model.getVarByName(f"qU[{i},{v},{t}]").X for t in range(1, T_val+1) if model.getVarByName(f"qU[{i},{v},{t}]"))
                    total_tM = sum(model.getVarByName(f"tM[{i},{v},{t}]").X for t in range(1, T_val+1) if model.getVarByName(f"tM[{i},{v},{t}]"))
                    
                    if total_qL > 0.01 or total_qU > 0.01 or total_tM > 0.01:
                        net_service = total_qU - total_qL # Positive = Net Bikes Delivered (Unload > Load)
                        station_net_change += net_service
                        
                        print(f"  ---  Vehicle {v} Action ---")
                        
                        if net_service > 0.01:
                            print(f"     **NET DELIVERY:** {net_service:.2f} bikes added to station.")
                        elif net_service < -0.01:
                            print(f"     **NET PICKUP:** {abs(net_service):.2f} bikes removed from station.")
                        else:
                            print(f"     **NET ZERO:** Picked up and delivered same amount.")
                        
                        # Detail the transaction
                        if total_qL > 0.01:
                            print(f"    - Loaded (qL): {total_qL:.2f} bikes")
                        if total_qU > 0.01:
                            print(f"    - Unloaded (qU): {total_qU:.2f} bikes")

                        # Maintenance detail
                        if total_tM > 0.01:
                            m_iv = model.getVarByName(f"m_iv[{i},{v}]")
                            print(f"     Maintenance: {total_tM:.2f} min (Reward: {total_tM * test_data['r_M']:.4f})")
                
                # Print the total effect of all vehicles at this station
                if abs(station_net_change) > 0.01:
                    total_inv = I_N0 + station_net_change
                    action_type = "ADDED" if station_net_change > 0 else "REMOVED"
                    print(f"  **TOTAL EFFECT:** {abs(station_net_change):.2f} bikes {action_type}. Inventory now: {total_inv:.2f}")

                print("-" * 30) # Separator for stations
                        
            # 4. Inventory and Penalties
            print("\n---  FINAL INVENTORY AND PENALTIES ---")
            for i in test_data["N"]:
                print(f"  Station {i}: (Target I_T={test_data['I_T'][i]})")
                
                # Inventory and Penalties at T=6
                lN_T = model.getVarByName(f"lN[{i},{T_val}]")
                d_abs = model.getVarByName(f"dev[{i}]")
                s_var = model.getVarByName(f"starv[{i},{T_val}]")
                c_var = model.getVarByName(f"cong[{i},{T_val}]")

                if lN_T:
                    print(f"    Final Inventory (lN_iT): {lN_T.X:.2f}")
                if d_abs and d_abs.X > 0.01:
                    print(f"    Deviation (d_abs): {d_abs.X:.2f} (Penalty: {d_abs.X * test_data['w_D']:.4f})")
                if s_var and s_var.X > 0.01:
                     print(f"    Starvation (s_var_iT): {s_var.X:.2f} (Penalty: {s_var.X * test_data['w_S']:.4f})")
                if c_var and c_var.X > 0.01:
                     print(f"    Congestion (c_var_iT): {c_var.X:.2f} (Penalty: {c_var.X * test_data['w_C']:.4f})")


        else:
            print(f"\n---  Optimization Failed (Status: {model.status}) ---")
            print("Possible reasons: Infeasible, Unbounded, or Time Limit exceeded.")
            
    except Exception as e:
        print(f"\n---  An error occurred during execution: {e} ---")