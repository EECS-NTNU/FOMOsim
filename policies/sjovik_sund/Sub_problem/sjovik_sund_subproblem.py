from gurobipy import *
import time
import numpy as np
import json
import datetime
 
#######################################################################################################
# This is the DSBRP subproblem for the Sjovik Sund policy
#######################################################################################################
 
def run_subproblem_model(data):
 
    try:
        m = Model("DSBRP_Subproblem")
        m.setParam('OutputFlag', False)
        m.setParam('TimeLimit', 300)  # 5 minutes max
        m.setParam('MIPGap', 0.05)  # Stop at 5% gap (faster, good-enough solutions)
        m.setParam('Presolve', 2)  # Aggressive presolve
        m.setParam('MIPFocus', 1)  # Focus on finding good feasible solutions quickly
 
        ###########################################################################################################
        # SETS
        ############################################################################################################
 
        T = data["T"]
        tau = data["tau"]
        N = list(data["N"])
        s = data["s"]; d = data["d"]
        Vh = list(data["V"])
        N0 = N + [s, d]
        Tpos = list(range(1, T+1))   # {1,...,T}
        T0 = list(range(0, T+1))     # {0,...,T}
 
        ###########################################################################################################
        # PARAMETERS
        ############################################################################################################
 
        T_D  = data["T_D"]
        T_DD = data["T_DD"]
        T_L  = float(data["T_L"])
        T_M_min = data["T_M_min"]
        T_M_max = data["T_M_max"]
        Q_V  = data["Q_V"]
        Q_V0 = data["Q_V0"]
        Q_S  = data["Q_S"]
        I_N0 = data["I_N0"]
        I_T  = data["I_T"]
        D    = data["D"]
        w_S, w_C, w_D = data["w_S"], data["w_C"], data["w_D"]
        r_M = data["r_M"]
        eta = data["eta"]  # Initial destination station for each vehicle
 
        ###############################################################################################################
        # Sanity checks and preprocessing -> ensure T_DD[ii]=1 and T_D[ii]=0
        ################################################################################################################
     
        for i in N0:
            if (i,i) not in T_DD: T_DD[(i,i)] = 1
            if (i,i) not in T_D:  T_D[(i,i)]  = 0.0
 
        ###################################################################################################
        # Decision variables
        ###################################################################################################

        # Build list of feasible arcs first
        feasible_arcs = []

        # 1. Explicitly add Source -> Start Station arcs (Crucial!)
        for v in Vh:
            feasible_arcs.append((s, eta[v], v, 0))

        # 2. General Network Arcs (Station -> Station)
        for i in N:
            for j in N:
                travel_time = T_DD.get((i, j), None)
                if travel_time is not None:
                    for v in Vh:
                        for t in T0:
                            if t + travel_time <= T:
                                feasible_arcs.append((i, j, v, t))

        # 3. Sink Arcs (Station -> Sink)
        # Assuming 0 travel time to sink, allowed only at time T
        for j in N:
            for v in Vh:
                     feasible_arcs.append((j, d, v, T))

        # Debug: Print initialized nodes/arcs summary
        print("\n=== FEASIBLE ARCS SUMMARY ===")
        print(f"Total feasible arcs: {len(feasible_arcs)}")
        
        # Count arcs by type
        source_arcs = [a for a in feasible_arcs if a[0] == s]
        sink_arcs = [a for a in feasible_arcs if a[1] == d]
        network_arcs = [a for a in feasible_arcs if a[0] in N and a[1] in N]
        
        print(f"Source arcs (s->node): {len(source_arcs)}")
        print(f"Sink arcs (node->d): {len(sink_arcs)}")
        print(f"Network arcs (node->node): {len(network_arcs)}")
        
        # Print specific source arcs to verify initialization
        print("\nInitialized Source Arcs:")
        for arc in source_arcs:
            print(f"  {arc}")
            
        print("=============================\n")
        
        """
        for i in N0:
            for j in N0:
                travel_time = T_DD.get((i, j), None)
                if travel_time is not None:
                    if j == s or i == d:  # No arcs TO source, no arcs FROM sink
                        continue
                    for v in Vh:
                        for t in T0:
                            if t + travel_time <= T:
                                if i == s and t > 0:
                                    continue
                                if i in N and j == d and t == 0:
                                    continue
                                if i == s and j == d and t == 0:
                                    continue
                                feasible_arcs.append((i, j, v, t))
        """
        # Batch create x and qV variables using tupledict (much faster!)
        x = m.addVars(feasible_arcs, vtype=GRB.BINARY, name="x")
        qV = m.addVars(feasible_arcs, vtype = GRB.INTEGER, lb=0.0, name="qV")
        
        # Helper functions to safely access variables (returns 0 if variable doesn't exist)
        def get_x(i, j, v, t):
            return x.get((i, j, v, t), 0)
        
        def get_qV(i, j, v, t):
            return qV.get((i, j, v, t), 0)
        
        # Maintenance selection m_iv ∈ {0,1}
        m_iv = m.addVars(N, Vh, vtype=GRB.BINARY, name="m_iv")

        # qL, qU ≥ 0 (integer unless relaxed), defined for i in N (stations only), v in V, t in Tpos
        qL = m.addVars(N, Vh, Tpos, lb=0.0, vtype=GRB.INTEGER, name="qL")
        qU = m.addVars(N, Vh, Tpos, lb=0.0, vtype=GRB.INTEGER, name="qU")        

        # lN_it ≥ 0 for i in N, t in T0
        lN = m.addVars(N, T0, vtype=GRB.CONTINUOUS, lb=0.0, name="lN")
 
        # tM_ivt ≥ 0 continuous, for i in N, v in V, t in Tpos
        tM = m.addVars(N, Vh, Tpos, vtype=GRB.CONTINUOUS, lb=0.0, name="tM")
 
        # starvations / congestions ≥ 0 continuous
        s_var = m.addVars(N, Tpos, vtype=GRB.CONTINUOUS, lb=0.0, name="starv")
        c_var = m.addVars(N, Tpos, vtype=GRB.CONTINUOUS, lb=0.0, name="cong")
 
        # deviation di ≥ 0 continuous
        d_abs = m.addVars(N, vtype=GRB.CONTINUOUS, lb=0.0, name="dev")

         # Helper: export objective-term breakdown for analysis / thesis
        def _export_objective_breakdown(model, data, ts=None):
            try:
                if ts is None:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
 
                # Aggregate raw sums
                starv_sum = sum(s_var[i, t].X for i in N for t in Tpos)
                cong_sum = sum(c_var[i, t].X for i in N for t in Tpos)
                maint_time = sum(tM[i, v, t].X for i in N for v in Vh for t in Tpos)
                dev_sum = sum(d_abs[i].X for i in N)
 
                # Weighted contributions (match objective expression)
                starv_contrib = data.get('w_S', 1.0) * starv_sum
                cong_contrib = data.get('w_C', 1.0) * cong_sum
                dev_contrib = data.get('w_D', 1.0) * dev_sum
                maint_contrib = - data.get('r_M', 0.0) * maint_time
 
                obj_calc = starv_contrib + cong_contrib + dev_contrib + maint_contrib
                model_obj = float(model.ObjVal) if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL or model.Status == GRB.FEASIBLE else None
             
 
 
                out = {
                    'timestamp': ts,
                    'model_obj': model_obj,
                    'terms': {
                        'starvation': {'sum': starv_sum, 'weight': data.get('w_S', 1.0), 'contribution': starv_contrib},
                        'congestion': {'sum': cong_sum, 'weight': data.get('w_C', 1.0), 'contribution': cong_contrib},
                        'deviation': {'sum': dev_sum, 'weight': data.get('w_D', 1.0), 'contribution': dev_contrib},
                        'maintenance_time': {'sum': maint_time, 'rate': data.get('r_M', 0.0), 'contribution': maint_contrib} # i stedet for denne altså:
                    },
                    'computed_obj_from_terms': obj_calc
                }
                #denne kan gå inn over
                """'maintenance_time': {
                            'sum': maint_time,
                            'rate': data.get('r_M', 0.0),
                            'contribution_time': maint_contrib_time,
                            'events': maint_events,
                            'contribution_event': maint_contrib_event
                        }"""
 
                fname = f"subproblem_objective_breakdown_{ts}.json"
                with open(fname, 'w') as fh:
                    json.dump(out, fh, indent=2)
 
                # Print compact summary
                print('\n=== Objective Breakdown ===')
                print(f"Model objective: {model_obj}")
                print(f"Starvation contribution: {starv_contrib} (raw {starv_sum})")
                print(f"Congestion  contribution: {cong_contrib} (raw {cong_sum})")
                print(f"Deviation   contribution: {dev_contrib} (raw {dev_sum})")
                print(f"Maintenance contribution: {maint_contrib} (raw time {maint_time})")
                #print(f"Maintenance contribution (time-based): {maint_contrib_time} (raw time {maint_time})")
                #print(f"Maintenance contribution (per-event): {maint_contrib_event} (events {maint_events})")
                print(f"Sum of contributions: {obj_calc}")
                print(f"Wrote objective breakdown to {fname}\n")
            except Exception as e:
                print(f"Failed to export objective breakdown: {e}")
 
 
        # Helper: export objective-term breakdown for analysis / thesis
        def _export_objective_breakdown(model, data, ts=None):
            try:
                if ts is None:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                # Aggregate raw sums
                starv_sum = sum(s_var[i, t].X for i in N for t in Tpos)
                cong_sum = sum(c_var[i, t].X for i in N for t in Tpos)
                maint_time = sum(tM[i, v, t].X for i in N for v in Vh for t in Tpos)
                dev_sum = sum(d_abs[i].X for i in N)

                # Weighted contributions (match objective expression)
                starv_contrib = data.get('w_S', 1.0) * starv_sum
                cong_contrib = data.get('w_C', 1.0) * cong_sum
                dev_contrib = data.get('w_D', 1.0) * dev_sum
                maint_contrib = - data.get('r_M', 0.0) * maint_time

                obj_calc = starv_contrib + cong_contrib + dev_contrib + maint_contrib
                model_obj = float(model.ObjVal) if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL or model.Status == GRB.FEASIBLE else None
             
                out = {
                    'timestamp': ts,
                    'model_obj': model_obj,
                    'terms': {
                        'starvation': {'sum': starv_sum, 'weight': data.get('w_S', 1.0), 'contribution': starv_contrib},
                        'congestion': {'sum': cong_sum, 'weight': data.get('w_C', 1.0), 'contribution': cong_contrib},
                        'deviation': {'sum': dev_sum, 'weight': data.get('w_D', 1.0), 'contribution': dev_contrib},
                        'maintenance_time': {'sum': maint_time, 'rate': data.get('r_M', 0.0), 'contribution': maint_contrib}
                    },
                    'computed_obj_from_terms': obj_calc
                }

                fname = f"subproblem_objective_breakdown_{ts}.json"
                with open(fname, 'w') as fh:
                    json.dump(out, fh, indent=2)

                # Print compact summary
                print('\n=== Objective Breakdown ===')
                print(f"Model objective: {model_obj}")
                print(f"Starvation contribution: {starv_contrib} (raw {starv_sum})")
                print(f"Congestion  contribution: {cong_contrib} (raw {cong_sum})")
                print(f"Deviation   contribution: {dev_contrib} (raw {dev_sum})")
                print(f"Maintenance contribution: {maint_contrib} (raw time {maint_time})")
                print(f"Sum of contributions: {obj_calc}")
                print(f"Wrote objective breakdown to {fname}\n")
            except Exception as e:
                print(f"Failed to export objective breakdown: {e}")

        ###########################################################################################################
        # OBJECTIVE
        ###########################################################################################################
    
        m.setObjective(quicksum(
            quicksum(w_S*s_var[i,t] + w_C*c_var[i,t] - quicksum(r_M * tM[i,v,t] for v in Vh) for t in Tpos)+ w_D * d_abs[i]
            for i in N
        ),
        sense=GRB.MINIMIZE)
 
        ##########################################################################################
        # Routing constraints
        ############################################################################################
        
        
        # (2) Each vehicle departs from source at t=0 to its designated station eta^v
        # Ensures vehicle v goes from source s to station eta[v] at time 0: x_{s,eta^v,v,0} = 1
        for v in Vh:
            target_station = eta[v]
            m.addConstr(
                get_x(s, target_station, v, 0) == 1,
                name=f"dep_source_v{v}_to_eta{target_station}"
            )
 
        # (3) arrival at sink within horizon: ∑_i ∑_t x_{i d v t} = 1
        for v in Vh:
            #m.addConstr(quicksum(x[i, d, v, t] for i in N for t in Tpos) == 1, name=f"arr_sink_v{v}")
            m.addConstr(
                # quicksum(get_x(i, d, v, t) for i in N for t in Tpos) == 1, FJERNET FOR Å TILLATE SOURCE -> SINK
                quicksum(get_x(i, d, v, t) for i in (N + [s]) for t in T0) == 1, 
                name=f"arr_sink_v{v}"
            )
            
        # (4) vehicle flow conservation at stations j∈N with travel time delays
        #  Time-indexed flow conservation accounting for travel delays
        # Apply to all time periods including t=0
        for v in Vh:
            for j in N:
                for t in Tpos:  # Changed from Tpos to T0 to include t=0
                    # Inflow: vehicles arriving at j at time t (considering travel time from i to j)
                    inflow = quicksum(
                        #x[(i, j, v, t) - T_DD[(i, j)]]
                        get_x(i, j, v, t - T_DD[(i, j)]) 
                        for i in (N + [s])  # Only stations and source, not sink
                        if (i, j) in T_DD and t - T_DD[(i, j)] >= 0
                    )
                    # Outflow: vehicles leaving j at time t
                    outflow = quicksum(
                        #x[(i, j, v, t)]
                        get_x(j, k, v, t) 
                        for k in (N + [d])  # Only stations and sink, not source
                        if (j, k) in T_DD
                    )
                    
                    m.addConstr(inflow == outflow, name=f"flow_v{v}_j{j}_t{t}")
 
 
        # (5) single trip per period: ∑_{i,j∈N0} x_{i j v t} ≤ 1 for each v, t∈Tpos
        for v in Vh:
            for t in Tpos:
                #m.addConstr(quicksum(x[i, j, v, t] for i in N0 for j in N0) <= 1, name=f"one_trip_v{v}_t{t}")
                m.addConstr(
                    quicksum(get_x(i, j, v, t) for i in N0 for j in N0) <= 1, 
                    name=f"one_trip_v{v}_t{t}"
                )
 
        # (6) single visit per station by entire fleet within horizon:
        #     ∑_{i∈N\{j}} ∑_{v} ∑_{t} x_{i j v t} ≤ 1  for each j∈N
        # Use T0 to include t=0 (initial arrival from source)

        
        for j in N:
            for v in Vh:
                m.addConstr(
                    #quicksum(x[i, j, v, t] for i in N if i != j for v in Vh for t in Tpos) <= 1,
                    quicksum(get_x(i, j, v, t) for i in N0 if i != j for t in T0 for v in Vh) <= 1,
                    name=f"single_visit_j{j}"
                )        
        
        

        ##########################################################################################
        # Station inventory balance constraints
        ###########################################################################################
 
        # (7) inventory balance:
        for i in N:
            for t in Tpos:
                m.addConstr(
                    lN[i, t-1] + D[(i, t)] + quicksum(qU[i, v, t] - qL[i, v, t] for v in Vh) + s_var[i, t] - c_var[i, t]
                    == lN[i, t],
                    name=f"inv_bal_i{i}_t{t}"
                )
 
        # (8) initial inventory:
        for i in N:
            m.addConstr(lN[i, 0] == I_N0[i], name=f"init_inv_i{i}")
 
        # (9) capacity
        for i in N:
            for t in T0:
                m.addConstr(lN[i, t] <= Q_S[i], name=f"cap_i{i}_t{t}")
 
        #############################################################################################################
        # Vehicle loading, unloading, and capacity constraints
        ############################################################################################################
       
        # (10) initial vehicle load:
        """for v in Vh:
            m.addConstr(quicksum(get_qV(s, j, v, 0) for j in N0) == Q_V0[v], name=f"init_vehicle_load_v{v}")"""
        # (10) initial vehicle load:
        for v in Vh:
            target_station = eta[v]
            m.addConstr(get_qV(s, target_station, v, 0) == Q_V0[v], name=f"init_vehicle_load_v{v}")
 
        # Helper: safe lookup of shifted t-index (t - T_DD_ij)
        def valid_tshift(t, i, j):
            return t - T_DD[(i, j)]
 
        # (11) vehicle inventory balance at nodes/times for usable bikes carried by vehicles:
        for i in N:
            for v in Vh:
                for t in Tpos:
                    incoming = quicksum(
                        get_qV(j, i, v, valid_tshift(t, j, i))
                        for j in (N + [s])  # Exclude sink - no arcs FROM sink
                        if (j, i) in T_DD and valid_tshift(t, j, i) >= 0
                    )
                    outgoing = quicksum(get_qV(i, k, v, t) for k in (N + [d]) if (i, k) in T_DD)  # Exclude source - no arcs TO source
                    m.addConstr(
                        incoming - qU[i, v, t] + qL[i, v, t] == outgoing,
                        name=f"veh_load_bal_i{i}_v{v}_t{t}"
                    )
 
        # (12) cannot unload more than carried upon arrival:
        for i in N:
            for v in Vh:
                for t in Tpos:
                    incoming = quicksum(
                        get_qV(j, i, v, valid_tshift(t, j, i))
                        for j in (N + [s])  # Exclude sink - no arcs FROM sink
                        if (j, i) in T_DD and valid_tshift(t, j, i) >= 0
                    )
                    m.addConstr(qU[i, v, t] <= incoming, name=f"unload_le_incoming_i{i}_v{v}_t{t}")
 
        # (13) cannot load more than station inventory upon arrival
        for i in N:
            for v in Vh:
                for t in Tpos:
                    m.addConstr(qL[i, v, t] <= lN[i, t-1], name=f"load_le_inv_i{i}_v{v}_t{t}")
 
        # (14) vehicle capacity link
        for i in N0:
            for j in N0:
                for v in Vh:
                    for t in T0:
                        if (i, j, v, t) in feasible_arcs:  # Only add constraint if variable exists
                            m.addConstr(get_qV(i, j, v, t) <= Q_V[v] * get_x(i, j, v, t),
                                        name=f"cap_link_i{i}_j{j}_v{v}_t{t}")
    



        #############################################################################################################
        # Timing constraints & maintenance integration (15)–(20)
        ##############################################################################################################
        # Constraints (15) & (16): Time bounds - CORRECTED VERSION
        for v in Vh:
            for t in Tpos:
                # Calculate total time spent by vehicle v up to (and including) period t
                
                # 1. Completed travel segments (journeys that finished by period t)
                completed_travel = quicksum(
                    T_D[(i, j)] * get_x(i, j, v, t_prime - T_DD[(i, j)])
                    for t_prime in Tpos if t_prime <= t
                    for i in (N + [s]) for j in N
                    if (i, j) in T_DD and t_prime - T_DD[(i, j)] >= 0
                )
                
                # 2. Partial travel time (for journeys still in progress at period t)
                partial_travel = quicksum(
                    min(t * tau, T_D[(i, j)]) * get_x(i, j, v, start_time)
                    for start_time in T0 if start_time <= t
                    for i in (N + [s]) for j in N
                    if (i, j) in T_DD and start_time + T_DD[(i, j)] > t  # Journey not yet completed
                )
                
                # 3. Service time (loading, unloading, maintenance)
                service_term = quicksum(
                    T_L * (qL[i, v, t_prime] + qU[i, v, t_prime]) + tM[i, v, t_prime]
                    for t_prime in Tpos if t_prime <= t 
                    for i in N
                )
                
                # Upper bound (15): Total time ≤ available time
                m.addConstr(
                    completed_travel + partial_travel + service_term <= t * tau, 
                    name=f"time_ub_v{v}_t{t}"
                )
                
                # Lower bound (16): Total time ≥ minimum required time
                m.addConstr(
                    completed_travel + partial_travel + service_term >= (t - 1) * tau, 
                    name=f"time_lb_v{v}_t{t}"
                )
        
        # (17) Global maintenance upper bound per station
        for i in N:
            m.addConstr(quicksum(tM[i, v, t] for v in Vh for t in Tpos) <= T_M_max[i], name=f"maint_max_i{i}")
 
 
        # (18) Per vehicle/station minimum if maintenance chosen
        for i in N:
            for v in Vh:
                m.addConstr(quicksum(tM[i, v, t] for t in Tpos) >= T_M_min[i] * m_iv[i, v],
                            name=f"maint_min_i{i}_v{v}")

        # (18b) Rolling horizon: Only allow maintenance at current station (where vehicle starts)
        # This prevents rewarding phantom future maintenance that won't be executed
        for i in N:
            for v in Vh:
                #if i != eta[v]:  # If not the current station for this vehicle
                    #m.addConstr(m_iv[i, v] == 0, name=f"maint_current_only_i{i}_v{v}")
                m.addConstr(m_iv[i,v] <= quicksum(get_x(i, j, v, t) for j in N0 for t in Tpos), name=f"maint_current_only_i{i}_v{v}")    

        # (19) Link service (loading/unloading/maintenance) to presence in period t
        for i in N:
            for v in Vh:
                for t in Tpos:
                    m.addConstr(
                        T_L * (qL[i, v, t] + qU[i, v, t]) + tM[i, v, t]
                        #<= 2* tau * quicksum(x[i, j, v, t] for j in N0),
                        <= 2 * tau * quicksum(get_x(i, j, v, t) for j in N0),
                        name=f"service_presence_i{i}_v{v}_t{t}"
                    )
 
        # (20) Maintenance time upper bound link
        for i in N:
            for v in Vh:
                for t in Tpos:
                    m.addConstr(tM[i, v, t] <= 2 * tau * m_iv[i, v], name=f"maint_flag_i{i}_v{v}_t{t}")
 
        #############################################################################################################
        # Deviation absolute value at horizon (21)–(22)
        ##########################################################################################
 
        for i in N:
            m.addConstr(d_abs[i] >= I_T[i] - lN[i, T], name=f"dev_pos_i{i}")
            m.addConstr(d_abs[i] >= lN[i, T] - I_T[i], name=f"dev_neg_i{i}")
 
        ##########################################################################################
        # set some Gurobi parameters for speed/stability
        m.Params.OutputFlag = 1
 
        m.optimize()

        # Export objective-term breakdown when a solution (or incumbent) exists
        try:
            if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.USER_OBJ_LIMIT):
                _export_objective_breakdown(m, data)
        except Exception as e:
            # Best-effort: don't fail the solver wrapper if export breaks
            print("Warning: objective breakdown export failed.")
            print(str(e))
            print(traceback.format_exc())

        if m.Status == GRB.INFEASIBLE:
            print("\nModel is infeasible. Computing IIS...")
            m.computeIIS()
            m.write("model_iis.ilp")
            print("IIS written to model_iis.ilp")
            
            # Optional: Print the constraints in the IIS
            print("\nConstraints in IIS:")
            for c in m.getConstrs():
                if c.IISConstr:
                    print(f"  {c.ConstrName}")
            # Check bounds
            for v in m.getVars():
                 if v.IISLB > 0 or v.IISUB > 0:
                     print(f" Variable bound: {v.VarName}")
 
        # Return the model object so the policy can extract solution variables
        # obj_val = m.getObjective().getValue()
        return m
 
    except GurobiError as e:
        print("\n=== Gurobi Error ===")
        print(f"Error message: {e.message}")
        print(f"Error code: {e.errno if hasattr(e, 'errno') else 'N/A'}")
        print("="*50)
        raise  # Re-raise the error so we can see the full traceback
