from gurobipy import *
import time
import numpy as np
 
#######################################################################################################
# This is the DSBRP subproblem for the Sjovik Sund policy
#######################################################################################################
 
def run_subproblem_model(data):
 
    try:
        m = Model("DSBRP_Subproblem")
        m.setParam('OutputFlag', False)
        m.setParam('TimeLimit', 300)  # 5 minutes instead of 1 hour
 
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
 
        ###############################################################################################################
        # Sanity checks and preprocessing -> ensure T_DD[ii]=1 and T_D[ii]=0
        ################################################################################################################
     
        for i in N0:
            if (i,i) not in T_DD: T_DD[(i,i)] = 1
            if (i,i) not in T_D:  T_D[(i,i)]  = 0.0
 
        ###################################################################################################
        # Decision variables
        ###################################################################################################
     
        # Routing selection x_ijvt ∈ {0,1}
        #x = m.addVars(N0, N0, Vh, T0, vtype=GRB.BINARY, name="x")
        # create only possible x variables, i.e., where travel time fits in horizon
        # this means arcs where t + T_DD_ij <= T and the one that are from source to the station of the vehicle triggering th subproblem and those in transit
        # Create only valid arcs that:
        # 1. Have defined travel times in T_DD
        # 2. Fit within time horizon (t + T_DD_ij <= T)
        # 3. Are physically possible (no arcs TO source, no arcs FROM sink, etc.)
        
        x = {}
        for i in N0:
            for j in N0:
                travel_time = T_DD.get((i, j), None)
                if travel_time is not None:
                    # Filter out physically impossible arcs
                    # No arcs TO source (source is only for departures)
                    if j == s:
                        continue
                    # No arcs FROM sink (sink is only for arrivals)
                    if i == d:
                        continue
                    
                    for v in Vh:
                        for t in T0:
                            # Arc must complete within horizon
                            if t + travel_time <= T:
                                # Source can only be used at t=0
                                if i == s and t > 0:
                                    continue
                                # Stations cannot go to sink at t=0 (must make at least one move)
                                if i in N and j == d and t == 0:
                                    continue
                                # Source cannot go directly to sink at t=0
                                if i == s and j == d and t == 0:
                                    continue
                                
                                x[(i, j, v, t)] = m.addVar(vtype=GRB.BINARY, name=f"x[{i},{j},{v},{t}]")
        
        # Helper function to safely access x variables (returns 0 if variable doesn't exist)
        def get_x(i, j, v, t):
            return x.get((i, j, v, t), 0)   
        
        
        
        # Maintenance selection m_iv ∈ {0,1}
        m_iv = m.addVars(N, Vh, vtype=GRB.BINARY, name="m_iv")
 
        # qL, qU ≥ 0 (integer unless relaxed), defined for i in N (stations only), v in V, t in Tpos
        qL = m.addVars(N, Vh, Tpos, lb=0.0, name="qL")
        qU = m.addVars(N, Vh, Tpos, lb=0.0, name="qU")
 
        # qV_ijvt ≥ 0 (integer unless relaxed), for i,j in N0, v in V, t in T0
        qV = m.addVars(N0, N0, Vh, T0, lb=0.0, name="qV")
 
        # lN_it ≥ 0 for i in N, t in T0
        lN = m.addVars(N, T0, vtype=GRB.CONTINUOUS, lb=0.0, name="lN")
 
        # tM_ivt ≥ 0 continuous, for i in N, v in V, t in Tpos
        tM = m.addVars(N, Vh, Tpos, vtype=GRB.CONTINUOUS, lb=0.0, name="tM")
 
        # starvations / congestions ≥ 0 continuous
        s_var = m.addVars(N, Tpos, vtype=GRB.CONTINUOUS, lb=0.0, name="starv")
        c_var = m.addVars(N, Tpos, vtype=GRB.CONTINUOUS, lb=0.0, name="cong")
 
        # deviation di ≥ 0 continuous
        d_abs = m.addVars(N, vtype=GRB.CONTINUOUS, lb=0.0, name="dev")
 
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
        
        """# (1) Prevent physically impossible arcs
        # No arcs TO the source (source is only for departures)
        for v in Vh:
            for i in N0:
                for t in T0:
                    m.addConstr(x[i, s, v, t] == 0, name=f"no_arc_to_source_i{i}_v{v}_t{t}")
        
        # No arcs FROM the sink (sink is only for arrivals)
        for v in Vh:
            for j in N0:
                for t in T0:
                    m.addConstr(x[d, j, v, t] == 0, name=f"no_arc_from_sink_j{j}_v{v}_t{t}")
        
        # Source can only be used at t=0 (initial departure)
        for v in Vh:
            for j in N0:
                for t in Tpos:
                    m.addConstr(x[s, j, v, t] == 0, name=f"source_only_t0_j{j}_v{v}_t{t}")
        
        # Stations cannot go directly to sink at t=0 (vehicle must make at least one move)
        for v in Vh:
            for i in N:
                m.addConstr(x[i, d, v, 0] == 0, name=f"no_station_to_sink_t0_i{i}_v{v}")
        
        # Source cannot go directly to sink at t=0 (vehicle must visit at least one station)
        for v in Vh:
            m.addConstr(x[s, d, v, 0] == 0, name=f"no_source_to_sink_t0_v{v}")"""
        
        # (2) Each vehicle departs from source at t=0 to exactly one station
        for v in Vh:
            m.addConstr(
                quicksum(get_x(s, j, v, 0) for j in N) == 1, 
                name=f"dep_source_v{v}"
            )
            
            
 
        # (3) arrival at sink within horizon: ∑_i ∑_t x_{i d v t} = 1
        for v in Vh:
            #m.addConstr(quicksum(x[i, d, v, t] for i in N for t in Tpos) == 1, name=f"arr_sink_v{v}")
            m.addConstr(
                quicksum(get_x(i, d, v, t) for i in N for t in Tpos) == 1, 
                name=f"arr_sink_v{v}"
            )
        # (4) vehicle flow conservation at stations j∈N with travel time delays
        # OLD (WRONG): ∑_i ∑_t x_{i j v t} = ∑_k ∑_t x_{j k v t}
        # for v in Vh:
        #     for j in N:
        #         m.addConstr(
        #             quicksum(x[i, j, v, t] for i in N for t in Tpos) ==
        #             quicksum(x[j, k, v, t] for k in N for t in Tpos),
        #             name=f"flow_v{v}_j{j}"
        #         )
        
        # NEW: Time-indexed flow conservation accounting for travel delays
        # Apply to all time periods including t=0
        for v in Vh:
            for j in N:
                for t in T0:  # Changed from Tpos to T0 to include t=0
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
 
        # (4b) Each vehicle must depart from source exactly once
        # This ensures the vehicle starts at its current location (represented by source node)
        for v in Vh:
            m.addConstr(
                #quicksum(x[s, j, v, t] for j in N for t in T0) == 1,
                quicksum(get_x(s, j, v, t) for j in N for t in T0) == 1,
                name=f"depart_source_v{v}"
            )
 
        # (5) single trip per period: ∑_{i,j∈N0} x_{i j v t} ≤ 1 for each v,t∈Tpos
        for v in Vh:
            for t in Tpos:
                #m.addConstr(quicksum(x[i, j, v, t] for i in N0 for j in N0) <= 1, name=f"one_trip_v{v}_t{t}")
                m.addConstr(
                    quicksum(get_x(i, j, v, t) for i in N0 for j in N0) <= 1, 
                    name=f"one_trip_v{v}_t{t}"
                )
 
        # (6) single visit per station by entire fleet within horizon:
        #     ∑_{i∈N\{j}} ∑_{v} ∑_{t} x_{i j v t} ≤ 1  for each j∈N
        for j in N:
            m.addConstr(
                #quicksum(x[i, j, v, t] for i in N if i != j for v in Vh for t in Tpos) <= 1,
                quicksum(get_x(i, j, v, t) for i in N if i != j for v in Vh for t in Tpos) <= 1,
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
        for v in Vh:
            m.addConstr(quicksum(qV[s, j, v, 0] for j in N0) == Q_V0[v], name=f"init_vehicle_load_v{v}")
 
        # Helper: safe lookup of shifted t-index (t - T_DD_ij)
        def valid_tshift(t, i, j):
            return t - T_DD[(i, j)]
 
        # (11) vehicle inventory balance at nodes/times for usable bikes carried by vehicles:
        for i in N:
            for v in Vh:
                for t in Tpos:
                    incoming = quicksum(
                        qV[j, i, v, valid_tshift(t, j, i)]
                        for j in (N + [s])  # Exclude sink - no arcs FROM sink
                        if (j, i) in T_DD and valid_tshift(t, j, i) >= 0
                    )
                    outgoing = quicksum(qV[i, k, v, t] for k in (N + [d]) if (i, k) in T_DD)  # Exclude source - no arcs TO source
                    m.addConstr(
                        incoming - qU[i, v, t] + qL[i, v, t] == outgoing,
                        name=f"veh_load_bal_i{i}_v{v}_t{t}"
                    )
 
        # (12) cannot unload more than carried upon arrival:
        for i in N:
            for v in Vh:
                for t in Tpos:
                    incoming = quicksum(
                        qV[j, i, v, valid_tshift(t, j, i)]
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
                        if (i, j, v, t) in x:  # Only add constraint if variable exists
                            m.addConstr(qV[i, j, v, t] <= Q_V[v] * x[(i, j, v, t)],
                                        name=f"cap_link_i{i}_j{j}_v{v}_t{t}")
 
        #############################################################################################################
        # Timing constraints & maintenance integration (15)–(20)
        ##############################################################################################################
 
 
        for v in Vh:
            for t in Tpos:
                # Upper bound (15) - Build the LHS expression
                travel_term = quicksum(
                    #T_D[(i, j)] * x[(i, j, v, t_prime - T_DD[(i, j)])]
                    T_D[(i, j)] * get_x(i, j, v, t_prime - T_DD[(i, j)])
                    for t_prime in Tpos if t_prime <= t
                    for i in (N + [s]) for j in N  # Exclude sink from i - no arcs FROM sink
                    #if t_prime - T_DD[(i, j)] >= 0
                    if (i, j) in T_DD and t_prime - T_DD[(i, j)] >= 0
                )
                service_term = quicksum(
                    T_L * (qL[i, v, t_prime] + qU[i, v, t_prime]) + tM[i, v, t_prime]
                    for t_prime in Tpos if t_prime <= t for i in N
                )
                m.addConstr(travel_term + service_term <= t * tau, name=f"time_ub_v{v}_t{t}")
 
                # Lower bound (16) - Build the LHS expression
                m.addConstr(travel_term + service_term >= (t - 2) * tau, name=f"time_lb_v{v}_t{t}")
 
 
        # (17) Global maintenance upper bound per station
        for i in N:
            m.addConstr(quicksum(tM[i, v, t] for v in Vh for t in Tpos) <= T_M_max[i], name=f"maint_max_i{i}")
 
 
        # (18) Per vehicle/station minimum if maintenance chosen
        for i in N:
            for v in Vh:
                m.addConstr(quicksum(tM[i, v, t] for t in Tpos) >= T_M_min[i] * m_iv[i, v],
                            name=f"maint_min_i{i}_v{v}")
 
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
 
        # Return the model object so the policy can extract solution variables
        # obj_val = m.getObjective().getValue()
        return m
 
    except GurobiError as e:
        print("\n=== Gurobi Error ===")
        print(f"Error message: {e.message}")
        print(f"Error code: {e.errno if hasattr(e, 'errno') else 'N/A'}")
        print("="*50)
        raise  # Re-raise the error so we can see the full traceback
 