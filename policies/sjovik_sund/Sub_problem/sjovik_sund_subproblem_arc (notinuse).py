from gurobipy import *
import time
import numpy as np
 
#######################################################################################################
# This is the DSBRP subproblem for the Sjovik Sund policy - formulated with arcs
#######################################################################################################
 
def run_subproblem_arc_model(data):
 
    try:
        m = Model("DSBRP_Subproblem_arcformulation")
        m.setParam('OutputFlag', False)
        m.setParam('TimeLimit', 300)  # 5 minutes max
        m.setParam('MIPGap', 0.05)  # Stop at 5% gap (faster, good-enough solutions)
        m.setParam('Presolve', 2)  # Aggressive presolve
        m.setParam('MIPFocus', 1)  # Focus on finding good feasible solutions quickly
 
        ###########################################################################################################
        # SETS
        ############################################################################################################
 
        T  = int(data["T"])
        N = list(data["N"])
        s = data["s"]; d = data["d"]
        Vh = list(data["V"])
        N0 = N + [s, d]
        Tpos = list(range(1, T+1))   # {1,...,T}
        T0 = list(range(0, T+1))     # {0,...,T}
        
        # Get travel times early (needed for arc construction)
        T_DD = data["T_DD"]

        # Build the set of feasible arcs A based on travel times and time horizon
        # An arc (i,j) is feasible if:
        # 1. Travel time T_DD[i,j] exists
        # 2. The arc is physically possible (no arcs TO source, no arcs FROM sink, etc.)
        A = set()
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
                    # No self-loops at source or sink
                    if i == s and j == s:
                        continue
                    if i == d and j == d:
                        continue
                    
                    # Arc is feasible
                    A.add((i, j))
        
        # Build delta^+(i): set of incoming arcs to node i
        # delta^+(i) = {(j,i) ∈ A : arc ends at i}
        delta_in = {i: set() for i in N0}
        for (i_arc, j_arc) in A:
            delta_in[j_arc].add((i_arc, j_arc))
        
        # Build delta^-(i): set of outgoing arcs from node i
        # delta^-(i) = {(i,j) ∈ A : arc starts at i}
        delta_out = {i: set() for i in N0}
        for (i_arc, j_arc) in A:
            delta_out[i_arc].add((i_arc, j_arc))
        
        # Store in data structure for easy access
        data["A"] = list(A)
        data["delta_in"] = delta_in
        data["delta_out"] = delta_out
 
        ###########################################################################################################
        # PARAMETERS
        ############################################################################################################
 
        tau  = data["tau"]  # Period length in minutes (needed for time constraints)
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
     
        # Build list of feasible arc-vehicle-time tuples
        feasible_x_indices = []
        feasible_qV_indices = []
        
        for a in A:
            i, j = a
            travel_time = T_DD[(i, j)]
            for v in Vh:
                for t in T0:
                    if t + travel_time <= T:
                        if i == s and t > 0:
                            continue
                        if i in N and j == d and t == 0:
                            continue
                        if i == s and j == d and t == 0:
                            continue
                        
                        feasible_x_indices.append((a, v, t))
                        feasible_qV_indices.append((a, v, t))
        
        # Batch create variables using tupledict (much faster!)
        x = m.addVars(feasible_x_indices, vtype=GRB.BINARY, name="x")
        qV = m.addVars(feasible_qV_indices, lb=0.0, name="qV")
        
        # Helper functions
        def get_x(a, v, t):
            return x.get((a, v, t), 0)
        
        def get_qV(a, v, t):
            return qV.get((a, v, t), 0)
        
        # Maintenance selection m_iv ∈ {0,1}
        m_iv = m.addVars(N, Vh, vtype=GRB.BINARY, name="m_iv")

        # qL, qU ≥ 0 (integer unless relaxed), defined for i in N (stations only), v in V, t in Tpos
        qL = m.addVars(N, Vh, Tpos, lb=0.0, name="qL")
        qU = m.addVars(N, Vh, Tpos, lb=0.0, name="qU")
 
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
        
        
        # (2) Each vehicle departs from source at t=0 to its designated station eta^v
        # Ensures vehicle v goes from source s to station eta[v] at time 0
        for v in Vh:
            # Find the arc (s, eta[v]) in delta_out[s]
            target_station = eta[v]
            target_arc = None
            for a in delta_out[s]:
                if a[1] == target_station:  # a = (s, j), we want j == eta[v]
                    target_arc = a
                    break
            
            if target_arc is not None:
                m.addConstr(
                    get_x(target_arc, v, 0) == 1,
                    name=f"dep_source_v{v}_to_eta{target_station}"
                )
            else:
                # Fallback: if arc doesn't exist, allow any station (shouldn't happen). This is the previous formulation.
                m.addConstr(
                    quicksum(get_x(a, v, 0) for a in delta_out[s]) == 1,
                    name=f"dep_source_v{v}"
                )
            
            
 
        # (3) arrival at sink within horizon: ∑_a∈δ⁺(d) ∑_t x_{a v t} = 1
        for v in Vh:
            m.addConstr(
                quicksum(get_x(a, v, t) for a in delta_in[d] for t in Tpos) == 1, 
                name=f"arr_sink_v{v}"
            )
            
        # (4) vehicle flow conservation at stations j∈N with travel time delays
        # Using arc-indexed formulation with delta^+(j) and delta^-(j)
        # Apply to all time periods including t=0
        for v in Vh:
            for j in N:
                for t in T0:
                    # Inflow: vehicles arriving at j at time t using delta^+(j)
                    # Sum over all incoming arcs a=(i,j) considering travel time delay
                    inflow = quicksum(
                        get_x(a, v, t - T_DD[a]) 
                        for a in delta_in[j]
                        if t - T_DD[a] >= 0
                    )
                    
                    # Outflow: vehicles leaving j at time t using delta^-(j)
                    # Sum over all outgoing arcs a=(j,k)
                    outflow = quicksum(
                        get_x(a, v, t) 
                        for a in delta_out[j]
                    )
                    
                    m.addConstr(inflow == outflow, name=f"flow_v{v}_j{j}_t{t}")
 
        # (5) single trip per period: ∑_{a∈A} x_{a v t} ≤ 1 for each v,t∈Tpos
        for v in Vh:
            for t in Tpos:
                m.addConstr(
                    quicksum(get_x(a, v, t) for a in A) <= 1, 
                    name=f"one_trip_v{v}_t{t}"
                )
 
        # (6) single visit per station by entire fleet within horizon:
        #     ∑_{a∈δ⁺(j)} ∑_{v} ∑_{t} x_{a v t} ≤ 1  for each j∈N
        # Use T0 to include t=0 (initial arrival from source)
        for j in N:
            m.addConstr(
                quicksum(get_x(a, v, t) for a in delta_in[j] for v in Vh for t in T0 if a[0] != j) <= 1,
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
            m.addConstr(quicksum(get_qV(a, v, 0) for a in delta_out[s]) == Q_V0[v], name=f"init_vehicle_load_v{v}")
 
        # Helper: safe lookup of shifted t-index (t - T_DD_ij)
        def valid_tshift(t, i, j):
            return t - T_DD[(i, j)]
 
        # (11) vehicle inventory balance at nodes/times for usable bikes carried by vehicles:
        for i in N:
            for v in Vh:
                for t in Tpos:
                    incoming = quicksum(
                        get_qV(a, v, t - T_DD[a])
                        for a in delta_in[i]
                        if t - T_DD[a] >= 0
                    )
                    outgoing = quicksum(get_qV(a, v, t) for a in delta_out[i])
                    m.addConstr(
                        incoming - qU[i, v, t] + qL[i, v, t] == outgoing,
                        name=f"veh_load_bal_i{i}_v{v}_t{t}"
                    )
 
        # (12) cannot unload more than carried upon arrival:
        for i in N:
            for v in Vh:
                for t in Tpos:
                    incoming = quicksum(
                        get_qV(a, v, t - T_DD[a])
                        for a in delta_in[i]
                        if t - T_DD[a] >= 0
                    )
                    m.addConstr(qU[i, v, t] <= incoming, name=f"unload_le_incoming_i{i}_v{v}_t{t}")
 
        # (13) cannot load more than station inventory upon arrival
        for i in N:
            for v in Vh:
                for t in Tpos:
                    m.addConstr(qL[i, v, t] <= lN[i, t-1], name=f"load_le_inv_i{i}_v{v}_t{t}")
 
        # (14) vehicle capacity link
        for a in A:
            i, j = a
            for v in Vh:
                for t in T0:
                    if (a, v, t) in x:  # Only add constraint if variable exists
                        m.addConstr(get_qV(a, v, t) <= Q_V[v] * x[(a, v, t)],
                                    name=f"cap_link_i{i}_j{j}_v{v}_t{t}")
 
        # (14b) Explicit vehicle capacity after service operations
        # BURDE EGT IKKE TRENGE DETTE, DA DET ER IMPLISITT I ANDRE KONSTRANTER
        # Ensures that: incoming_load - unload + load <= vehicle_capacity
        # This prevents the vehicle from exceeding capacity after loading at a station
        for i in N:
            for v in Vh:
                for t in Tpos:
                    incoming = quicksum(
                        get_qV(a, v, t - T_DD[a])
                        for a in delta_in[i]
                        if t - T_DD[a] >= 0
                    )
                    m.addConstr(
                        incoming - qU[i, v, t] + qL[i, v, t] <= Q_V[v],
                        name=f"explicit_cap_i{i}_v{v}_t{t}"
                    )

        #############################################################################################################
        # Timing constraints & maintenance integration (15)–(20)
        ##############################################################################################################
 
 
        for v in Vh:
            for t in Tpos:
                # Upper bound (15) - Build the LHS expression using arc indexing
                travel_term = quicksum(
                    T_D[a] * get_x(a, v, t_prime - T_DD[a])
                    for t_prime in Tpos if t_prime <= t
                    for a in A if a[0] in (N + [s]) and a[1] in N  # Exclude sink from origin
                    if t_prime - T_DD[a] >= 0
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
                        <= 2 * tau * quicksum(get_x(a, v, t) for a in delta_out[i]),
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
 