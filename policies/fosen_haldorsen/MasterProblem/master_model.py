from gurobipy import *

def run_master_model(parameters):

    try:
        m = Model("Heuristic Master")
        m.setParam('OutputFlag', False)
        m.setParam('TimeLimit', 60 * 60)

        # ------ SETS -----------------------------------------------------------------------------
        Stations = parameters.stations
        Swap_Stations = parameters.swap_stations
        Vehicles = parameters.vehicles
        Routes = parameters.routes
        Patterns = parameters.patterns
        Scenarios = parameters.scenarios

        # ------ PARAMETERS -----------------------------------------------------------------------
        A = parameters.origin_matrix
        start_st = parameters.starting_stations
        R = parameters.subproblem_objectives
        P = parameters.scenario_probabilities
        Q_FCL = parameters.pattern_fcl
        Q_CCL = parameters.pattern_ccl
        Q_FCU = parameters.pattern_fcu
        Q_CCU = parameters.pattern_ccu
        Q_B = parameters.pattern_b
        Q_CV = parameters.vehicle_bike_caps
        Q_S = parameters.station_caps
        L_BV = parameters.init_battery_load
        L_FV = parameters.init_flat_bike_load
        L_CV = parameters.init_charged_bike_load
        L_FS = parameters.init_flat_station_load
        L_CS = parameters.init_charged_station_load

        # ------ VARIABLES ------------------------------------------------------------------------
        lam = m.addVars({(v, r, p, s) for v in Vehicles for r in Routes[v] for p in Patterns[v] for s in
                         Scenarios},
                        vtype=GRB.CONTINUOUS, lb=0, ub=1, name="lam")

        x = m.addVars({(i, v, s) for i in Stations for v in Vehicles for s in Scenarios}, vtype=GRB.CONTINUOUS,
                      name="x")

        x_nac = m.addVars({(i,v) for i in Stations for v in Vehicles}, vtype=GRB.BINARY, lb=0, name="x_nac")
        q_CCL_nac = m.addVars({(v) for v in Vehicles}, vtype=GRB.INTEGER, lb=0,
                             name="q_CCL_nac")
        q_FCL_nac = m.addVars({(v) for v in Vehicles}, vtype=GRB.INTEGER, lb=0,
                              name="q_FCL_nac")
        q_CCU_nac = m.addVars({(v) for v in Vehicles}, vtype=GRB.INTEGER, lb=0,
                             name="q_CCU_nac")
        q_FCU_nac = m.addVars({(v) for v in Vehicles}, vtype=GRB.INTEGER, lb=0,
                             name="q_FCU_nac")
        q_B_nac = m.addVars({(v) for v in Vehicles}, vtype=GRB.INTEGER, lb=0,
                             name="q_B_nac")

        # ------ CONSTRAINTS -----------------------------------------------------------------------
        m.addConstrs(quicksum(A[v][r][i] * quicksum(lam[(v, r, p, s)] for p in Patterns[v]) for r in Routes[v])
                     == x[(i, v, s)] for i in Stations for v in Vehicles for s in Scenarios)

        m.addConstrs(x.sum('*', v, s) <= 1 for v in Vehicles for s in Scenarios)
        m.addConstrs(x.sum(i, '*', s) <= 1 for i in Swap_Stations for s in Scenarios)

        m.addConstrs(
            quicksum(lam[(v, r, p, s)] * Q_FCL[v][r][p] for r in Routes[v] for p in Patterns[v]) == q_FCL_nac[v] for
            v in Vehicles for s in Scenarios)
        m.addConstrs(
            quicksum(lam[(v, r, p, s)] * Q_CCL[v][r][p] for r in Routes[v] for p in Patterns[v]) == q_CCL_nac[v] for
            v in Vehicles for s in Scenarios)
        m.addConstrs(
            quicksum(lam[(v, r, p, s)] * Q_FCU[v][r][p] for r in Routes[v] for p in Patterns[v]) == q_FCU_nac[v] for
            v in Vehicles for s in Scenarios)
        m.addConstrs(
            quicksum(lam[(v, r, p, s)] * Q_CCU[v][r][p] for r in Routes[v] for p in Patterns[v]) == q_CCU_nac[v] for
            v in Vehicles for s in Scenarios)
        m.addConstrs(
            quicksum(lam[(v, r, p, s)] * Q_B[v][r][p] for r in Routes[v] for p in Patterns[v]) == q_B_nac[v] for v in
            Vehicles for s in Scenarios)
        m.addConstrs(quicksum(lam[(v, r, p, s)] for p in Patterns[v] for r in Routes[v]) == 1
                     for v in Vehicles for s in Scenarios)

        # Secure that first move is legal in terms of capacities
        m.addConstrs(q_CCL_nac[v] + q_FCL_nac[v] <= Q_CV[v] - L_FV[v] - L_CV[v] + q_CCU_nac[v] + q_FCU_nac[v] for v in
                    Vehicles)
        m.addConstrs(q_CCU_nac[v] + q_FCU_nac[v] <= Q_S[start_st[v]] - L_CS[start_st[v]] - L_FS[start_st[v]] + q_CCL_nac[v]
                     + q_FCL_nac[v] for v in Vehicles)
        m.addConstrs(q_B_nac[v] <= L_FS[start_st[v]] + q_FCU_nac[v] - q_FCL_nac[v] for v in Vehicles)
        m.addConstrs(q_B_nac[v] <= L_BV[v] for v in Vehicles)
        m.addConstrs(q_FCL_nac[v] <= L_FS[start_st[v]] for v in Vehicles)
        m.addConstrs(q_CCL_nac[v] <= L_CS[start_st[v]] for v in Vehicles)
        m.addConstrs(q_CCU_nac[v] <= L_CV[v] for v in Vehicles)
        m.addConstrs(q_FCU_nac[v] <= L_FV[v] for v in Vehicles)

        # Non-anticipativity constraints for x
        m.addConstrs(x_nac[(i, v)] == x[(i, v, s)] for i in Stations for v in Vehicles for s in Scenarios)
        m.addConstrs(x_nac.sum('*', v) == 1 for v in Vehicles)

        # ------ OBJECTIVE -----------------------------------------------------------------------
        m.setObjective(quicksum(P[s] * R[v][r][p][s] * lam[(v, r, p, s)] for s in Scenarios
                                for v in Vehicles for r in Routes[v] for p in Patterns[v]), GRB.MAXIMIZE)

        m.optimize()

        return m

    except GurobiError:
        print("\nGurobi error\n")
        print(GurobiError.message)
