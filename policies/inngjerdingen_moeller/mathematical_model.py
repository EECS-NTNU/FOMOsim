from gurobipy import *

def run_model(data):
    m = Model("MILP")

    #Sets
    stations = data.stations
    stations_with_source_sink = data-stations_with_source_sink
    neighboring_stations = data.neighboring_stations
    vehicles = data.vehicles
    time_periods = data.time_periods

    #Parameters
    W_D = data.W_D
    W_C = data.W_C
    W_S = data.W_S
    W_R = data.W_R

    T_bar = data.T_bar
    T_D = data.T_D 
    T_DD = data.T_DD
    T_W = data.T_W
    T_DW = data.T_DW
    T_C = data.T_C
    T_DC = data.T_DC
    T_H = data.T_H
    
    L_0 = data.L_0
    L_T = data.L_T

    D = data.D

    Q_0 = data.Q_0
    Q_V = data.Q_V
    Q_S = data.Q_S


    
    #Variables
    c = m.addVar({(i, t) for i in stations for t in time_periods},lb=0, vtype=GRB.CONTINUOUS, name="c")
    s = m.addVar({(i, t) for i in stations for t in time_periods},lb=0, vtype=GRB.CONTINUOUS, name="s")
    d = m.addVar({(i) for i in stations},lb=0, vtype=GRB.CONTINUOUS, name="d")

    r_B = m.addVar({(i, j, t) for i in stations for j in neighboring_stations[i] for t in time_periods},lb=0, vtype=GRB.CONTINUOUS, name="r_B")
    r_L = m.addVar({(i, j, t) for i in stations for j in neighboring_stations[i] for t in time_periods},lb=0, vtype=GRB.CONTINUOUS, name="r_L")

    q_L = m.addVar({(i, t, v) for i in stations for t in time_periods for v in vehicles},lb=0, vtype=GRB.CONTINUOUS, name="q_L")
    q_U = m.addVar({(i, t, v) for i in stations for t in time_periods for v in vehicles},lb=0, vtype=GRB.CONTINUOUS, name="q_U")
    q = m.addVar({(i, t, v) for i in stations for t in time_periods for v in vehicles},lb=0, vtype=GRB.CONTINUOUS, name="q")

    x = m.addVar({(i, j, t, v) for i in stations for j in stations for t in time_periods for v in vehicles},lb=0, vtype=GRB.BINARY, name="x")

    l = m.addVar({(i, t) for i in stations for t in time_periods},lb=0, vtype=GRB.CONTINUOUS, name="l") 


    #Constraints
    #Station balance:
    #m.addConstr(L_0[i]+quicksum(r_L))

    #Deviations:
    m.addConstr(d[(i)] >= L_T[i]-l[(i, T_bar)] for i in stations)
    m.addConstr(d[(i)] >= l[(i,T_bar)]-L_T[i] for i in stations)

    #Vehicle constraints
    m.addConstr(quicksum(x[(j, i, t-T_DD[i][j], v)] for j in stations_with_source_sink) == quicksum(x[(i, k, t, v)] for k in stations_with_source_sink) for i in stations for t in range(1,T_bar) for v in vehicles)
    m.addConstr(quickSum(x[(0, j, 0, v)] for j in stations_with_source_sink) == 1 for v in vehicles)
    m.addConstr(quicksum(x[(i, 0, T_bar-T_DD[i][0], v)] for i in stations_with_source_sink) == 1 for v in vehicles)

    m.addConstr(quicksum(q[(0, i, 0, v)] for i in stations) == Q_0[v] for v in vehicles)
    m.addConstr(quicksum(q[(j, i, t-T_DD[i][j], v)] for j in stations_with_source_sink) - q_U[(i, v, t)] + q_L[(i, v, t)] == quicksum(q[(i, k, t, v)] for k in stations_with_source_sink) for i in stations for t in range(1, T_bar+1) for v in vehicles)
    m.addConstr(q[(i, j, t, v)] <= Q_V[v]*x[(i, j, t, v)] for  i in stations for j in stations for t in time_periods for v in vehicles)

    

    #Objective function

    m.setObjective(quicksum(quicksum(W_C*c[(i, t)] + W_S*s[(i, t)] + quicksum(W_R*(T_W[i][j]*r_B[(i, j, t)]+T_C[i][j]*r_L[(i, j, t)]) for j in neighboring_stations[i]) for t in time_periods)+ W_D*d[(i)] for i in stations), GRB.MINIMIZE)

    m.optimize()
    
    return m
