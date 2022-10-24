from gurobipy import *
from data_MILP import *

def run_model(data):
    m = Model("MILP")

    #Sets
    stations = data.stations
    stations_with_source_sink = data.stations_with_source_sink
    neighboring_stations = data.neighboring_stations
    vehicles = data.vehicles
    time_periods = data.time_periods

    #Parameters
    W_D = data.W_D
    W_C = data.W_C
    W_S = data.W_S
    W_R = data.W_R

    T_bar = time_periods[-1]
    T_D = data.T_D 
    T_DD = data.T_DD
    T_W = data.T_W
    T_DW = data.T_DW
    T_C = data.T_C
    T_DC = data.T_DC
    T_H = data.T_H
    tau= data.tau
    
    L_0 = data.L_0
    L_T = data.L_T

    D = data.D

    Q_0 = data.Q_0
    Q_V = data.Q_V
    Q_S = data.Q_S


    
    #Variables
    c = m.addVars({(i, t) for i in stations for t in time_periods},lb=0, vtype=GRB.CONTINUOUS, name="c")
    s = m.addVars({(i, t) for i in stations for t in time_periods},lb=0, vtype=GRB.CONTINUOUS, name="s")
    d = m.addVars({(i) for i in stations},lb=0, vtype=GRB.CONTINUOUS, name="d")

    r_B = m.addVars({(i, j, t) for i in stations for j in neighboring_stations[i] for t in time_periods},lb=0, vtype=GRB.CONTINUOUS, name="r_B")
    r_L = m.addVars({(i, j, t) for i in stations for j in neighboring_stations[i] for t in time_periods},lb=0, vtype=GRB.CONTINUOUS, name="r_L")

    q_L = m.addVars({(i, t, v) for i in stations for t in time_periods for v in vehicles},lb=0, vtype=GRB.CONTINUOUS, name="q_L")
    q_U = m.addVars({(i, t, v) for i in stations for t in time_periods for v in vehicles},lb=0, vtype=GRB.CONTINUOUS, name="q_U")
    # ChECK INDICES FOR q
    q = m.addVars({(i, t, v) for i in stations_with_source_sink for t in time_periods for v in vehicles},lb=0, vtype=GRB.CONTINUOUS, name="q")

    x = m.addVars({(i, j, t, v) for i in stations_with_source_sink for j in stations_with_source_sink for t in time_periods for v in vehicles},lb=0, vtype=GRB.BINARY, name="x")

    l = m.addVars({(i, t) for i in stations for t in time_periods},lb=0, vtype=GRB.CONTINUOUS, name="l") 


    #Constraints
    #Station balance:
    #m.addConstrs(L_0[i]+quicksum(r_L[(j, i, t-T_DC[i][j])]-r_B[(j, i, t-T_DW[i][j])]-r_L[(i, j, t)]+r_B[(i, j, t)] for j in neighboring_stations[i])+D[i][t]+quicksum(q_U[(i, v, t)]-q_L[(i, v, t)] for v in vehicles) + s[(i, t)] - c[(i, t)] == l[(i, t)] for t in range(1,2) for i in stations)
    #m.addConstrs(l[(i, t-1)]+quicksum(r_L[(j, i, t-T_DC[i][j])]-r_B[(j, i, t-T_DW[i][j])]-r_L[(i, j, t)]+r_B[(i, j, t)] for j in neighboring_stations[i])+D[i][t]+quicksum(q_U[(i, v, t)]-q_L[(i, v, t)] for v in vehicles) + s[(i, t)] - c[(i, t)] == l[(i, t)] for t in range(2,T_bar+1) for i in stations)
    
    m.addConstrs(l[(i, t)] <= Q_S[i] for i in stations for t in time_periods)

    #Deviations:
    m.addConstrs(d[(i)] >= L_T[i]-l[(i, T_bar)] for i in stations)
    m.addConstrs(d[(i)] >= l[(i,T_bar)]-L_T[i] for i in stations)

    #Vehicle constraints:
    m.addConstrs(quicksum(x[(j, i, t-T_DD[i][j], v)] for j in stations_with_source_sink) == quicksum(x[(i, k, t, v)] for k in stations_with_source_sink) for i in stations for t in range(1,T_bar) for v in vehicles)
    m.addConstrs(quicksum(x[(0, j, 0, v)] for j in stations_with_source_sink) == 1 for v in vehicles)
    m.addConstrs(quicksum(x[(i, 0, T_bar-T_DD[i][0], v)] for i in stations_with_source_sink) == 1 for v in vehicles)

    m.addConstrs(quicksum(q[(0, i, 0, v)] for i in stations) == Q_0[v] for v in vehicles)
    m.addConstrs(quicksum(q[(j, i, t-T_DD[i][j], v)] for j in stations_with_source_sink) - q_U[(i, v, t)] + q_L[(i, v, t)] == quicksum(q[(i, k, t, v)] for k in stations_with_source_sink) for i in stations for t in range(1, T_bar+1) for v in vehicles)
    m.addConstrs(q[(i, j, t, v)] <= Q_V[v]*x[(i, j, t, v)] for  i in stations for j in stations for t in time_periods for v in vehicles)
    
    #Raviv constraints:
    m.addConstrs(quicksum(T_D[i][j]*x[(i, j, t-T_DD[i][j], v)] for i in stations_with_source_sink for j in stations_with_source_sink for t in range(0, t_marked+1))+quicksum(T_H*(q_L[(i, t, v)]+q_U[(i, t, v)]) for i in stations_with_source_sink for t in range(0, t_marked+1)) <= t_marked*tau for t_marked in range(1, T_bar+1) for v in vehicles)
    m.addConstrs(quicksum(T_D[i][j]*x[(i, j, t-T_DD[i][j], v)] for i in stations_with_source_sink for j in stations_with_source_sink for t in range(0, t_marked+1))+quicksum(T_H*(q_L[(i, t, v)]+q_U[(i, t, v)]) for i in stations_with_source_sink for t in range(0, t_marked+1)) >= (t_marked-2)*tau for t_marked in range(1, T_bar+1) for v in vehicles)
    
    #Loading/unloading quantities 
    m.addConstrs(q_L[(i, t, v)] <= 2*T_H*quicksum(x[(i, j, t, v)]for j in stations_with_source_sink) for i in stations_with_source_sink for t in range(1, T_bar+1) for v in vehicles)
    m.addConstrs(q_L[(i, t, v)] <= Q_S[i]*quicksum(x[(i, j, t, v)]for j in stations_with_source_sink) for i in stations_with_source_sink for t in range(1, T_bar+1) for v in vehicles)
    m.addConstrs(q_L[(i, t, v)] <= Q_V[v]*quicksum(x[(i, j, t, v)]for j in stations_with_source_sink) for i in stations_with_source_sink for t in range(1, T_bar+1) for v in vehicles)
    m.addConstrs(q_U[(i, t, v)] <= 2*T_H*quicksum(x[(i, j, t, v)]for j in stations_with_source_sink) for i in stations_with_source_sink for t in range(1, T_bar+1) for v in vehicles)
    m.addConstrs(q_U[(i, t, v)] <= Q_S[i]*quicksum(x[(i, j, t, v)]for j in stations_with_source_sink) for i in stations_with_source_sink for t in range(1, T_bar+1) for v in vehicles)
    m.addConstrs(q_U[(i, t, v)] <= Q_V[v]*quicksum(x[(i, j, t, v)]for j in stations_with_source_sink) for i in stations_with_source_sink for t in range(1, T_bar+1) for v in vehicles)
    
    #Objective function
    m.setObjective(quicksum(quicksum(W_C*c[(i, t)] + W_S*s[(i, t)] + quicksum(W_R*(T_W[i][j]*r_B[(i, j, t)]+T_C[i][j]*r_L[(i, j, t)]) for j in neighboring_stations[i]) for t in time_periods)+ W_D*d[(i)] for i in stations), GRB.MINIMIZE)

    m.optimize()
    print("Ayy vi kom gjennom")
    return m

test_data = MILP_data()
run_model(test_data)