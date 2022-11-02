import os
import sys
from pathlib import Path
path = Path(__file__).parents[2]     
os.chdir(path)
sys.path.insert(0, '') #make sure the modules are found in the new working directory

from gurobipy import *
from policies.inngjerdingen_moeller.parameters_MILP import *

def run_model(data):
    m = Model("MILP")

    #Sets
    stations = data.stations
    stations_with_source_sink = data.stations_with_source_sink
    neighboring_stations = data.neighboring_stations
    possible_previous_stations_driving = data.possible_previous_stations_driving
    possible_previous_stations_cycling = data.possible_previous_stations_cycling
    possible_previous_stations_walking = data.possible_previous_stations_walking

    vehicles = data.vehicles
    time_periods = data.time_periods

    #Parameters
    W_D = data.W_D
    W_C = data.W_C
    W_S = data.W_S
    W_R = data.W_R

    TW_max = data.TW_max
    T_bar = time_periods[-1]
    T_D = data.T_D 
    T_DD = data.T_DD
    T_W = data.T_W
    T_DW = data.T_DW
    T_C = data.T_C
    T_DC = data.T_DC
    T_H = data.T_H

    TAU= data.TAU
    DEPOT_ID= data.DEPOT_ID
    
    L_0 = data.L_0
    L_T = data.L_T

    D = data.D

    Q_0 = data.Q_0
    Q_V = data.Q_V
    Q_S = data.Q_S


    
    #Variables
    c = m.addVars({(i, t) for i in stations for t in range(1, T_bar+1)},lb=0, vtype=GRB.CONTINUOUS, name="c")
    s = m.addVars({(i, t) for i in stations for t  in range(1, T_bar+1)},lb=0, vtype=GRB.CONTINUOUS, name="s")
    d = m.addVars({i for i in stations},lb=0, vtype=GRB.CONTINUOUS, name="d")

    r_B = m.addVars({(i, j, t) for i in stations for j in neighboring_stations[i] for t in range(1, T_bar+1)},lb=0, vtype=GRB.CONTINUOUS, name="r_B")
    r_L = m.addVars({(i, j, t) for i in stations for j in stations if i != j for t in range(1, T_bar+1)},lb=0, vtype=GRB.CONTINUOUS, name="r_L")

    q_L = m.addVars({(i, t, v) for i in stations for t in range(1,T_bar+1) for v in vehicles},lb=0, vtype=GRB.INTEGER, name="q_L")
    q_U = m.addVars({(i, t, v) for i in stations for t in range(1,T_bar+1) for v in vehicles},lb=0, vtype=GRB.INTEGER, name="q_U")
    q = m.addVars({(i, j, t, v) for i in stations_with_source_sink for j in stations_with_source_sink for t in time_periods for v in vehicles},lb=0, vtype=GRB.CONTINUOUS, name="q")

    x = m.addVars({(i, j, t, v) for i in stations_with_source_sink for j in stations_with_source_sink for t in time_periods for v in vehicles},lb=0, vtype=GRB.BINARY, name="x")

    l = m.addVars({(i, t) for i in stations for t in time_periods},lb=0, vtype=GRB.CONTINUOUS, name="l") 


    #Constraints
    #Station balance:
    m.addConstrs(l[(i, 0)] == L_0[i] for i in stations)
    m.addConstrs(l[(i, t-1)] + quicksum(r_L[(j, i, t-T_DC[(j,i)])] for j in possible_previous_stations_cycling[(i,t)]) - quicksum(r_B[(j, i, t-T_DW[(j,i)])] for j in possible_previous_stations_walking[(i,t)]) - quicksum(r_L[(i, j, t)] for j in stations if j != i) + quicksum(r_B[(i, j, t)] for j in neighboring_stations[i]) + D[(i,t)] + quicksum(q_U[(i, t, v)] - q_L[(i, t, v)] for v in vehicles) + s[(i, t)] - c[(i, t)] == l[(i, t)] for t in range(1,T_bar+1) for i in stations)
    
    m.addConstrs(l[(i, t)] <= Q_S[i] for i in stations for t in time_periods)

    #Deviations:
    m.addConstrs(d[i] >= L_T[i]-l[(i, T_bar)] for i in stations) 
    m.addConstrs(d[i] >= l[(i,T_bar)]-L_T[i] for i in stations)

    #Vehicle constraints:
    #using the set with possible previous stations:
    m.addConstrs(quicksum(x[(j, i, t-T_DD[(j,i)], v)] for j in possible_previous_stations_driving[(i,t)]) == quicksum(x[(i, k, t, v)] for k in stations_with_source_sink) for i in stations for t in range(1,T_bar+1) for v in vehicles)
    # Hvis den kjører flere ganger inn kan den kjøre flere ganger ut
    m.addConstrs(quicksum(x[(i, j, t, v)] for i in stations_with_source_sink for j in stations_with_source_sink) <= 1 for t in time_periods for v in vehicles)
    m.addConstrs(quicksum(x[(DEPOT_ID, j, t, v)] for j in stations_with_source_sink for t in time_periods) == 1 for v in vehicles)

    m.addConstrs(quicksum(x[(DEPOT_ID, j, 0, v)] for j in stations_with_source_sink) == 1 for v in vehicles)
    m.addConstrs(quicksum(x[(i, DEPOT_ID, T_bar, v)] for i in stations_with_source_sink) == 1 for v in vehicles)

    m.addConstrs(quicksum(q[(DEPOT_ID, i, 0, v)] for i in stations) == Q_0[v] for v in vehicles)
    m.addConstrs(quicksum(q[(j, i, t-T_DD[(i,j)], v)] for j in possible_previous_stations_driving[(i,t)]) - q_U[(i, t, v)] + q_L[(i, t, v)] == quicksum(q[(i, k, t, v)] for k in stations_with_source_sink) for i in stations for t in range(1, T_bar+1) for v in vehicles)
    m.addConstrs(q[(i, j, t, v)] <= Q_V[v]*x[(i, j, t, v)] for  i in stations_with_source_sink for j in stations_with_source_sink for t in time_periods for v in vehicles)
    
    #Raviv constraints:
    m.addConstrs(quicksum(T_D[(i,j)]*x[(i, j, t_prime-T_DD[(i,j)], v)] for t_prime in range(1, t+1) for j in stations for i in possible_previous_stations_driving[(j,t_prime)]) + quicksum(T_H*(q_L[(j, t_prime, v)]+q_U[(j, t_prime, v)]) for j in stations for t_prime in range(1, t+1)) <= t*TAU for t in range(1, T_bar+1) for v in vehicles)
    m.addConstrs(quicksum(T_D[(i,j)]*x[(i, j, t_prime-T_DD[(i,j)], v)] for t_prime in range(1, t+1) for j in stations for i in possible_previous_stations_driving[(j,t_prime)]) + quicksum(T_H*(q_L[(j, t_prime, v)]+q_U[(j, t_prime, v)]) for j in stations for t_prime in range(1, t+1)) >= (t-2)*TAU for t in range(1, T_bar+1) for v in vehicles)
    
    #Loading/unloading quantities 
    m.addConstrs(q_L[(i, t, v)] <= 2*(TAU/T_H)*quicksum(x[(i, j, t, v)]for j in stations_with_source_sink) for i in stations for t in range(1, T_bar+1) for v in vehicles)
    m.addConstrs(q_L[(i, t, v)] <= Q_S[i]*quicksum(x[(i, j, t, v)]for j in stations_with_source_sink) for i in stations for t in range(1, T_bar+1) for v in vehicles)
    m.addConstrs(q_L[(i, t, v)] <= Q_V[v]*quicksum(x[(i, j, t, v)]for j in stations_with_source_sink) for i in stations for t in range(1, T_bar+1) for v in vehicles)
    m.addConstrs(q_U[(i, t, v)] <= 2*(TAU/T_H)*quicksum(x[(i, j, t, v)]for j in stations_with_source_sink) for i in stations for t in range(1, T_bar+1) for v in vehicles)
    m.addConstrs(q_U[(i, t, v)] <= Q_S[i]*quicksum(x[(i, j, t, v)]for j in stations_with_source_sink) for i in stations for t in range(1, T_bar+1) for v in vehicles)
    m.addConstrs(q_U[(i, t, v)] <= Q_V[v]*quicksum(x[(i, j, t, v)]for j in stations_with_source_sink) for i in stations for t in range(1, T_bar+1) for v in vehicles)
    
    #Objective function
    m.setObjective(quicksum(quicksum(W_C*c[(i, t)] + W_S*s[(i, t)] + quicksum(W_R*(T_W[(i,j)]/TW_max)*r_B[(i, j, t)] for j in neighboring_stations[i]) + quicksum(W_R*T_C[(i,j)]*r_L[(i, j, t)] for j in stations if j != i)  for t in range(1, T_bar+1))+ W_D*d[i] for i in stations), GRB.MINIMIZE)

    m.optimize()
    m.printAttr("X")
    return m

test_data = MILP_data()
test_data.initalize_parameters()

run_model(test_data)