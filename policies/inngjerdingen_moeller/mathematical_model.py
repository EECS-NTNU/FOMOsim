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

    T_D = data.T_D 
    T_DD = data.T_DD
    T_DW = data.T_DW
    T_DC = data.T_DC
    T_H = data.T_H
    
    L_0 = data.L_0
    L_T = data.L_T

    D = data.D

    Q_V = data.Q_V
    Q_S = data.Q_S


    
    #Variables
    c = m.addVar({(i, t) for i in stations for t in time_periods},lb=0, vtype=GRB.CONTINUOUS, name="c")
    s = m.addVar({(i, t) for i in stations for t in time_periods},lb=0, vtype=GRB.CONTINUOUS, name="s")
    d = m.addVar({(i, t) for i in stations for t in time_periods},lb=0, vtype=GRB.CONTINUOUS, name="d")

    r_B = m.addVar({(i, j, t) for i in stations for j in neighboring_stations[i] for t in time_periods},lb=0, vtype=GRB.CONTINUOUS, name="r_B")
    r_L = m.addVar({(i, j, t) for i in stations for j in neighboring_stations[i] for t in time_periods},lb=0, vtype=GRB.CONTINUOUS, name="r_L")

    q_L = m.addVar({(i, t, v) for i in stations for t in time_periods for v in vehicles},lb=0, vtype=GRB.CONTINUOUS, name="q_L")
    q_U = m.addVar({(i, t, v) for i in stations for t in time_periods for v in vehicles},lb=0, vtype=GRB.CONTINUOUS, name="q_U")
    q = m.addVar({(i, t, v) for i in stations for t in time_periods for v in vehicles},lb=0, vtype=GRB.CONTINUOUS, name="q")

    x = m.addVar({(i, j, t, v) for i in stations for j in stations for t in time_periods for v in vehicles},lb=0, vtype=GRB.BINARY, name="x")

    l = m.addVar({(i, t) for i in stations for t in time_periods},lb=0, vtype=GRB.CONTINUOUS, name="l") 


    #Constraints

    


    #Objective function

    
    return m
