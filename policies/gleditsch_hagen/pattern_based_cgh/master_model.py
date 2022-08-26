from gurobipy import *
import time
import numpy as np


def run_master_model(data):

    
    m = Model("Master")
    m.setParam('OutputFlag', False)  #To do: should be at False.
    
    
    #testing
    # for i in range(len(data.generated_routes)):
    #     print('Route: ', i)
    #     print('Vehicle: ', data.generated_routes[i].vehicle.id)
    #     print('Stations: ', [station.id for station in data.generated_routes[i].stations])
    #     print('Arrival times: ', data.generated_routes[i].arrival_times)
    #     print('loading: ', data.generated_routes[i].loading)
    #     print('unloading: ', data.generated_routes[i].unloading)
    #     print('vehicle_level: ', data.generated_routes[i].vehicle_level)
    #     print('--------------------------------------------')

    # ------ VARIABLES ------------------------------------------------------------------------
    
    #input: tuple_list  
    l_lambda = m.addVars([(v,r) for v in data.V_VEHICLES for r in data.R_ROUTES[v]],vtype=GRB.BINARY,name="l_lambda") #name should be the same
    # OR m.addVars({(v,r) for v in data.V_VEHICLES for r in data.R_ROUTES[v]},vtype=GRB.BINARY,name="lambda")
    
    # ------ CONSTRAINTS -----------------------------------------------------------------------
    # Pick one route per vehicle
    m.addConstrs(quicksum(l_lambda[(v,r)] for r in data.R_ROUTES[v]) == 1 for v in data.V_VEHICLES)
    
    # Every station visited at most once
    m.addConstrs(quicksum(data.A_MATRIX[(i,v,r)] * l_lambda[(v,r)] for v in data.V_VEHICLES for r in data.R_ROUTES[v]) <= 1 for i in data.S_STATIONS)
    #THIS CONSTRAINT CAN CAUSE PROBLEMS. THE CURRENT BRANCHING ALGORITHM CONSTRUCTS ROUTES THAT ARE VERY SIMILAR, FOR EACH VEHICLE, THERE IS OVERLAP IN THE STATIONS THAT ARE BEING CHOSEN
    #SO, REMOVE THE STATION FROM POTENTIAL STATIONS FOR THE OTHER ROUTES!!!!!!
    
    

    # ------ OBJECTIVE -----------------------------------------------------------------------
    m.setObjective((data.W_OMEGA_V*(data.V_BASE - quicksum( data.V_PREVENTED[(v,r)]*l_lambda[(v,r)] for v in data.V_VEHICLES for r in data.R_ROUTES[v])) + 
                   data.W_OMEGA_D*(data.D_BASE - quicksum( data.D_PREVENTED[(v,r)]*l_lambda[(v,r)] for v in data.V_VEHICLES for r in data.R_ROUTES[v]))) , 
                   GRB.MINIMIZE)

    m.optimize()
    #infeasible model....
       
    #https://www.gurobi.com/documentation/9.5/refman/optimization_status_codes.html
    if m.status != 2:  #optimal
        print('model not optimal')
        if m.status == 3:
            print('model is infeasible')  #ADD A BREAKPOINT IN THIS IF STATEMENT FOR DEBUGGING. THEN INSPECT THE OBJECTS.
        else:
            print('status code: ', m.status)
    
    #obj_val = m.getObjective().getValue()    

    # for var in m.getVars():
    #     round_val = round(var.x, 0)
    #     print(var,': ', round_val)

    return m



    
    
    
    



    # TESTING
    #list2 = [0,1]
    #list1 = {0:[0,1], 1:[1,2]}
    #x = {}
    #for j in list2:
    #    for i in list1[j]:
    #        x[(i,j)] = 1

    #sum(x[(i,j)] for j in list2 for i in list1[j]  )    # DETTE ER RÆKKEFØLGEN SOM DEN MÅ VÆRE!!
