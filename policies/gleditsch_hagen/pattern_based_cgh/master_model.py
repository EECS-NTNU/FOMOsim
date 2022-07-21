from gurobipy import *
import time
import numpy as np


def run_master_model(data):

    
    m = Model("Master")
    m.setParam('OutputFlag', True)  #To do: should be at False.
    

    # ------ VARIABLES ------------------------------------------------------------------------
    
    #input: tuple_list  
    l_lambda = m.addVars([(v,r) for v in data.V_VEHICLES for r in data.R_ROUTES[v]],vtype=GRB.BINARY,name="l_lambda") #name should be the same
    # OR m.addVars({(v,r) for v in data.V_VEHICLES for r in data.R_ROUTES[v]},vtype=GRB.BINARY,name="lambda")
    
    # ------ CONSTRAINTS -----------------------------------------------------------------------
    # Pick one route per vehicle
    m.addConstrs(quicksum(l_lambda[(v,r)] for r in data.R_ROUTES[v]) == 1 for v in data.V_VEHICLES)
    
    # Every station visited at most once
    m.addConstrs(quicksum(data.A_MATRIX[(i,v,r)] * l_lambda[(v,r)] for v in data.V_VEHICLES for r in data.R_ROUTES[v]) <= 1 for i in data.S_STATIONS)
    

    # ------ OBJECTIVE -----------------------------------------------------------------------
    m.setObjective((data.W_OMEGA_V*(data.V_BASE - quicksum( data.V_PREVENTED[(v,r)]*l_lambda[(v,r)] for v in data.V_VEHICLES for r in data.R_ROUTES[v])) + 
                   data.W_OMEGA_D*(data.D_BASE - quicksum( data.D_PREVENTED[(v,r)]*l_lambda[(v,r)] for v in data.V_VEHICLES for r in data.R_ROUTES[v]))) , 
                   GRB.MINIMIZE)

    m.optimize()
    #infeasible model....
        
    obj_val = m.getObjective().getValue()

    return m




    # TESTING
    #list2 = [0,1]
    #list1 = {0:[0,1], 1:[1,2]}
    #x = {}
    #for j in list2:
    #    for i in list1[j]:
    #        x[(i,j)] = 1

    #sum(x[(i,j)] for j in list2 for i in list1[j]  )    # DETTE ER RÆKKEFØLGEN SOM DEN MÅ VÆRE!!
