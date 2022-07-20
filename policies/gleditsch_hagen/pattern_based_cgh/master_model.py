from gurobipy import *
import time
import numpy as np


def run_master_model(data):

    
    m = Model("Master")
    m.setParam('OutputFlag', False)


    # ------ VARIABLES ------------------------------------------------------------------------
    
    #input: tuple_list  
    l_lambda = m.addVars([(v,r) for v in data.V_VEHICLES for r in data.R_ROUTES[v]],vtype=GRB.BINARY,name="lambda")
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

    #obj_val = m.getObjective().getValue()

    return m

