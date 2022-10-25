from gurobipy import *

############################################################
#SIMPLE EXAMPLE
#define model
m=Model("test")

dic = dict()
dic["One"] = 1
dic["Two"] = 2

#create variables:
x=m.addVar({(i) for i in dic.keys()}, vtype=GRB.BINARY, name="x")
y=m.addVar(vtype=GRB.BINARY, name="y")
z=m.addVar(vtype=GRB.BINARY, name="z")

#updates the variables so they can be used in constraints
m.update()

#add constraints
m.addConstr(x+2*y+3*z<=4, "c0")
m.addConstr(x+y>=1, "c1")

#add objective
m.setObjective(x+y+2*z, GRB.MAXIMIZE)

#solve and print solution
m.optimize()
m.printAttr("X")
############################################################





