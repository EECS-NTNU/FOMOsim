import numpy as np
import matplotlib.pyplot as plt
import sim
from parameters_MILP import MILP_data
from gurobipy import *

class Visualizer():
    def __init__(self, gurobi_output, parameters_data):
        self.gurobi_output = gurobi_output
        self.parameters_data = parameters_data

    def visualize_subproblem(self):
        filename = self.parameters_data.simul[0].state.mapdata[0]
        bBox = self.parameters_data.simul[0].state.mapdata[1]
        image = plt.imread(filename)
        aspect_img = len(image[0]) / len(image)
        aspect_geo = (bBox[1]-bBox[0]) / (bBox[3]-bBox[2])
        aspect = aspect_geo / aspect_img

        cm = plt.cm.get_cmap('RdYlBu_r')

        fig, ax = plt.subplots()
        #im = ax.scatter(xx, yy, c=cc, s=30, edgecolors="black", cmap=cm)
        ax.set_title('Subproblem')
        ax.set_xlim(bBox[0],bBox[1])
        ax.set_ylim(bBox[2],bBox[3])
        ax.imshow(image, extent = bBox, aspect=aspect)

        fig.colorbar(im, ax=ax)

        

        for var in self.gurobi_output.getVars():
            if round(var.x,0) !=0:
                variable = var.varName.strip("]").split("[")
                name = variable[0]
                indices = variable[1].split(',')
       
        plt.show()

   