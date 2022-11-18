import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *

class Visualizer():
    def __init__(self, gurobi_output, parameters_data):
        self.gurobi_output = gurobi_output
        self.parameters_data = parameters_data

    def visualize_route(self):
        vehicle_colors= ['red', 'green', 'pink', "purple"]
        x_offset= 0.0025
        y_offset= 0.00125
        filename = self.parameters_data.state.mapdata[0]
        bBox = self.parameters_data.state.mapdata[1]
        image = plt.imread(filename)
        aspect_img = len(image[0]) / len(image)
        aspect_geo = (bBox[1]-bBox[0]) / (bBox[3]-bBox[2])
        aspect = aspect_geo / aspect_img
        fig, ax = plt.subplots()
        
        ax.set_title('Subproblem')
        ax.set_xlim(bBox[0],bBox[1])
        ax.set_ylim(bBox[2],bBox[3])
        ax.imshow(image, extent = bBox, aspect=aspect)
       
        loading_dict= dict() #stores the quantities loaded on different stations
        unloading_dict= dict() 
        for var in self.gurobi_output.getVars():
            if round(var.x,0) != 0:
                variable = var.varName.strip("]").split("[")
                name = variable[0]
                indices = variable[1].split(',')
                if name == 'x' and indices[0] != '-1':
                    vehicle_id = int(indices[3]) 
                    station = self.parameters_data.stations.get(int(indices[0])) #returns station object
                    lat = station.get_lat()
                    lon = station.get_lon()
                    #ax.plot(lon, lat, 'bo') #dot at station
                    ax.text(lon, lat, str(indices[0]), color="blue", bbox={'facecolor': 'none', 'edgecolor': 'blue', 'boxstyle':'round'}) #textbox with station_id
                    if indices[1]!= indices[0] and indices[1]!='-1': #end station
                        end_station = self.parameters_data.stations.get(int(indices[1]))
                        end_lat = end_station.get_lat()
                        end_lon = end_station.get_lon()
                        xx=[lon, end_lon]
                        yy=[lat, end_lat]
                        ax.plot(xx,yy,linewidth=1, color= vehicle_colors[vehicle_id])
                        driving_time= str(round(self.parameters_data.T_D.get((int(indices[0]),int(indices[1]))),1))
                        ax.text((lon+end_lon)/2, (lat+end_lat)/2, driving_time + " min", color=vehicle_colors[vehicle_id])

                if name == 'q_L':
                    load_station = self.parameters_data.stations.get(int(indices[0])) #returns station object 
                    if loading_dict.get(load_station) == None: #the station has no prior loading
                        loading_dict[load_station] = round(var.x,0)
                    else:
                        loading_dict[load_station] += round(var.x,0)



                if name == 'q_U':
                    unload_station = self.parameters_data.stations.get(int(indices[0]))
                    if unloading_dict.get(unload_station) == None: #the station has no prior loading
                                    unloading_dict[unload_station] = round(var.x,0)
                    else:
                        unloading_dict[unload_station] += round(var.x,0)
        
        for station in loading_dict:
            lat = station.get_lat()
            lon = station.get_lon()
            ax.text(lon+x_offset, lat, "Load: "+ str(loading_dict[station]))

        for station in unloading_dict:
            lat = station.get_lat()
            lon = station.get_lon()
            ax.text(lon+x_offset, lat, "Unload: "+ str(unloading_dict[station]))
            
        plt.show()

    #