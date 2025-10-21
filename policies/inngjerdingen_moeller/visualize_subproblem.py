import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *

class Visualizer():
    def __init__(self, gurobi_output, parameters_data):
        self.gurobi_output = gurobi_output
        self.parameters_data = parameters_data

    def visualize_route(self):
        vehicle_colors= ['red', 'lime', 'pink'] 
        time_colors=['maroon','darkgreen','deeppink']
        x_offset= 0.00175
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
        ax.imshow(image, extent = bBox, aspect=aspect, alpha=0.6)
       
        loading_dict= dict() #stores the quantities loaded on different stations
        unloading_dict= dict() 
        for var in self.gurobi_output.getVars():
            if round(var.x,2) != 0:
                variable = var.varName.strip("]").split("[")
                name = variable[0]
                indices = variable[1].split(',')
                if name == 'x' and indices[0] != '-1':
                    vehicle_id = int(indices[3]) 
                    station = self.parameters_data.stations.get(int(indices[0])) #returns station object
                    lat = station.get_lat()
                    lon = station.get_lon()
                    #ax.plot(lon, lat, 'bo') #dot at station
                    ax.text(lon, lat, str(indices[0]), color="blue", bbox={'facecolor': 'white', 'edgecolor': 'blue', 'boxstyle':'round'}) #textbox with station_id
                    if indices[1]!= indices[0] and indices[1]!='-1': #end station
                        end_station = self.parameters_data.stations.get(int(indices[1]))
                        end_lat = end_station.get_lat()
                        end_lon = end_station.get_lon()
                        xx=[lon, end_lon]
                        yy=[lat, end_lat]
                        ax.plot(xx,yy,linewidth=3, color= vehicle_colors[vehicle_id])
                        driving_time= str(round(self.parameters_data.T_D.get((int(indices[0]),int(indices[1]))),1))
                        ax.text((lon+end_lon)/2, (lat+end_lat)/2, driving_time + " min", color=time_colors[vehicle_id],weight='bold')

                if name == 'q_L':
                    load_station = self.parameters_data.stations.get(int(indices[0])) #returns station object 
                    if loading_dict.get(load_station) == None: #the station has no prior loading
                        loading_dict[load_station] = round(var.x,2)
                    else:
                        loading_dict[load_station] += round(var.x,2)



                if name == 'q_U':
                    unload_station = self.parameters_data.stations.get(int(indices[0]))
                    if unloading_dict.get(unload_station) == None: #the station has no prior loading
                                    unloading_dict[unload_station] = round(var.x,2)
                    else:
                        unloading_dict[unload_station] += round(var.x,2)
        
        for station in loading_dict:
            lat = station.get_lat()
            lon = station.get_lon()
            ax.text(lon+x_offset, lat, "Load: "+ str(loading_dict[station]),weight='bold')

        for station in unloading_dict:
            lat = station.get_lat()
            lon = station.get_lon()
            ax.text(lon+x_offset, lat, "Unload: "+ str(unloading_dict[station]),weight='bold')
            
        plt.show()

    def visualize_map_and_route(self):
        vehicle_colors = ['red', 'green', 'pink',]
        time_colors=['maroon','darkgreen','deeppink']
        x_offset = 0.0014
        y_offset = 0.00125
        filename = self.parameters_data.state.mapdata[0]
        bBox = self.parameters_data.state.mapdata[1]
        image = plt.imread(filename)
        aspect_img = len(image[0]) / len(image)
        aspect_geo = (bBox[1]-bBox[0]) / (bBox[3]-bBox[2])
        aspect = aspect_geo / aspect_img
        fig, ax = plt.subplots()
        
        ax.set_title('Subproblem and system state')
        ax.set_xlim(bBox[0],bBox[1])
        ax.set_ylim(bBox[2],bBox[3])
        ax.imshow(image, extent = bBox, aspect=aspect, alpha=0.6)
       
        loading_dict= dict() #stores the quantities loaded on different stations
        unloading_dict= dict() 
        plotted_stations = []
        for var in self.gurobi_output.getVars():
            if round(var.x,2) != 0:
                variable = var.varName.strip("]").split("[")
                name = variable[0]
                indices = variable[1].split(',')
                if name == 'x' and indices[0] != '-1':
                    vehicle_id = int(indices[3]) 
                    station = self.parameters_data.stations.get(int(indices[0])) #returns station object
                    plotted_stations.append(station)
                    lat = station.get_lat()
                    lon = station.get_lon()
                    #ax.plot(lon, lat, 'bo') #dot at station
                    ax.text(lon, lat, str(indices[0]), color="blue", bbox={'facecolor': 'white', 'edgecolor': 'blue', 'boxstyle':'round'}) #textbox with station_id
                    ax.text(lon - x_offset, lat, str(self.parameters_data.L_0[station.location_id]), size = 10, color="blue", bbox={'facecolor': 'white', 'edgecolor': 'blue', 'boxstyle':'circle'})
                    if indices[1]!= indices[0] and indices[1]!='-1': #end station
                        end_station = self.parameters_data.stations.get(int(indices[1]))
                        end_lat = end_station.get_lat()
                        end_lon = end_station.get_lon()
                        xx=[lon, end_lon]
                        yy=[lat, end_lat]
                        ax.plot(xx,yy,linewidth=3, color= vehicle_colors[vehicle_id])
                        driving_time= str(round(self.parameters_data.T_D.get((int(indices[0]),int(indices[1]))),1))
                        ax.text((lon+end_lon)/2-x_offset/4, (lat+end_lat)/2-y_offset/4, driving_time + " min", color=time_colors[vehicle_id],weight='bold')

                if name == 'q_L':
                    load_station = self.parameters_data.stations.get(int(indices[0])) #returns station object 
                    if loading_dict.get(load_station) == None: #the station has no prior loading
                        loading_dict[load_station] = round(var.x,2)
                    else:
                        loading_dict[load_station] += round(var.x,2)

                if name == 'q_U':
                    unload_station = self.parameters_data.stations.get(int(indices[0]))
                    if unloading_dict.get(unload_station) == None: #the station has no prior loading
                                    unloading_dict[unload_station] = round(var.x,2)
                    else:
                        unloading_dict[unload_station] += round(var.x,2)
        
        for station in self.parameters_data.stations.values():
            if station not in plotted_stations:
                lat = station.get_lat()
                lon = station.get_lon()
                if self.parameters_data.L_0[station.location_id] == 0:
                    ax.text(lon, lat, str(self.parameters_data.L_0[station.location_id]), size = 10, color="black", bbox={'facecolor': 'lightcoral', 'edgecolor': 'dimgray', 'boxstyle':'circle'})
                elif self.parameters_data.L_0[station.location_id] == self.parameters_data.Q_S[station.location_id]:
                    ax.text(lon, lat, str(self.parameters_data.L_0[station.location_id]), size = 10, color="black", bbox={'facecolor': 'yellow', 'edgecolor': 'dimgray', 'boxstyle':'circle'})
                else:
                    ax.text(lon, lat, str(self.parameters_data.L_0[station.location_id]), size = 10, color="black", bbox={'facecolor': 'silver', 'edgecolor': 'dimgray', 'boxstyle':'circle'})
            
        for station in loading_dict:
            lat = station.get_lat()
            lon = station.get_lon()
            ax.text(lon-x_offset/2, lat+y_offset/4, "Load: "+ str(loading_dict[station]),weight='bold')

        for station in unloading_dict:
            lat = station.get_lat()
            lon = station.get_lon()
            ax.text(lon-x_offset/2, lat+y_offset/4, "Unload: "+ str(unloading_dict[station]),weight='bold')

       
        plt.show()
    
    def visualize_stations(self):
        filename = self.parameters_data.state.mapdata[0]
        bBox = self.parameters_data.state.mapdata[1]
        image = plt.imread(filename)
        aspect_img = len(image[0]) / len(image)
        aspect_geo = (bBox[1]-bBox[0]) / (bBox[3]-bBox[2])
        aspect = aspect_geo / aspect_img
        fig, ax = plt.subplots()
        
        ax.set_title('Station IDs')
        ax.set_xlim(bBox[0],bBox[1])
        ax.set_ylim(bBox[2],bBox[3])
        ax.imshow(image, extent = bBox, aspect=aspect)
       
        for station in self.parameters_data.stations.values():
            lat = station.get_lat()
            lon = station.get_lon()
            ax.text(lon, lat, str(station.location_id), size = 6, color="black", bbox={'facecolor': 'lightskyblue', 'edgecolor': 'dimgray', 'boxstyle':'circle'})
            
        plt.show()


def visualize_stations_from_simulator(simul):
    filename = simul.state.mapdata[0]
    bBox = simul.state.mapdata[1]
    image = plt.imread(filename)
    aspect_img = len(image[0]) / len(image)
    aspect_geo = (bBox[1]-bBox[0]) / (bBox[3]-bBox[2])
    aspect = aspect_geo / aspect_img
    fig, ax = plt.subplots()
    
    ax.set_title('Station IDs')
    ax.set_xlim(bBox[0],bBox[1])
    ax.set_ylim(bBox[2],bBox[3])
    ax.imshow(image, extent = bBox, aspect=aspect)
    
    for station in simul.state.stations.values():
        lat = station.get_lat()
        lon = station.get_lon()
        if len(station.bikes) == 0:
            ax.text(lon, lat, str(0), size = 10, color="black", bbox={'facecolor': 'lightcoral', 'edgecolor': 'dimgray', 'boxstyle':'circle'})
        elif len(station.bikes) == station.capacity:
            ax.text(lon, lat, str(len(station.bikes)), size = 10, color="black", bbox={'facecolor': 'yellow', 'edgecolor': 'dimgray', 'boxstyle':'circle'})
        else:
            ax.text(lon, lat, str(len(station.bikes)), size = 10, color="black", bbox={'facecolor': 'silver', 'edgecolor': 'dimgray', 'boxstyle':'circle'})

        
    plt.show()
