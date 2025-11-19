import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *

class Visualizer():
    def __init__(self, gurobi_output, parameters_data):
        self.gurobi_output = gurobi_output
        self.parameters_data = parameters_data

    def visualize_route(self):
        vehicle_colors = ['red', 'lime', 'pink', 'orange', 'cyan'] 
        time_colors = ['maroon', 'darkgreen', 'deeppink', 'darkorange', 'darkblue']
        x_offset = 0.00175
        y_offset = 0.00125
        filename = self.parameters_data.state.mapdata[0]
        bBox = self.parameters_data.state.mapdata[1]
        image = plt.imread(filename)
        aspect_img = len(image[0]) / len(image)
        aspect_geo = (bBox[1] - bBox[0]) / (bBox[3] - bBox[2])
        aspect = aspect_geo / aspect_img
        fig, ax = plt.subplots()
        
        ax.set_title('Subproblem Solution')
        ax.set_xlim(bBox[0], bBox[1])
        ax.set_ylim(bBox[2], bBox[3])
        ax.imshow(image, extent=bBox, aspect=aspect, alpha=0.6)
       
        loading_dict = dict()  # stores the quantities loaded on different stations
        unloading_dict = dict() 
        
        for var in self.gurobi_output.getVars():
            if round(var.x, 2) != 0:
                variable = var.varName.strip("]").split("[")
                name = variable[0]
                indices = variable[1].split(',')
                
                # x[i,j,v,t] where i=from, j=to, v=vehicle, t=period
                if name == 'x' and indices[0] != '-1':
                    from_idx = int(indices[0])
                    to_idx = int(indices[1])
                    vehicle_idx = int(indices[2])
                    period = int(indices[3])
                    
                    # Map index back to station ID
                    from_station_id = self.parameters_data.index_to_station_id[from_idx]
                    station = self.parameters_data.state.stations[from_station_id]
                    lat = station.get_lat()
                    lon = station.get_lon()
                    
                    # Draw station ID
                    ax.text(lon, lat, str(from_station_id), color="blue", 
                           bbox={'facecolor': 'white', 'edgecolor': 'blue', 'boxstyle':'round'})
                    
                    if to_idx != from_idx and to_idx >= 0:  # Not holding at same station, not depot
                        to_station_id = self.parameters_data.index_to_station_id[to_idx]
                        end_station = self.parameters_data.state.stations[to_station_id]
                        end_lat = end_station.get_lat()
                        end_lon = end_station.get_lon()
                        xx = [lon, end_lon]
                        yy = [lat, end_lat]
                        ax.plot(xx, yy, linewidth=3, color=vehicle_colors[vehicle_idx % len(vehicle_colors)])
                        
                        # Show driving time
                        driving_time = str(round(self.parameters_data.T_D.get((from_idx, to_idx), 0), 1))
                        ax.text((lon + end_lon)/2, (lat + end_lat)/2, driving_time + " min", 
                               color=time_colors[vehicle_idx % len(time_colors)], weight='bold')

                # qL[i,v,t] where i=station, v=vehicle, t=period
                if name == 'qL':
                    station_idx = int(indices[0])
                    station_id = self.parameters_data.index_to_station_id[station_idx]
                    load_station = self.parameters_data.state.stations[station_id]
                    
                    if loading_dict.get(load_station) == None:
                        loading_dict[load_station] = round(var.x, 2)
                    else:
                        loading_dict[load_station] += round(var.x, 2)

                # qU[i,v,t] where i=station, v=vehicle, t=period
                if name == 'qU':
                    station_idx = int(indices[0])
                    station_id = self.parameters_data.index_to_station_id[station_idx]
                    unload_station = self.parameters_data.state.stations[station_id]
                    
                    if unloading_dict.get(unload_station) == None:
                        unloading_dict[unload_station] = round(var.x, 2)
                    else:
                        unloading_dict[unload_station] += round(var.x, 2)
        
        for station in loading_dict:
            lat = station.get_lat()
            lon = station.get_lon()
            ax.text(lon + x_offset, lat, "Load: " + str(loading_dict[station]), weight='bold')

        for station in unloading_dict:
            lat = station.get_lat()
            lon = station.get_lon()
            ax.text(lon + x_offset, lat, "Unload: " + str(unloading_dict[station]), weight='bold')
            
        plt.show()

    def visualize_map_and_route(self):
        vehicle_colors = ['red', 'green', 'pink', 'orange', 'cyan']
        time_colors = ['maroon', 'darkgreen', 'deeppink', 'darkorange', 'darkblue']
        x_offset = 0.0014
        y_offset = 0.00125
        filename = self.parameters_data.state.mapdata[0]
        bBox = self.parameters_data.state.mapdata[1]
        image = plt.imread(filename)
        aspect_img = len(image[0]) / len(image)
        aspect_geo = (bBox[1] - bBox[0]) / (bBox[3] - bBox[2])
        aspect = aspect_geo / aspect_img
        fig, ax = plt.subplots()
        
        ax.set_title('Subproblem and System State')
        ax.set_xlim(bBox[0], bBox[1])
        ax.set_ylim(bBox[2], bBox[3])
        ax.imshow(image, extent=bBox, aspect=aspect, alpha=0.6)
       
        loading_dict = dict()
        unloading_dict = dict() 
        plotted_stations = []
        
        for var in self.gurobi_output.getVars():
            if round(var.x, 2) != 0:
                variable = var.varName.strip("]").split("[")
                name = variable[0]
                indices = variable[1].split(',')
                
                if name == 'x' and indices[0] != '-1':
                    from_idx = int(indices[0])
                    to_idx = int(indices[1])
                    vehicle_idx = int(indices[2])
                    
                    from_station_id = self.parameters_data.index_to_station_id[from_idx]
                    station = self.parameters_data.state.stations[from_station_id]
                    plotted_stations.append(station)
                    lat = station.get_lat()
                    lon = station.get_lon()
                    
                    # Show station ID
                    ax.text(lon, lat, str(from_station_id), color="blue", 
                           bbox={'facecolor': 'white', 'edgecolor': 'blue', 'boxstyle':'round'})
                    
                    # Show current bike inventory at station
                    station_inventory = self.parameters_data.I_N0[from_idx]
                    ax.text(lon - x_offset, lat, str(station_inventory), size=10, color="blue", 
                           bbox={'facecolor': 'white', 'edgecolor': 'blue', 'boxstyle':'circle'})
                    
                    if to_idx != from_idx and to_idx >= 0:
                        to_station_id = self.parameters_data.index_to_station_id[to_idx]
                        end_station = self.parameters_data.state.stations[to_station_id]
                        end_lat = end_station.get_lat()
                        end_lon = end_station.get_lon()
                        xx = [lon, end_lon]
                        yy = [lat, end_lat]
                        ax.plot(xx, yy, linewidth=3, color=vehicle_colors[vehicle_idx % len(vehicle_colors)])
                        driving_time = str(round(self.parameters_data.T_D.get((from_idx, to_idx), 0), 1))
                        ax.text((lon + end_lon)/2 - x_offset/4, (lat + end_lat)/2 - y_offset/4, 
                               driving_time + " min", color=time_colors[vehicle_idx % len(time_colors)], 
                               weight='bold')

                if name == 'qL':
                    station_idx = int(indices[0])
                    station_id = self.parameters_data.index_to_station_id[station_idx]
                    load_station = self.parameters_data.state.stations[station_id]
                    
                    if loading_dict.get(load_station) == None:
                        loading_dict[load_station] = round(var.x, 2)
                    else:
                        loading_dict[load_station] += round(var.x, 2)

                if name == 'qU':
                    station_idx = int(indices[0])
                    station_id = self.parameters_data.index_to_station_id[station_idx]
                    unload_station = self.parameters_data.state.stations[station_id]
                    
                    if unloading_dict.get(unload_station) == None:
                        unloading_dict[unload_station] = round(var.x, 2)
                    else:
                        unloading_dict[unload_station] += round(var.x, 2)
        
        # Show all other stations not in the route
        for station_id, station in self.parameters_data.state.stations.items():
            if station not in plotted_stations:
                lat = station.get_lat()
                lon = station.get_lon()
                station_idx = self.parameters_data.station_id_to_index[station_id]
                num_bikes = self.parameters_data.I_N0[station_idx]
                capacity = self.parameters_data.Q_S[station_idx]
                
                # Color code based on inventory
                if num_bikes == 0:
                    ax.text(lon, lat, str(num_bikes), size=10, color="black", 
                           bbox={'facecolor': 'lightcoral', 'edgecolor': 'dimgray', 'boxstyle':'circle'})
                elif num_bikes >= capacity:
                    ax.text(lon, lat, str(num_bikes), size=10, color="black", 
                           bbox={'facecolor': 'yellow', 'edgecolor': 'dimgray', 'boxstyle':'circle'})
                else:
                    ax.text(lon, lat, str(num_bikes), size=10, color="black", 
                           bbox={'facecolor': 'silver', 'edgecolor': 'dimgray', 'boxstyle':'circle'})
            
        for station in loading_dict:
            lat = station.get_lat()
            lon = station.get_lon()
            ax.text(lon - x_offset/2, lat + y_offset/4, "Load: " + str(loading_dict[station]), weight='bold')

        for station in unloading_dict:
            lat = station.get_lat()
            lon = station.get_lon()
            ax.text(lon - x_offset/2, lat + y_offset/4, "Unload: " + str(unloading_dict[station]), weight='bold')

        plt.show()
    
    def visualize_stations(self):
        filename = self.parameters_data.state.mapdata[0]
        bBox = self.parameters_data.state.mapdata[1]
        image = plt.imread(filename)
        aspect_img = len(image[0]) / len(image)
        aspect_geo = (bBox[1] - bBox[0]) / (bBox[3] - bBox[2])
        aspect = aspect_geo / aspect_img
        fig, ax = plt.subplots()
        
        ax.set_title('Station IDs')
        ax.set_xlim(bBox[0], bBox[1])
        ax.set_ylim(bBox[2], bBox[3])
        ax.imshow(image, extent=bBox, aspect=aspect)
       
        for station_id, station in self.parameters_data.state.stations.items():
            lat = station.get_lat()
            lon = station.get_lon()
            ax.text(lon, lat, str(station_id), size=6, color="black", 
                   bbox={'facecolor': 'lightskyblue', 'edgecolor': 'dimgray', 'boxstyle':'circle'})
            
        plt.show()


def visualize_stations_from_simulator(simul):
    """Standalone function to visualize current state from simulator."""
    filename = simul.state.mapdata[0]
    bBox = simul.state.mapdata[1]
    image = plt.imread(filename)
    aspect_img = len(image[0]) / len(image)
    aspect_geo = (bBox[1] - bBox[0]) / (bBox[3] - bBox[2])
    aspect = aspect_geo / aspect_img
    fig, ax = plt.subplots()
    
    ax.set_title('Current System State')
    ax.set_xlim(bBox[0], bBox[1])
    ax.set_ylim(bBox[2], bBox[3])
    ax.imshow(image, extent=bBox, aspect=aspect)
    
    for station in simul.state.stations.values():
        lat = station.get_lat()
        lon = station.get_lon()
        num_bikes = len(station.bikes)
        
        if num_bikes == 0:
            ax.text(lon, lat, str(0), size=10, color="black", 
                   bbox={'facecolor': 'lightcoral', 'edgecolor': 'dimgray', 'boxstyle':'circle'})
        elif num_bikes >= station.capacity:
            ax.text(lon, lat, str(num_bikes), size=10, color="black", 
                   bbox={'facecolor': 'yellow', 'edgecolor': 'dimgray', 'boxstyle':'circle'})
        else:
            ax.text(lon, lat, str(num_bikes), size=10, color="black", 
                   bbox={'facecolor': 'silver', 'edgecolor': 'dimgray', 'boxstyle':'circle'})
        
    plt.show()
