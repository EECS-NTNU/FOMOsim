import numpy as np
import matplotlib.pyplot as plt
import csv
from settings import *
import os
import datetime
from sim import Action, EBike, State, Vehicle, Metric
import matplotlib.pyplot as plt
import copy
import matplotlib.dates as mdates
from matplotlib import gridspec

def write_sim_results_to_file(filename, simulator_metrics, seed, duration, policy_name, append=False):
    header = ['Duration',
              'Events',
              'Trips',
              'Random trips',
              'Bike Departures',
              'Escooter Departures',
              'Bike Arrivals',
              'Escooter Arrivals',
              'Vehicle Arrivals',
              'Failed Events',
              'Starvations',
              'Escooter Starvations',
              'Bike Starvations',
              'Battery Starvations',
              'Battery Violations',
              'Long Congestions',
              'Short Congestions',
              'Roaming for bikes',
              'Roaming distance for bikes',
              'Roaming for escooters',
              'Roaming distance for escooters',
              'Roaming for locks',
              'Roaming distance for locks',
              'Number of bikes delivered',
              'Number of bikes picked up',
              'Number of battery swaps',
              'Number of esooters delivered',
              'Number of escooters picked up',
              'Number of helping bikes delivered',
              'Number of helping bikes picked up',
              'Number of helping esooters delivered',
              'Number of helping escooters picked up',
              'Seed']
    
    data=[duration, 
          simulator_metrics.get_aggregate_value('events'), 
          simulator_metrics.get_aggregate_value('trips'), 
          simulator_metrics.get_aggregate_value('random trips'), 
          simulator_metrics.get_aggregate_value('bike departure'), 
          simulator_metrics.get_aggregate_value('escooter departure'),
          simulator_metrics.get_aggregate_value('bike arrival'),
          simulator_metrics.get_aggregate_value('escooter arrival'),
          simulator_metrics.get_aggregate_value('vehicle arrivals'),

          simulator_metrics.get_aggregate_value('failed events'), 
          simulator_metrics.get_aggregate_value('starvations'),
          simulator_metrics.get_aggregate_value('escooter starvations'),
          simulator_metrics.get_aggregate_value('bike starvations'),
          simulator_metrics.get_aggregate_value('battery starvations'), 
          simulator_metrics.get_aggregate_value('battery violations'), 
          simulator_metrics.get_aggregate_value('long congestions'),
          simulator_metrics.get_aggregate_value('short congestions'), 

          simulator_metrics.get_aggregate_value('roaming for bikes'), 
          round(simulator_metrics.get_aggregate_value('roaming distance for bikes'), 2),
          simulator_metrics.get_aggregate_value('roaming for escooters'), 
          round(simulator_metrics.get_aggregate_value('roaming distance for escooters'), 2),
          simulator_metrics.get_aggregate_value('roaming for locks'), 
          round(simulator_metrics.get_aggregate_value('roaming distance for locks'), 2),

          simulator_metrics.get_aggregate_value('num bike deliveries'),
          simulator_metrics.get_aggregate_value('num bike pickups'),
          simulator_metrics.get_aggregate_value('num battery swaps'),
          simulator_metrics.get_aggregate_value('num escooter deliveries'),
          simulator_metrics.get_aggregate_value('num escooter pickups'),
          simulator_metrics.get_aggregate_value('num helping bike deliveries'),
          simulator_metrics.get_aggregate_value('num helping bike pickups'),
          simulator_metrics.get_aggregate_value('num helping escooter deliveries'),
          simulator_metrics.get_aggregate_value('num helping escooter pickups'),

          seed]
    
    try:
        folder_path= f'./policies/hlv_master/results/{policy_name}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        path= folder_path + '/' + filename

        # If file does not exist, add header
        if append==False:
            with open(path,'w', newline='') as f:
                writer=csv.writer(f)
                writer.writerow(header)
                writer.writerow(data)
        
        # If file exists, do not add header
        else: 
            with open(path,'a',newline='') as f:
                writer=csv.writer(f)
                writer.writerow(data)
    except: 
        print("Error writing to CSV, write_sim_results_to_file")
        return None

def write_parameters_to_file(filename, policies, policy_name, num_vehicles, duration):
    data = {}
    for policy, is_SB in policies:
        data[is_SB] = {
            'max_depth': policy.max_depth, 
            'number_of_successors': policy.number_of_successors, 
            'time_horizon': policy.time_horizon, 
            'criticality_weights_sets': policy.criticality_weights_set, 
            'evaluation_weights': policy.evaluation_weights, 
            'number_of_scenarios': policy.number_of_scenarios, 
            'discounting_factor': policy.discounting_factor,
            'num_vehicles': num_vehicles,
            'duration': duration//(24*60)
        }
    try:
        path= f'./policies/hlv_master/results/{policy_name}/' + filename
        with open(path,'a',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(list(data.values()))
    except: 
        print("Error writing to CSV, write_parameters_to_file")
        return None

def write_sol_time_to_file(filename, simulator_metrics, policy_name, append = False):
    header = ['Accumulated time to find action',
              'Accumulated time to find location',
              'Accumulated action time',
              'Accumulated driving time',
              'Accumulated solution time',
              'Accumulated time to find helping actions',
              'Number of get_best_action']
    data=[simulator_metrics.get_aggregate_value('accumulated find action time'),
          simulator_metrics.get_aggregate_value('accumulated find location time'),
          simulator_metrics.get_aggregate_value('accumulated action time'),
          simulator_metrics.get_aggregate_value('accumulated driving time'),
          simulator_metrics.get_aggregate_value('accumulated sol time'),
          simulator_metrics.get_aggregate_value('accumulated find helping action time'),
          simulator_metrics.get_aggregate_value('get_best_action')]
    try:
        path= f'./policies/hlv_master/results/{policy_name}/' + filename

        # If file does not exist, add header
        if append==False:
            with open(path,'w', newline='') as f:
                writer=csv.writer(f)
                writer.writerow(header)
                writer.writerow(data)
        
        # If file exists, do not add header
        else: 
            with open(path,'a',newline='') as f:
                writer=csv.writer(f)
                writer.writerow(data)
    except: 
        print("Error writing to CSV, write_sol_time_to_file")
        return None

def visualize_aggregated_violations_and_roaming(aggregated_data, filename):
    data = {'Starvations': aggregated_data['starvation'],
            'No scooters': aggregated_data['starvations, no bikes'],
            'No battery': aggregated_data['starvations, no battery'],
            'Battery Violation': aggregated_data['battery violation'],
            'Roaming': aggregated_data['roaming for bikes']}
    type_of_event = list(data.keys())
    values = list(data.values())
    colors = ['salmon', 'mediumpurple', 'cornflowerblue', 'red', 'green']
    fig, ax = plt.subplots()
    ax.grid(visible = True, axis = 'y', color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.4, zorder = 1)
    plt.bar(type_of_event, values, width=0.4, color = colors, alpha = 1, zorder = 2)
    plt.ylabel("No. of events")
    ax.set_title('Different events', fontsize = 16, fontweight = 'bold')
    for i in range(len(values)):
        plt.annotate(str(values[i]), xy=(type_of_event[i],values[i]), ha='center', va='bottom')
    outfile=filename[:-4]+".png"
    plt.savefig('./policies/hlv_master/results/aggr_violations_and_roaming_'+outfile)
    
def visualize_aggregated_total_roaming_distances(aggregated_data, filename):
    data = {# 'Roaming for locks':aggregated_data['roaming distance for locks'],
            'Roaming for bikes':aggregated_data['roaming distance for bikes']}
    type_of_roaming = list(data.keys())
    values = list(data.values())
    colors = [#'mediumpurple', 
        'cornflowerblue']
    fig, ax = plt.subplots()
    ax.grid(visible = True, axis = 'y', color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.4, zorder = 1)
    plt.bar(type_of_roaming, values, width=0.4, color = colors, alpha = 1, zorder = 2)
    plt.ylabel("Distance [km]")
    ax.set_title('Total roaming distance', fontsize = 16, fontweight = 'bold')
    ax.axes.set_xlim(-0.5,1.5)
    for i in range(len(values)):
        plt.annotate(str(round(values[i], 2)), xy=(type_of_roaming[i],values[i]), ha='center', va='bottom')
    outfile=filename[:-4]+".png"
    plt.savefig('./policies/hlv_master/results/aggr_total_roaming_distances_'+outfile)

def visualize_aggregated_average_roaming_distances(aggregated_data, filename):
    long_congestions = aggregated_data['long congestion']
    short_congestions = aggregated_data['short congestion']
    roaming_for_bikes = aggregated_data['roaming for bikes']
    roaming_distance_for_locks = aggregated_data['roaming distance for locks']
    roaming_distance_for_bikes = aggregated_data['roaming distance for bikes']
    if (long_congestions+short_congestions) > 0:
        avg_roaming_for_locks = roaming_distance_for_locks/(long_congestions+short_congestions)
    else:
        avg_roaming_for_locks = 0
    if roaming_for_bikes > 0:
        avg_roaming_for_bikes = roaming_distance_for_bikes/roaming_for_bikes
    else:
        avg_roaming_for_bikes = 0
    data = {'Roaming for locks':avg_roaming_for_locks,
            'Roaming for bikes':avg_roaming_for_bikes}
    type_of_roaming = list(data.keys())
    values = list(data.values())
    colors = ['mediumpurple', 'cornflowerblue']
    fig, ax = plt.subplots()
    ax.grid(visible = True, axis = 'y', color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.4, zorder = 1)
    plt.bar(type_of_roaming, values, width=0.4, color = colors, alpha = 1, zorder = 2)
    plt.ylabel("Avg. distance [km]")
    ax.set_title('Average roaming distance', fontsize = 16, fontweight = 'bold')
    ax.axes.set_xlim(-0.5,1.5)
    for i in range(len(values)):
        plt.annotate(str(round(values[i], 2)), xy=(type_of_roaming[i],values[i]), ha='center', va='bottom')
    outfile=filename[:-4]+".png"
    plt.savefig('./policies/hlv_master/results/aggr_average_roaming_distances_'+outfile)

def visualize_aggregated_share_of_events(aggregated_data, filename):
    tot_events = aggregated_data['events']
    bike_starvations = aggregated_data['starvations, no bikes']
    battery_starvations = aggregated_data['starvations, no battery']
    battery_violation = aggregated_data['battery violation']
    tot_successful = tot_events - bike_starvations - battery_starvations - battery_violation
    data = {'Successful events':tot_successful/tot_events,
            'Bike Starvations':bike_starvations/tot_events,
            'Battery Starvations':battery_starvations/tot_events,
            'Battery Violation': battery_violation/tot_events}
    colors = ['seagreen', 'mediumpurple', 'salmon', 'red']
    explode = (0, 0.05, 0.05, 0.05)
    labels = list(data.keys())
    values = list(data.values())
    fig, ax = plt.subplots()
    ax.set_title('Share of different events', fontsize = 16, fontweight = 'bold')
    ax.pie(values, labels=labels, explode=explode, autopct='%1.1f%%', colors=colors)
    ax.axis('equal')
    outfile=filename[:-4]+".png"
    plt.savefig('./policies/hlv_master/results/aggr_share_of_events_'+outfile)


def visualize_aggregated_results(filename, policy_name):
        try:
            path = f'./policies/hlv_master/results/{policy_name}/' + filename
            with open(path, 'r', newline='') as f:
                reader = csv.reader(f, delimiter=',')
                aggregated_data = {'events':0, 
                                   'starvations':0, 
                                   'bike starvations':0,
                                   'escooter starvations':0,
                                   'battery starvations':0,
                                   'battery violations': 0, 
                                   'long congestions':0,
                                   'short congestions': 0,
                                   'roaming distance for bikes':0} 
                for col in reader:
                    aggregated_data['events'] += int(col[1])
                    aggregated_data['starvation'] += int(col[2])
                    aggregated_data['starvations, no bikes'] += int(col[3])
                    aggregated_data['starvations, no battery'] += int(col[4])
                    aggregated_data['battery violation'] += int(col[5])
                    aggregated_data['roaming for bikes'] += float(col[6])
                    aggregated_data['roaming distance for bikes'] += float(col[7])

            visualize_aggregated_violations_and_roaming(aggregated_data, filename) # 
            visualize_aggregated_total_roaming_distances(aggregated_data, filename)
            visualize_aggregated_average_roaming_distances(aggregated_data, filename)
            visualize_aggregated_share_of_events(aggregated_data, filename)
        
        except:
            print("Error with CSV, visualize_aggregated_results")
            return None


def totime(ts):
    weektext = "2022 " + str(1) + " 1 00:00"
    startdate=datetime.datetime.strptime(weektext, "%Y %W %w %H:%M")
    return datetime.datetime.fromtimestamp(startdate.timestamp() + ts * 60)

def visualize(filename, policy_name, metrics, metric_name="starvations"):
    fig, ax = plt.subplots()
    ax.set_xlabel("Time", labelpad=10, fontsize=12)
    ax.set_ylabel("Number", labelpad=10, fontsize=12)
    ax.set_title(metric_name, fontsize=14)

    if type(metrics) is list:
        metricObj = Metric.merge_metrics(metrics)
    else:
        metricObj = metrics

    xx = []
    yy = []

    # add start point
    if metricObj.isAggregate(metric_name):
        xx.append(totime(metricObj.min_time))
        yy.append(0)

    # add main points
    if metric_name in metricObj.metrics:
        xx.extend([totime(item[0]) for item in metricObj.metrics[metric_name]])
        yy.extend([item[1] for item in metricObj.metrics[metric_name]])

    # add end point
    if metricObj.isAggregate(metric_name):
        xx.append(totime(metricObj.max_time))
        yy.append(yy[-1])

    data_pairs = zip(xx,yy)

    try:
        path= f'./policies/hlv_master/results/{policy_name}/cumulated_'+ metric_name +'_' + filename
        with open(path,'a',newline='') as f:
            writer=csv.writer(f)
            for pair in data_pairs:
                writer.writerow(pair)
    except: 
        print("Error writing to CSV, write_cumulated_results")

    ax.plot(xx, yy)

    ax.set_ylim(ymin=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%a %H:%M"))

    outfile=filename[:-4]+".png"
    plt.savefig(f'./policies/hlv_master/results/{policy_name}/' + 'cumulated_'+ metric_name +'_'+outfile)
    plt.close(fig)