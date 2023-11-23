import numpy as np
import matplotlib.pyplot as plt
import csv

class VisualizeResults():
    def __init__(self, simulator):
        self.simulator = simulator
    
    def visualize_violations_and_roaming(self, aggregated_data = None, show=False):
        if aggregated_data == None:
            data = {'Starvations':self.simulator.metrics.get_aggregate_value('starvation'),
                'Long Congestions':self.simulator.metrics.get_aggregate_value('long_congestion'),
                'Short Congestions':self.simulator.metrics.get_aggregate_value('short_congestion'),
                'Roaming for bikes':self.simulator.metrics.get_aggregate_value('roaming for bikes')}
        else:
            data = {'Starvations':aggregated_data['starvation'],
                'Long Congestions':aggregated_data['long_congestion'],
                'Short Congestions':aggregated_data['short_congestion'],
                'Roaming for bikes':aggregated_data['roaming for bikes']}
        type_of_violations = list(data.keys())
        values = list(data.values())
        colors = ['salmon', 'mediumpurple', 'cornflowerblue', 'green']
        fig, ax = plt.subplots()
       # ax.grid(b = True, axis = 'y', color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.4, zorder = 1)
        ax.grid(visible=True, axis='y', color='grey', linestyle='-.', linewidth=0.5, alpha=0.4, zorder=1)
        plt.bar(type_of_violations, values, width=0.4, color = colors, alpha = 1, zorder = 2)
        plt.ylabel("No. of events")
        ax.set_title('Different events', fontsize = 16, fontweight = 'bold')
        for i in range(len(values)):
            plt.annotate(str(values[i]), xy=(type_of_violations[i],values[i]), ha='center', va='bottom')
        if show:
            plt.show()
    
    def visualize_total_roaming_distances(self, aggregated_data = None, show=False):
        if aggregated_data == None:
            data = {'Roaming for locks':self.simulator.metrics.get_aggregate_value('roaming distance for locks'),
                'Roaming for bikes':self.simulator.metrics.get_aggregate_value('roaming distance for bikes')}
        else:
            data = {'Roaming for locks':aggregated_data['roaming distance for locks'],
                'Roaming for bikes':aggregated_data['roaming distance for bikes']}
        type_of_roaming = list(data.keys())
        values = list(data.values())
        colors = ['mediumpurple', 'cornflowerblue']
        fig, ax = plt.subplots()
        ax.grid(visible = True, axis = 'y', color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.4, zorder = 1)
        plt.bar(type_of_roaming, values, width=0.4, color = colors, alpha = 1, zorder = 2)
        plt.ylabel("Distance [km]")
        ax.set_title('Total roaming distance', fontsize = 16, fontweight = 'bold')
        ax.axes.set_xlim(-0.5,1.5)
        for i in range(len(values)):
            plt.annotate(str(round(values[i], 2)), xy=(type_of_roaming[i],values[i]), ha='center', va='bottom')
        if show:
            plt.show()

    def visualize_average_roaming_distances(self, aggregated_data = None, show=False):
        if aggregated_data == None:
            long_congestions = self.simulator.metrics.get_aggregate_value('long_congestion')
            short_congestions = self.simulator.metrics.get_aggregate_value('short_congestion')
            roaming_for_bikes = self.simulator.metrics.get_aggregate_value('roaming for bikes')
            roaming_distance_for_locks = self.simulator.metrics.get_aggregate_value('roaming distance for locks')
            roaming_distance_for_bikes = self.simulator.metrics.get_aggregate_value('roaming distance for bikes')
        else:
            long_congestions = aggregated_data['long_congestion']
            short_congestions = aggregated_data['short_congestion']
            roaming_for_bikes = aggregated_data['roaming for bikes']
            roaming_distance_for_locks = aggregated_data['roaming distance for locks']
            roaming_distance_for_bikes = aggregated_data['roaming distance for bikes']
        if long_congestions+short_congestions > 0:
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
        if show:
            plt.show()

    def visualize_share_of_events(self, aggregated_data = None, show=False):
        if aggregated_data == None:
            tot_events = self.simulator.metrics.get_aggregate_value('events')
            long_congestions = self.simulator.metrics.get_aggregate_value('long_congestion')
            starvations = self.simulator.metrics.get_aggregate_value('starvation')
            roaming_for_bikes = self.simulator.metrics.get_aggregate_value('roaming for bikes')
        else:
            tot_events = aggregated_data['events']
            long_congestions = aggregated_data['long_congestion']
            starvations = aggregated_data['starvation']
            roaming_for_bikes = aggregated_data['roaming for bikes']
        tot_successful = tot_events - long_congestions - starvations
        data = {'Successful events':tot_successful/tot_events, 'Long congestions':long_congestions/tot_events,
                'Starvations':starvations/tot_events}
        colors = ['seagreen', 'mediumpurple', 'salmon']
        explode = (0, 0.05, 0.05)
        labels = list(data.keys())
        values = list(data.values())
        fig, ax = plt.subplots()
        ax.set_title('Share of different events', fontsize = 16, fontweight = 'bold')
        ax.pie(values, labels=labels, explode=explode, autopct='%1.1f%%', colors=colors)
        ax.axis('equal')
        plt.savefig('./policies/hlm/simulation_results/test1.png')
        if show:
            plt.show()

def write_sim_results_to_file(filename, simulator, duration, append=False):
    header = ['Duration','Events','Starvations','Roaming for bikes',  'Long congestions', 'Short congestions', 'Roaming distance for bikes','Roaming distance for locks', 'Seed']
    data=[duration, simulator.metrics.get_aggregate_value('events'), simulator.metrics.get_aggregate_value('starvation'), 
          simulator.metrics.get_aggregate_value('roaming for bikes'), simulator.metrics.get_aggregate_value('long_congestion'), 
          simulator.metrics.get_aggregate_value('short_congestion'), round(simulator.metrics.get_aggregate_value('roaming distance for bikes'),2), 
          round(simulator.metrics.get_aggregate_value('roaming distance for locks'),2), simulator.state.seed]
    try:
        path= './policies/hlm/simulation_results/'+filename
        if append==False:
            with open(path,'w', newline='') as f:
                writer=csv.writer(f)
                writer.writerow(header)
                writer.writerow(data)
        else: 
            with open(path,'a',newline='') as f:
                writer=csv.writer(f)
                writer.writerow(data)
    except: 
        print("Error writing to CSV, write_sim_results_to_file")
        return None
    

def write_sol_time_to_file(filename, simulator):
    data=[simulator.metrics.get_aggregate_value('accumulated solution time'), simulator.metrics.get_aggregate_value('number of problems solved')]
    try:
        path= './policies/hlm/simulation_results/'+filename
        with open(path,'a',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(data)
    except: 
        print("Error writing to CSV, write_sol_time_to_file")
        return None

def write_sim_results_to_list(simulator, duration):
    data=[duration, simulator.metrics.get_aggregate_value('events'), simulator.metrics.get_aggregate_value('starvation'), 
          simulator.metrics.get_aggregate_value('roaming for bikes'), simulator.metrics.get_aggregate_value('long_congestion'), 
          simulator.metrics.get_aggregate_value('short_congestion'), round(simulator.metrics.get_aggregate_value('roaming distance for bikes'),2), 
          round(simulator.metrics.get_aggregate_value('roaming distance for locks'),2), simulator.state.seed]
    return data

def visualize_aggregated_violations_and_roaming(aggregated_data, filename):
    data = {'Starvations':aggregated_data['starvation'],
            'Long congestions':aggregated_data['long congestion'],
            'Short congestions':aggregated_data['short congestion'],
            'Roaming for bikes':aggregated_data['roaming for bikes']}
    type_of_event = list(data.keys())
    values = list(data.values())
    colors = ['salmon', 'mediumpurple', 'cornflowerblue', 'green']
    fig, ax = plt.subplots()
    ax.grid(visible = True, axis = 'y', color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.4, zorder = 1)
    plt.bar(type_of_event, values, width=0.4, color = colors, alpha = 1, zorder = 2)
    plt.ylabel("No. of events")
    ax.set_title('Different events', fontsize = 16, fontweight = 'bold')
    for i in range(len(values)):
        plt.annotate(str(values[i]), xy=(type_of_event[i],values[i]), ha='center', va='bottom')
    outfile=filename[:-4]+".png"
    plt.savefig('./policies/hlm/simulation_results/aggr_violations_and_roaming_'+outfile)
    
def visualize_aggregated_total_roaming_distances(aggregated_data, filename):
    data = {'Roaming for locks':aggregated_data['roaming distance for locks'],
            'Roaming for bikes':aggregated_data['roaming distance for bikes']}
    type_of_roaming = list(data.keys())
    values = list(data.values())
    colors = ['mediumpurple', 'cornflowerblue']
    fig, ax = plt.subplots()
    ax.grid(visible = True, axis = 'y', color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.4, zorder = 1)
    plt.bar(type_of_roaming, values, width=0.4, color = colors, alpha = 1, zorder = 2)
    plt.ylabel("Distance [km]")
    ax.set_title('Total roaming distance', fontsize = 16, fontweight = 'bold')
    ax.axes.set_xlim(-0.5,1.5)
    for i in range(len(values)):
        plt.annotate(str(round(values[i], 2)), xy=(type_of_roaming[i],values[i]), ha='center', va='bottom')
    outfile=filename[:-4]+".png"
    plt.savefig('./policies/hlm/simulation_results/aggr_total_roaming_distances_'+outfile)

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
    plt.savefig('./policies/hlm/simulation_results/aggr_average_roaming_distances_'+outfile)

def visualize_aggregated_share_of_events(aggregated_data, filename):
    tot_events = aggregated_data['events']
    long_congestions = aggregated_data['long congestion']
    starvations = aggregated_data['starvation']
    roaming_for_bikes = aggregated_data['roaming for bikes']
    tot_successful = tot_events - long_congestions - starvations
    data = {'Successful events':tot_successful/tot_events, 'Long congestions':long_congestions/tot_events,
            'Starvations':starvations/tot_events}
    colors = ['seagreen', 'mediumpurple', 'salmon']
    explode = (0, 0.05, 0.05)
    labels = list(data.keys())
    values = list(data.values())
    fig, ax = plt.subplots()
    ax.set_title('Share of different events', fontsize = 16, fontweight = 'bold')
    ax.pie(values, labels=labels, explode=explode, autopct='%1.1f%%', colors=colors)
    ax.axis('equal')
    outfile=filename[:-4]+".png"
    plt.savefig('./policies/hlm/simulation_results/aggr_share_of_events_'+outfile)


def visualize_aggregated_results(filename):
        try:
            path = './policies/hlm/simulation_results/'+filename
            with open(path, 'r', newline='') as f:
                reader = csv.reader(f, delimiter=',')
                aggregated_data = {'events':0, 'starvation':0, 'roaming for bikes':0,'long congestion':0, 'short congestion':0,  'roaming distance for bikes':0, 'roaming distance for locks':0} 
                for col in reader:
                    aggregated_data['events'] += int(col[1])
                    aggregated_data['starvation'] += int(col[2])
                    aggregated_data['roaming for bikes'] += int(col[3])
                    aggregated_data['long congestion'] += int(col[4])
                    aggregated_data['short congestion'] += int(col[5])
                    aggregated_data['roaming distance for bikes'] += float(col[6])
                    aggregated_data['roaming distance for locks'] += float(col[7])
            visualize_aggregated_violations_and_roaming(aggregated_data, filename)
            visualize_aggregated_total_roaming_distances(aggregated_data, filename)
            visualize_aggregated_average_roaming_distances(aggregated_data, filename)
            visualize_aggregated_share_of_events(aggregated_data, filename)
        
        except:
            print("Error with CSV, visualize_aggregated_results")
            return None

