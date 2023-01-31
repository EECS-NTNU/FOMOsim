import numpy as np
import matplotlib.pyplot as plt
import csv

class VisualizeResults():
    def __init__(self, simulator):
        self.simulator = simulator
    
    def visualize_violations_and_roaming(self, aggregated_data = None):
        if aggregated_data == None:
            data = {'Starvations':self.simulator.metrics.get_aggregate_value('starvation'),
                'Congestions':self.simulator.metrics.get_aggregate_value('congestion'),
                'Roaming for bikes':self.simulator.metrics.get_aggregate_value('roaming for bikes')}
        else:
            data = {'Starvations':aggregated_data['starvation'],
                'Congestions':aggregated_data['congestion'],
                'Roaming for bikes':aggregated_data['roaming for bikes']}
        type_of_violations = list(data.keys())
        values = list(data.values())
        colors = ['salmon', 'mediumpurple', 'cornflowerblue']
        fig, ax = plt.subplots()
        ax.grid(b = True, axis = 'y', color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.4, zorder = 1)
        plt.bar(type_of_violations, values, width=0.4, color = colors, alpha = 1, zorder = 2)
        plt.ylabel("No. of violations")
        ax.set_title('Number of violations', fontsize = 16, fontweight = 'bold')
        for i in range(len(values)):
            plt.annotate(str(values[i]), xy=(type_of_violations[i],values[i]), ha='center', va='bottom')
        plt.show()
    
    def visualize_total_roaming_distances(self, aggregated_data = None):
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
        ax.grid(b = True, axis = 'y', color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.4, zorder = 1)
        plt.bar(type_of_roaming, values, width=0.4, color = colors, alpha = 1, zorder = 2)
        plt.ylabel("Distance [km]")
        ax.set_title('Total roaming distance', fontsize = 16, fontweight = 'bold')
        ax.axes.set_xlim(-0.5,1.5)
        for i in range(len(values)):
            plt.annotate(str(round(values[i], 2)), xy=(type_of_roaming[i],values[i]), ha='center', va='bottom')
        plt.show()

    def visualize_average_roaming_distances(self, aggregated_data = None):
        if aggregated_data == None:
            congestions = self.simulator.metrics.get_aggregate_value('congestion')
            roaming_for_bikes = self.simulator.metrics.get_aggregate_value('roaming for bikes')
            roaming_distance_for_locks = self.simulator.metrics.get_aggregate_value('roaming distance for locks')
            roaming_distance_for_bikes = self.simulator.metrics.get_aggregate_value('roaming distance for bikes')
        else:
            congestions = aggregated_data['congestion']
            roaming_for_bikes = aggregated_data['roaming for bikes']
            roaming_distance_for_locks = aggregated_data['roaming distance for locks']
            roaming_distance_for_bikes = aggregated_data['roaming distance for bikes']
        if congestions > 0:
            avg_roaming_for_locks = roaming_distance_for_locks/congestions
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
        ax.grid(b = True, axis = 'y', color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.4, zorder = 1)
        plt.bar(type_of_roaming, values, width=0.4, color = colors, alpha = 1, zorder = 2)
        plt.ylabel("Avg. distance [km]")
        ax.set_title('Average roaming distance', fontsize = 16, fontweight = 'bold')
        ax.axes.set_xlim(-0.5,1.5)
        for i in range(len(values)):
            plt.annotate(str(round(values[i], 2)), xy=(type_of_roaming[i],values[i]), ha='center', va='bottom')
        plt.show()

    def visualize_share_of_events(self, aggregated_data = None):
        if aggregated_data == None:
            tot_events = self.simulator.metrics.get_aggregate_value('trips')
            congestions = self.simulator.metrics.get_aggregate_value('congestion')
            starvations = self.simulator.metrics.get_aggregate_value('starvation')
            roaming_for_bikes = self.simulator.metrics.get_aggregate_value('roaming for bikes')
        else:
            tot_events = aggregated_data['trips']
            congestions = aggregated_data['congestion']
            starvations = aggregated_data['starvation']
            roaming_for_bikes = aggregated_data['roaming for bikes']
        tot_successfull = tot_events - congestions - starvations - roaming_for_bikes
        data = {'Successfull trips':tot_successfull/tot_events, 'Congestions':congestions/tot_events,
                'Starvations':starvations/tot_events, 'Roaming for bikes':roaming_for_bikes/tot_events}
        colors = ['seagreen', 'mediumpurple', 'salmon', 'cornflowerblue']
        explode = (0, 0.05, 0.05, 0.05)
        labels = list(data.keys())
        values = list(data.values())
        fig, ax = plt.subplots()
        ax.set_title('Share of different events', fontsize = 16, fontweight = 'bold')
        ax.pie(values, labels=labels, explode=explode, autopct='%1.1f%%', colors=colors)
        ax.axis('equal')
        plt.savefig('./policies/inngjerdingen_moeller/simulation_results/test1.png')
        plt.show()

    def visualize_aggregated_results(self, filename, ):
        try:
            path = 'policies/inngjerdingen_moeller/simulation_results/'+filename
            with open(path, 'r', newline='') as f:
                reader = csv.reader(f, delimiter=',')
                line_count = 0
                aggregated_data = {'trips':0, 'congestion':0, 'starvation':0, 'roaming for bikes':0, 'roaming distance for locks':0, 'roaming distance for bikes':0} 
                for row in reader:
                    if line_count == 0:
                        line_count += 1
                    else:
                        aggregated_data['trips'] += int(row[1])
                        aggregated_data['congestion'] += int(row[5])
                        aggregated_data['starvation'] += int(row[2])
                        aggregated_data['roaming for bikes'] += int(row[3])
                        aggregated_data['roaming distance for locks'] += float(row[6])
                        aggregated_data['roaming distance for bikes'] += float(row[4])
            self.visualize_violations_and_roaming(aggregated_data)
            self.visualize_total_roaming_distances(aggregated_data)
            self.visualize_average_roaming_distances(aggregated_data)
            self.visualize_share_of_events(aggregated_data)
        except:
            print("Error with CSV")
            return None

def write_sim_results_to_file(filename, simulator, duration, append=False):
    header = ['Duration','Trips','Starvations','Roaming for bikes', 'Roaming distance for bikes', 'Congestions/Roaming for locks', 'Roaming distance for locks']
    data=[duration, simulator.metrics.get_aggregate_value('trips'), simulator.metrics.get_aggregate_value('starvation'), simulator.metrics.get_aggregate_value('roaming for bikes'),round(simulator.metrics.get_aggregate_value('roaming distance for bikes'),2), simulator.metrics.get_aggregate_value('congestion'), round(simulator.metrics.get_aggregate_value('roaming distance for locks'),2)]
    try:
        path= 'policies/inngjerdingen_moeller/simulation_results/'+filename
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
        print("Error writing to CSV")
        return None