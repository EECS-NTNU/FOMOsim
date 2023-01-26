import numpy as np
import matplotlib.pyplot as plt
import csv

class VisualizeResults():
    def __init__(self, simulator):
        self.simulator = simulator
    
    def visualize_violations_and_roaming(self):
        data = {'Starvations':self.simulator.metrics.get_aggregate_value('starvation'),
                'Congestions':self.simulator.metrics.get_aggregate_value('congestion'),
                'Roaming for bikes':self.simulator.metrics.get_aggregate_value('roaming for bikes')}
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
        print("Error")
        return None