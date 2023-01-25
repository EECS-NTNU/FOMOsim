import csv

def write_sim_results_to_file(filename, simulator,duration, append=False):
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