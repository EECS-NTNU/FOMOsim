import os
import pandas as pd
from settings import *

test_cat = 'upper_threshold'

folder_path = '/Users/isabellam/NTNU/H2023/Prosjektoppgave/fomo/policies/hlm/simulation_results/OS_W31_2V_10S_5D_PILOT_TTT/threshold' # os.getcwd() + '/policies/hlm/threshold'# + RESULT_FOLDER + '/branching'

def get_avg_violations(filepath):
    data = pd.read_csv(filepath, header=None, names = ['Duration','Events','Starvations','No scooters',  'No battery', 'Battery Violation', 'Roaming for bikes', 'Roaming distance for bikes', 'Seed'])
    data['Violations'] = data['Starvations'] + data['Battery Violation']

    return data['Violations'].mean()

def get_avg_time(filepath):
    data = pd.read_csv(filepath, header=None, names = ['Total time', 'Actions'])
    data['Time per action'] = data['Total time'] / data['Actions']

    return data['Time per action'].mean()

def get_avg_events(filepath):
    data = pd.read_csv(filepath, header=None, names = ['Duration','Events','Starvations','No scooters',  'No battery', 'Battery Violation', 'Roaming for bikes', 'Roaming distance for bikes', 'Seed'])

    return data['Events'].mean()

results = []

for filename in os.listdir(folder_path):
    if filename.endswith('.csv') and filename.startswith(test_cat):
            avg_violations = get_avg_violations(folder_path + '/' + filename)
            avg_time = get_avg_time(folder_path + '/' + 'sol_time_' + filename)
            avg_events = get_avg_events(folder_path + '/' + filename)
            service_rate = avg_violations / avg_events * 100
            results.append({'filename': filename, 'avg_violations': avg_violations, 'avg_time': avg_time, 'avg_events': avg_events, 'service_rate': service_rate})


df_results = pd.DataFrame(results)
df_results.sort_values(by=['avg_violations']).reset_index().to_csv(folder_path + '/avg_' + test_cat + '.csv')