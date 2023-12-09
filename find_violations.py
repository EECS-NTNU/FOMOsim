import os
import pandas as pd
from settings import *

test_cat = 'num_scenarios'

folder_path = os.getcwd() + '/policies/hlm/' + RESULT_FOLDER

def get_avg_violations(filepath):
    data = pd.read_csv(filepath, header=None, names = ['Duration','Events','Starvations','No scooters',  'No battery', 'Battery Violation', 'Roaming for bikes', 'Roaming distance for bikes', 'Seed'])
    data['Violations'] = data['Starvations'] + data['Battery Violation']

    return data['Violations'].mean()

def get_avg_time(filepath):
    data = pd.read_csv(filepath, header=None, names = ['Total time', 'Actions'])
    data['Time per action'] = data['Total time'] / data['Actions']

    return data['Time per action'].mean()

results = []

for filename in os.listdir(folder_path):
    if filename.endswith('.csv') and filename.startswith(test_cat):
            avg_violations = get_avg_violations(folder_path + '/' + filename)
            avg_time = get_avg_time(folder_path + '/' + 'sol_time_' + filename)
            results.append({'filename': filename, 'avg_violations': avg_violations, 'avg_time': avg_time})


df_results = pd.DataFrame(results)
df_results.sort_values(by=['avg_violations']).reset_index().to_csv(folder_path + '/avg_' + test_cat + '.csv')