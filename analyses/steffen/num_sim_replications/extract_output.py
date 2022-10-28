#run with current working directory
import os
import sys
from pathlib import Path

path = Path(__file__).parents[3]
os.chdir(path)
#print(os.getcwd())

sys.path.insert(0, '') #make sure the modules are found in the new working directory

#-----------------

import pandas as pd
from create_runs_base_settings import * #SEEDS, ABBRVS2

#--------------------

filename = os.getcwd()+'\\analyses\\steffen\\num_sim_replications\\output_num_reps.csv'
df = pd.read_csv (filename,sep=';',
                        names=['analysis_type',	'alpha',	'beta','gamma','instance','analysis_name',
                        'target_state','policy','numvehicles','starvations','congestions','violations','trips','starvations_std'	,'congestions_std', 
                        'N_starv','N_cong','N_FINAL',
                        'time_start','duration'])

city_ranking = CITY_RANKING 
df['week'] = df['instance'].apply(lambda x: int(x.split('_W')[1]))
df['city'] = df['instance'].apply(lambda x: x.split('_W')[0])
df['city_ranking'] = df['city']
df = df.replace({"city_ranking": city_ranking})
df = df.sort_values(by=['week'],ascending=True)
df = df.sort_values(by=['city_ranking'],ascending=True)

for i in ['instance','N_starv','N_cong','N_FINAL','starvations','congestions','trips']:
    string = ''
    for j in range(len(list(df[i]))):
        value = list(df[i])[j]
        if (j)%4 == 0:
            string+='&'
        if i == 'trips':
            value = round(value/1000,1)
        elif i in ['starvations','congestions']:
            value = round(value,1)
        string +=' & '+str(value)
    print(string)

string = ''
for i,j in zip(df['instance'],df['N_FINAL']):
    string+= '"'+str(i)+'":'+str(j)+','
print(string)

