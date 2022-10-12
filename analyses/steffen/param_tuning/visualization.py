#!/bin/python3
"""
FOMO simulator, visualises results from a cluster run
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



#if __name__ == "__main__":



###############################################################################
## Some postprocessing used by Steffen
###############################################################################
df = pd.read_csv ('output.csv',sep=';',names=['run',	'instance',	'analyses',
                                            'target_state','policy','num_vehicles',
                                            'trips','starvations','congestions',
                                            'starvation_std'	,'congestion_std'])

#1. extract weights data
def extract_weights(string, index):
    if '[' not in string:
        output = 0
    else:
        output = [float(w) for w in string.split('[')[1].split(']')[0].split(',')][index-1]
    return output

for i in [1,2,3,4]:
    df['w'+str(i)] = df['analyses'].apply(extract_weights,index=i)

#calculate some additional statistics
df['violations'] = df['starvations'] + df['congestions'] 
df['cong_trips'] =  (df['congestions']/df['trips'])*100
df['starv_trips'] =  (df['starvations']/df['trips'])*100
df['lost_trips'] = df['cong_trips'] + df['starv_trips'] 
df['week'] = df['instance'].apply(lambda x: x.split('_W')[1] )
df['city'] = df['instance'].apply(lambda x: x.split('_W')[0] )

df = df.sort_values(by=['lost_trips'],ascending=True)

what_works_best = df.groupby(['w1', 'w2', 'w3','w4']).agg({'lost_trips':'mean', 'violations':'mean'})
what_works_best = what_works_best.sort_values(by=['lost_trips'],ascending=True).reset_index().head(20)
#Only steering on time to violation works best across all instances... (when using the same)

N_best = 1
best_policies = {}
for instance in df['instance'].unique():
    [city,week] = instance.split('_W')
    for num_veh in df['num_vehicles'].unique():
        # city = 'Oslo'
        # week = 33
        # num_veh = 2
        df_subset = df.loc[(df['city']==city)& (df['week']==str(week)) & 
            (df['num_vehicles']==num_veh)].reset_index()
        best_policies[(instance,num_veh)] = [df_subset['analyses'][i] for i in range(N_best)]

best_policies_oslo = {key: value for key, value in best_policies.items() if 'Oslo' in key[0] }



#df2 = df.loc[df['violations']<np.percentile(df['violations'],10)]
#df3 = df.loc[df['violations']>np.percentile(df['violations'],90)]

#df_extreme = df.loc[(df['w1']<0.001) | (df['w2']<0.001) | (df['w3']<0.001) | (df['w4']<0.001)] 
#df_single_measure = df.loc[(df['w1']>0.999) | (df['w2']>0.999) | (df['w3']>0.999) | (df['w4']>0.999)] 



################
## ANALYSIS 1 ##
################

# TO DO