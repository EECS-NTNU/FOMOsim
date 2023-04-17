#!/bin/python3
"""
FOMO simulator, visualises results from a cluster run
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy


#if __name__ == "__main__":
import os
import sys
os.chdir('C:\\Users\\steffejb\\OneDrive - NTNU\\Work\\GitHub\\FOMO-sim\\fomo')
sys.path.insert(0, '') #make sure the modules are found in the new working directory

filename = 'output_target_state_final.csv' #'output_param_tuning_all.csv'
###############################################################################
## Some postprocessing used by Steffen
###############################################################################

if __name__ == "__main__":

    #output_param_tuning_all.csv
    #output_param_tuning_shorter_service.csv
    df = pd.read_csv (os.getcwd()+'\\experiments\\steffen\\target_state\\'+filename,sep=';',
                        names=['run',	'instance',	'analyses','target_state','policy','num_vehicles',
                        'trips','starvations','congestions','starvation_std'	,'congestion_std', 'time_start','duration'])


    #1. extract weights data
    def extract_weights(string, index):
        if '[' not in str(string):
            output = 0
        else:
            output = [float(w) for w in string.split('[')[1].split(']')[0].split(',')][index-1]
        return output

    for i in [1,2,3,4]:
        df['w'+str(i)] = df['analyses'].apply(extract_weights,index=i)


    #calculate some additional statistics
    df['violations'] = df['starvations'] + df['congestions'] 
    df['violations_std'] = np.sqrt(df['starvation_std']**2 + df['congestion_std']**2)

    df['cong_trips'] =  (df['congestions']/df['trips'])*100
    df['starv_trips'] =  (df['starvations']/df['trips'])*100
    df['lost_trips'] = df['cong_trips'] + df['starv_trips'] 
    df['week'] = df['instance'].apply(lambda x: x.split('_W')[1])
    df['city'] = df['instance'].apply(lambda x: x.split('_W')[0])

    df = df.sort_values(by=['lost_trips'],ascending=True)
    instances = df['instance'].unique()

    # REMOVE EDINBURGH (if it exists)
    df = df.loc[(df['city']!='EH')]
    
    city_ranking = {'TD':1,'BG':2,'OS':3}
    df['city_ranking'] = df['city']
    df = df.replace({"city_ranking": city_ranking})
    
    df_init = copy.deepcopy(df)
    
    # Focus on 1 or 2 vehicles


    target_states = {'Outflow':'outflow_target_state', 'Even':'evenly_distributed_target_state', 
                    'EqualProb':'equal_prob_target_state','HalfCap':'us_target_state', 'UrbanSharing':'USTargetState'}
    weights_mapping = {'critic': [0.1,0.2,0.3,0.4],'ts_dev': [0.0,0.0,0.0,1.0]}
    

    for num_veh in [1,2]:
        for weight_name,weights in weights_mapping.items(): #weight_name = 'critic',    weights = [0.1,0.2,0.3,0.4]
            #num_veh = 1 
            
            df = copy.deepcopy(df_init.loc[(df_init['num_vehicles']==num_veh)])
            df = df.loc[(df['w1']==weights[0]) &
                        (df['w2']==weights[1]) &
                        (df['w3']==weights[2]) &
                        (df['w4']==weights[3])]
            df = pd.concat([df,df_init.loc[(df_init['num_vehicles']==0)]])

        ###############################################################################
        ## 3: Plotting
        ###############################################################################


            fig, ax  = plt.subplots(1, 1) # sharex=True

            num_vehicles = num_veh
            #for num_vehicles, ax in zip([1,2], axs.ravel()):
                #actual plot
                

            ax.set_xlabel("instance")
            ax.set_ylabel("lost trips (%)")
            ax.set_title(str(num_vehicles)+' vehicle, weights: '+str(weights))

            print('num_veh: ', num_veh)
            print('weights: ', weights)
            for ts, ts_name in target_states.items():
                df_sub = (df[(df['target_state']==ts_name) & (df['num_vehicles']==num_veh) ])
                df_sub = df_sub.sort_values(by=['week'],ascending=True)
                df_sub = df_sub.sort_values(by=['city_ranking'],ascending=True)
                #print(ts)
                #print(df_sub)
                ax.plot(df_sub['instance'],df_sub['lost_trips'],label=ts, linestyle = 'dashed', marker='.')
            df_sub = (df[(df['num_vehicles']==0) ])
            df_sub = df_sub.sort_values(by=['week'],ascending=True)
            df_sub = df_sub.sort_values(by=['city_ranking'],ascending=True)
            ax.plot(df_sub['instance'],df_sub['lost_trips'],label='Do nothing', linestyle = 'dashed', marker='.')
            print('-----')
                
                #https://stackoverflow.com/questions/43941245/line-plot-with-data-points-in-pandas

                #ax.fill_between(all_results['instance'], all_results['lost_trips_best'], all_results['lost_trips_avg_best'])

            plt.legend(loc="upper left")
            
            fact = 0.75
            fig_size = [18.5, 10.5]
            fig.set_size_inches(fig_size[0]*fact, fig_size[1]*fact)
            #plt.subplots_adjust())
            
            location = 'C:\\Users\\steffejb\\OneDrive - NTNU\\Work\\Projects\\Ongoing\\FOMO\Results\\Steffen\\target_state\\'
            filename = 'ts_nv'+str(num_veh)+'_'+weight_name+'.png'   #.pdf
            plt.savefig(location+filename, dpi=150)
            plt.show()

                #https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html



