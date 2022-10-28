#JUST MOVE TO THE MAIN FOLDER!!

import os 
import sys
from pathlib import Path

path = Path(__file__).parents[3]
os.chdir(path)
#print(os. getcwd())

sys.path.insert(0, '') #make sure the modules are found in the new working directory

###############################################################################


#!/bin/python3
"""
FOMO simulator, visualises results from a cluster run
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy

import os
print(os.getcwd())

#os.chdir('C:\\Users\\steffejb\\OneDrive - NTNU\\Work\\GitHub\\FOMO-sim\\fomo')
#filename = 'output_param_tuning_all.csv' #'output_param_tuning_all.csv'

from analyses.steffen.num_sim_replications.helpers import ci_half_length
from create_runs_base_settings import * #SEEDS, ABBRVS2









###############################################################################
## Some postprocessing used by Steffen
###############################################################################

if __name__ == "__main__":

    #output_param_tuning_all.csv
    #output_param_tuning_shorter_service.csv
    filename = 'output_param_tuning_all.csv'
    df = pd.read_csv (os.getcwd()+'\\analyses\\steffen\\param_tuning\\'+filename,sep=';',
                        names=['run',	'instance',	'analyses','target_state','policy','num_vehicles',
                        'trips','starvations','congestions','starvations_std'	,'congestions_std', 'time_start','duration'])


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
    df['violations_std'] = np.sqrt(df['starvations_std']**2 + df['congestions_std']**2)

    factor = 100/df['trips']
    df['cong_trips'] =  df['congestions']*factor
    df['cong_trips_std'] =  df['congestions_std']*factor
    df['starv_trips'] =  df['starvations']*factor
    df['starv_trips_std'] =  df['starvations_std']*factor
    df['lost_trips'] = df['cong_trips'] + df['starv_trips'] 
    df['lost_trips_std'] = df['violations_std']*factor

    df['week'] = df['instance'].apply(lambda x: x.split('_W')[1])
    df['city'] = df['instance'].apply(lambda x: x.split('_W')[0])

    df = df.sort_values(by=['lost_trips'],ascending=True)
    instances = df['instance'].unique()

    # REMOVE EDINBURGH (if it exists)
    df = df.loc[(df['city']!='EH')]

    df_init = copy.deepcopy(df)

    # Focus on 1 or 2 vehicles

    output = {nv:None for nv in [1,2]}

    for num_veh in [1,2]:
        #num_veh = 1
        df = copy.deepcopy(df_init.loc[(df_init['num_vehicles']==num_veh)])
        df = pd.concat([df,df_init.loc[(df_init['num_vehicles']==0)]])

        all_results = pd.DataFrame({'instance':instances})
        all_results = all_results.sort_values(by=['instance'],ascending=True)

        ###############################################################################
        ## 1: Get the best/worst performing configuration(s) across all instances (on average)
        ###############################################################################


        aggregate_across_instances = df.groupby(['w1', 'w2', 'w3','w4']).agg({'lost_trips':'mean', 'violations':'mean'})
        aggregate_across_instances = aggregate_across_instances.sort_values(by=['lost_trips'],ascending=True).reset_index()
        
        what_works_best = aggregate_across_instances.sort_values(by=['lost_trips'],ascending=True).reset_index().head(5)
        print('best 5 on average:')
        print(what_works_best)
        # TIME TO VIOLATION IS BEST
        # Followed by some two factors

        what_works_worst = aggregate_across_instances.sort_values(by=['lost_trips'],ascending=False).reset_index().head(5)
        print('worst 5 on average:')
        print(what_works_worst)
        # NOT STEERING AT ALL IS WORST  -> skip this
        # BRIEFLY AFTER FOLLOWED BY DRIVING TIME, but also some mixes. 

        avg_best_config = what_works_best.iloc[0][['w1','w2','w3','w4']].values.flatten().tolist()
        avg_best_config = [str(i) for i in avg_best_config]
        best_string = ", ".join(avg_best_config)
        
        avg_worst_config = what_works_worst.iloc[1][['w1','w2','w3','w4']].values.flatten().tolist()  #the worst one is do nothing, so pick second element
        avg_worst_config = [str(i) for i in avg_worst_config]
        worst_string = ", ".join(avg_worst_config)


        performance_avg_best = df.loc[df['analyses'].str.contains(best_string)]
        performance_avg_best = performance_avg_best.rename(columns={'lost_trips': 'lost_trips_avg_best'}) #, index={'ONE': 'Row_1'}
        performance_avg_best = performance_avg_best.rename(columns={'lost_trips_std': 'lost_trips_avg_best_std'}) #, index={'ONE': 'Row_1'}


        performance_avg_worst = df.loc[df['analyses'].str.contains(worst_string)]
        performance_avg_worst = performance_avg_worst.rename(columns={'lost_trips': 'lost_trips_avg_worst'}) #, index={'ONE': 'Row_1'}
        performance_avg_worst = performance_avg_worst.rename(columns={'lost_trips_std': 'lost_trips_avg_worst_std'}) #, index={'ONE': 'Row_1'}
        
        all_results = pd.merge(all_results, performance_avg_best[['instance','lost_trips_avg_best','lost_trips_avg_best_std']], on=['instance'])
        all_results = pd.merge(all_results, performance_avg_worst[['instance','lost_trips_avg_worst','lost_trips_avg_worst_std']], on=['instance'])
        
        ###############################################################################
        ## 2: Get the best performing configuration(s) + do nothing for each instance
        ###############################################################################

        
        policy_best = {instance:None for instance in instances}
        lost_trips = {instance:None for instance in instances}
        lost_trips_std = {instance:None for instance in instances}
        lost_trips_dn = {instance:None for instance in instances}
        lost_trips_dn_std = {instance:None for instance in instances}
        for instance in instances:
            #[city,week] = instance.split('_W')
            df_subset = df.loc[(df['instance']==instance)].reset_index()

            if len(df_subset)>0:
                policy_best[instance] = df_subset['analyses'][0]   #(instance,num_veh)
                lost_trips[instance] = df_subset['lost_trips'][0]
                lost_trips_std[instance] = df_subset['lost_trips_std'][0]
            
            df_subset = df_init.loc[(df_init['instance']==instance)&(df_init['num_vehicles']==0)].reset_index() 
            
            if len(df_subset)>0:
                lost_trips_dn[instance] = df_subset['lost_trips'][0]
                lost_trips_dn_std[instance] = df_subset['lost_trips_std'][0]

            #lost_trips_dn.append(df_init.loc[(df_init['instance']==instance)&(df_init['num_vehicles']==0)].reset_index()['lost_trips'][0])
        

        output_best = pd.DataFrame.from_dict({'instance':instances,
                                                'best_policy':policy_best.values(),
                                                'lost_trips_best':lost_trips.values(),
                                                'lost_trips_best_std':lost_trips_std.values(),
                                                'lost_trips_dn':lost_trips_dn.values(),
                                                'lost_trips_dn_std':lost_trips_dn_std.values()
                                                })
        
        output_best
        
        all_results = pd.merge(all_results, output_best, on=['instance'])


        #df_subset = df.loc[(df['instance']=='OS_W22')].reset_index()


        #----------#
        #how performs each best one across the others?
        #----------#


        city_ranking = {'EH':0,'TD':1,'BG':2,'OS':3}

        all_results['week'] = all_results['instance'].apply(lambda x: int(x.split('_W')[1]))
        all_results['city'] = all_results['instance'].apply(lambda x: x.split('_W')[0])
        all_results['city_ranking'] = all_results['city']
        all_results = all_results.replace({"city_ranking": city_ranking})
        all_results = all_results.sort_values(by=['week'],ascending=True)
        all_results = all_results.sort_values(by=['city_ranking'],ascending=True)

        output[num_veh] = all_results

    ###############################################################################
    ## 3: Plotting
    ###############################################################################

    #for num_vehicles, ax in zip([1,2], axs.ravel()):

    for num_vehicles in [1,2]:
        fig, ax  = plt.subplots(1, 1) #sharex=True

        #actual plot
        
        df_plot = output[num_vehicles]

        ax.set_xlabel("instance")
        ax.set_ylabel("lost trips (%)")
        ax.set_title(str(num_vehicles)+' vehicle')

        x = df_plot['instance']
        y_name = ['lost_trips_dn','lost_trips_avg_worst','lost_trips_avg_best','lost_trips_best']
        seeds = [NUM_SEEDS[instance] for instance in x]
        labels=['Do nothing','Worst policy (avg.)','Best policy (avg.)','Best policy (per instance)']
        colors = ['red','orange','blue','green']
        for i in range(len(y_name)):
            y = df_plot[y_name[i]]
            std = df_plot[y_name[i]+'_std']
            N = seeds[i]
            half_length = ci_half_length(n=N,alpha=0.05,sample_std=std)    
            ax.plot(x,y,label=labels[i], color=colors[i], linestyle = 'dashed', marker='.')
            ax.fill_between(x,y-half_length,y+half_length,color=colors[i], alpha=.2)  
        #https://stackoverflow.com/questions/43941245/line-plot-with-data-points-in-pandas

        #ax.fill_between(all_results['instance'], all_results['lost_trips_best'], all_results['lost_trips_avg_best'])

        plt.legend(loc="upper left")
        
        fact = 0.75
        fig_size = [18.5, 10.5]
        fig.set_size_inches(fig_size[0]*fact, fig_size[1]*fact)
        #plt.subplots_adjust())
        
        location = 'C:\\Users\\steffejb\\OneDrive - NTNU\\Work\\Projects\\FOMO\Results\\Steffen\\ParameterTuning\\'
        filename = 'par_tuning_res_6am_8pm_numveh'+str(num_vehicles)+'.pdf'
        plt.savefig(location+filename, dpi=150)
        plt.show()

        #https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html



