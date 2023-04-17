#JUST MOVE TO THE MAIN FOLDER!!

import os
from platform import java_ver 
import sys
from pathlib import Path
from tkinter.tix import INCREASING

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

from experiments.steffen.helpers import ci_half_length
from runs_base_settings import * #SEEDS, ABBRVS2


###############################################################################
## Some postprocessing used by Steffen
###############################################################################

if __name__ == "__main__":

    #output_param_tuning_all.csv
    #output_param_tuning_shorter_service.csv
    filename = 'output_newest.csv'
    df = pd.read_csv (os.getcwd()+'\\experiments\\steffen\\param_tuning\\'+filename,sep=';',
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
    

    # REMOVE EDINBURGH (if it exists)
    df = df.loc[(df['city']!='EH')]
    df['instance_ranking'] = df['instance']
    df = df.replace({"instance_ranking": INSTANCE_RANKING})
    df = df.sort_values(by=['instance_ranking'],ascending=True)
    instances = df['instance'].unique() #ordered 

    df_init = copy.deepcopy(df)

    best_policy = {(nv,instance):None for nv in [1,2] for instance in instances}
    best_avg_policies = {nv:None for nv in [1,2]}
    worst_avg_policies = {nv:None for nv in [1,2]}


    for num_veh in [1,2]:
        #num_veh = 1
        df = copy.deepcopy(df_init.loc[(df_init['num_vehicles']==num_veh)])
        df = pd.concat([df,df_init.loc[(df_init['num_vehicles']==0)]])

        ###############################################################################
        ## 1: Get the best/worst performing configuration(s) across all instances (on average)
        ###############################################################################


        aggregate_across_instances = df.groupby(['w1', 'w2', 'w3','w4','analyses']).agg({'lost_trips':'mean', 'violations':'mean'})
        aggregate_across_instances = aggregate_across_instances.sort_values(by=['lost_trips'],ascending=True).reset_index()
        
        what_works_best = aggregate_across_instances.sort_values(by=['lost_trips'],ascending=True).reset_index().head(5)
        print('best 5 on average:')
        print(what_works_best)
        best_avg_policies[num_veh] = what_works_best['analyses'].values.flatten().tolist()
        # TIME TO VIOLATION IS BEST
        # Followed by some two factors

        what_works_worst = aggregate_across_instances.sort_values(by=['lost_trips'],ascending=False).reset_index().head(5)
        print('worst 5 on average:')
        print(what_works_worst)
        worst_avg_policies[num_veh] = what_works_worst['analyses'].values.flatten().tolist()
        # NOT STEERING AT ALL IS WORST  -> skip this
        # BRIEFLY AFTER FOLLOWED BY DRIVING TIME, but also some mixes. 


        ###############################################################################
        ## 2: Get the best performing configuration(s) + do nothing for each instance
        ###############################################################################

                
        for instance in instances:
            #[city,week] = instance.split('_W')
            df_subset = df.loc[(df['instance']==instance)]
            df_subset = df_subset.sort_values(by=['lost_trips'],ascending=True).reset_index()
            if len(df_subset)>0:
                best_policy[(num_veh,instance)] = df_subset['analyses'][0]   #(instance,num_veh)


    ###############################################################################
    ## 3: Plotting
    ###############################################################################

        #for num_vehicles, ax in zip([1,2], axs.ravel()):


        fig, ax  = plt.subplots(1, 1) #sharex=True

        #actual plot

        ax.set_xlabel("instance")
        ax.set_ylabel("lost trips (%)")
        ax.set_title(str(num_veh)+' vehicle')

        x = instances
        analysis_name = ['do_nothing',worst_avg_policies[num_veh][1],best_avg_policies[num_veh][0]] #best_policy[num_veh,'OS_W10']
        labels = ['Do nothing','Worst policy (avg.)','Best policy (avg.)']
        colors = ['red','orange','blue']
        seeds = [NUM_SEEDS[instance] for instance in x]

        for i in range(len(analysis_name)):
            if analysis_name[i] == 'do_nothing':
                df_subset = df_init.loc[(df_init['num_vehicles']==0)]
            else:
                df_subset = df_init.loc[(df_init['num_vehicles']==num_veh) & (df_init['analyses']==analysis_name[i])]
            df_subset = df_subset.sort_values(by=['instance_ranking'],ascending=True)
            y = df_subset['lost_trips'].values.flatten().tolist()
            std = df_subset['lost_trips_std'].values.flatten().tolist()
            half_lengths = [ci_half_length(n=seeds[j],alpha=0.05,sample_std=std[j]) for j in range(len(instances))]  
            ax.plot(x,y,label=labels[i], color=colors[i], linestyle = 'dashed', marker='.')
            ax.fill_between(x,  [yy - hl for (yy,hl) in zip(y, half_lengths)],
                                [yy + hl for (yy,hl) in zip(y, half_lengths)],
                            color=colors[i], alpha=.2)  
        #https://stackoverflow.com/questions/43941245/line-plot-with-data-points-in-pandas


        #best policy per instance
        x = []
        y = []
        half_lengths = []
        
        for instance in instances:
            best_pol = best_policy[(num_veh,instance)]
            df_subset = df_init.loc[(df_init['num_vehicles']==num_veh) & (df_init['analyses']==best_pol) 
                                        & (df_init['instance'] == instance)].reset_index()
            x.append(instance)
            y.append(df_subset.at[0,'lost_trips'])
            std = df_subset.at[0,'lost_trips_std']
            half_lengths.append(ci_half_length(n=NUM_SEEDS[instance],alpha=0.05,sample_std=std))
        print(y)
        print(x)
        print(half_lengths)
        color = 'green'
        ax.plot(x,y,label='Best policy per instance', color = color, linestyle = 'dashed', marker='.') #color = 
        ax.fill_between(x,  [yy - hl for (yy,hl) in zip(y, half_lengths)],
                            [yy + hl for (yy,hl) in zip(y, half_lengths)],color=color, alpha=.2)  
        #performance of a specific best policy
        inst_specific = ['OS_W10']
        for inst in inst_specific:
            best_pol = best_policy[(num_veh,inst)]
            df_subset = df_init.loc[(df_init['num_vehicles']==num_veh) & (df_init['analyses']==best_pol)].reset_index()
            x = df_subset['instance']
            y = df_subset['lost_trips']
            std = df_subset['lost_trips_std']
            half_lengths = [ci_half_length(n=seeds[j],alpha=0.05,sample_std=std[j]) for j in range(len(instances))]
            color = 'black'
            ax.plot(x,y,label='best policy from '+inst, color=color, linestyle = 'dashed', marker='.')
            ax.fill_between(x,  [yy - hl for (yy,hl) in zip(y, half_lengths)],
                                [yy + hl for (yy,hl) in zip(y, half_lengths)],color=color, alpha=.2)   
        plt.legend(loc="upper left")
        
        fact = 0.75
        fig_size = [18.5, 7.5]
        fig.set_size_inches(fig_size[0]*fact, fig_size[1]*fact)
        #plt.subplots_adjust())
        
        location = 'C:\\Users\\steffejb\\OneDrive - NTNU\\Work\\Projects\\Ongoing\\FOMO\\Results\\Steffen\\ParameterTuning'
        filename = 'par_tuning_res_6am_8pm_numveh'+str(num_veh)
        extensions = ['.pdf','.png']
        for extension in extensions:
            plt.savefig(location+filename+extension, dpi=150)
        plt.show()

        #https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html



