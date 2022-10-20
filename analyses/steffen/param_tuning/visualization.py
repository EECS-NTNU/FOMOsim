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
os.chdir('C:\\Users\\steffejb\\OneDrive - NTNU\\Work\\GitHub\\FOMO-sim\\fomo')

###############################################################################
## Some postprocessing used by Steffen
###############################################################################

if __name__ == "__main__":


        df = pd.read_csv (os.getcwd()+'\\analyses\\steffen\\param_tuning\\output_param_tuning_all.csv',sep=';',
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

            performance_avg_worst = df.loc[df['analyses'].str.contains(worst_string)]
            performance_avg_worst = performance_avg_worst.rename(columns={'lost_trips': 'lost_trips_avg_worst'}) #, index={'ONE': 'Row_1'}
            
            all_results = pd.merge(all_results, performance_avg_best[['instance','lost_trips_avg_best']], on=['instance'])
            all_results = pd.merge(all_results, performance_avg_worst[['instance','lost_trips_avg_worst']], on=['instance'])
            
            ###############################################################################
            ## 2: Get the best performing configuration(s) + do nothing for each instance
            ###############################################################################

            
            policy_best = {instance:None for instance in instances}
            lost_trips = {instance:None for instance in instances}
            lost_trips_dn = {instance:None for instance in instances}
            for instance in instances:
                #[city,week] = instance.split('_W')
                df_subset = df.loc[(df['instance']==instance)].reset_index()

                if len(df_subset)>0:
                    policy_best[instance] = df_subset['analyses'][0]   #(instance,num_veh)
                    lost_trips[instance] = df_subset['lost_trips'][0]

                df_subset = df_init.loc[(df_init['instance']==instance)&(df_init['num_vehicles']==0)].reset_index() 
                
                if len(df_subset)>0:
                    lost_trips_dn[instance] = df_subset['lost_trips'][0]

                #lost_trips_dn.append(df_init.loc[(df_init['instance']==instance)&(df_init['num_vehicles']==0)].reset_index()['lost_trips'][0])
            

            output_best = pd.DataFrame.from_dict({'instance':instances,
                                                    'best_policy':policy_best.values(),
                                                    'lost_trips_best':lost_trips.values(),
                                                    'lost_trips_dn':lost_trips_dn.values()
                                                    })
            
            all_results = pd.merge(all_results, output_best, on=['instance'])


            #----------#
            #how performs each best one across the others?
            #----------#



            ###############################################################################
            ## 3: Plotting
            ###############################################################################

            city_ranking = {'TD':1,'BG':2,'OS':3}

            all_results['week'] = all_results['instance'].apply(lambda x: x.split('_W')[1])
            all_results['city'] = all_results['instance'].apply(lambda x: x.split('_W')[0])
            all_results['city_ranking'] = all_results['city']
            all_results = all_results.replace({"city_ranking": city_ranking})
            all_results = all_results.sort_values(by=['week'],ascending=True)
            all_results = all_results.sort_values(by=['city_ranking'],ascending=True)

            output[num_veh] = all_results


        fig, axs  = plt.subplots(2, 1, sharex=True)

        for num_vehicles, ax in zip([1,2], axs.ravel()):
            #actual plot
            
            df_plot = output[num_vehicles]

            ax.set_xlabel("instance")
            ax.set_ylabel("lost trips (%)")
            ax.set_title(str(num_vehicles)+' vehicle')

            ax.plot(df_plot['instance'],df_plot['lost_trips_dn'],label='do nothing',
                    color='red', linestyle = 'dashed', marker='.')
            ax.plot(df_plot['instance'],df_plot['lost_trips_avg_worst'],label='avg. worst',
                    color='orange', linestyle = 'dashed', marker='.')
            
            ax.plot(df_plot['instance'],df_plot['lost_trips_avg_best'],label='avg. best',
                    color='blue', linestyle = 'dashed', marker='.')
            ax.plot(df_plot['instance'],df_plot['lost_trips_best'],label='best',
                    color='green', linestyle = 'dashed', marker='.')
            
            
            #https://stackoverflow.com/questions/43941245/line-plot-with-data-points-in-pandas

            #ax.fill_between(all_results['instance'], all_results['lost_trips_best'], all_results['lost_trips_avg_best'])

        plt.legend(loc="upper left")
        
        fact = 0.75
        fig_size = [18.5, 10.5]
        fig.set_size_inches(fig_size[0]*fact, fig_size[1]*fact)
        #plt.subplots_adjust())
        
        location = 'C:\\Users\\steffejb\\OneDrive - NTNU\\Work\\Projects\\FOMO\Results\\Steffen\\ParameterTuning\\'
        filename = 'par_tuning_res_6am_8pm.pdf'
        plt.savefig(location+filename, dpi=150)
        plt.show()

            #https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html








            #df2 = df.loc[df['violations']<np.percentile(df['violations'],10)]
            #df3 = df.loc[df['violations']>np.percentile(df['violations'],90)]

            #df_extreme = df.loc[(df['w1']<0.001) | (df['w2']<0.001) | (df['w3']<0.001) | (df['w4']<0.001)] 
            #df_single_measure = df.loc[(df['w1']>0.999) | (df['w2']>0.999) | (df['w3']>0.999) | (df['w4']>0.999)] 



            ################
            ## ANALYSIS 1 ##
            ################

            # TO DO