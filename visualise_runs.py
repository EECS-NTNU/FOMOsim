#!/bin/python3
"""
FOMO simulator, visualises results from a cluster run
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

###############################################################################

def lostTripsPlot(cities, policies, starv, starv_stdev, cong, cong_stdev):
    fig, subPlots = plt.subplots(nrows=1, ncols=len(cities), sharey=True)
    fig.suptitle("FOMO simulator - lost trips results", fontsize=15)
    
    if len(cities) == 1:
        subPlots = [ subPlots ]
    w = 0.3
    pos = []
    for city in range(len(cities)):
        pos.append([])
        for i in range(len(cong[city])):
            pos[city].append(starv[city][i] + cong[city][i])

        subPlots[city].bar(policies, starv[city], w, label='Starvation')
        subPlots[city].errorbar(policies, starv[city], yerr = starv_stdev[city], fmt='none', ecolor='red')
        subPlots[city].bar(policies, cong[city], w, bottom=starv[city], label='Congestion')
        
        # skew the upper error-bar with delta to avoid that they can overwrite each other
        delta = 0.05
        policiesPlussDelta = []
        for i in range(len(policies)):
            policiesPlussDelta.append(i + delta) 
        subPlots[city].errorbar(policiesPlussDelta, pos[city], yerr= cong_stdev[city], fmt='none', ecolor='black')
        subPlots[city].set_xlabel(cities[city])
        if city == 0:
            subPlots[city].set_ylabel("Violations (% of total number of trips)")
            subPlots[city].legend()


###############################################################################

if __name__ == "__main__":

    with open("output.csv", "r") as infile:
        lines = infile.readlines()

        run = {}

        for line in lines:
            words = line.split(';')

            n = int(words[0])
            instance_name = words[1]
            analysis_name = words[2]
            trips = float(words[3])
            starvations = float(words[4])
            congestions = float(words[5])
            starvations_stdev = float(words[6])
            congestions_stdev = float(words[7])

            if instance_name not in run:
                run[instance_name] = {}
            run[instance_name][analysis_name] = (trips, starvations, congestions, starvations_stdev, congestions_stdev)

        instance_names = []

        starvations = []
        congestions = []

        starvations_stdev = []
        congestions_stdev = []

        for instance_name in run.keys():
            instance_names.append(instance_name)

            starvations.append([])
            congestions.append([])

            starvations_stdev.append([])
            congestions_stdev.append([])

            analysis_names = []

            for analysis_name in run[instance_name].keys():
                analysis_names.append(analysis_name)

                scale = 100 / run[instance_name][analysis_name][0]

                starvations[-1].append(scale * run[instance_name][analysis_name][1])
                congestions[-1].append(scale * run[instance_name][analysis_name][2])
                starvations_stdev[-1].append(scale * run[instance_name][analysis_name][3])
                congestions_stdev[-1].append(scale * run[instance_name][analysis_name][4])

        print(starvations)
        print(congestions)

        lostTripsPlot(instance_names, analysis_names, starvations, starvations_stdev, congestions, congestions_stdev)

        plt.show()


    ###############################################################################
    ## Some postprocessing used by Steffen
    ###############################################################################

    df = pd.read_csv ('output.csv',sep=';',names=['run',	'Instance',	'Analyses',
                                                'target_state','policy','num_vehicles',
                                                'trips','starvations','congestions',
                                                'starvation_std'	,'congestion_std'])

    def extract_weights(string, index):
        if '[' not in string:
            output = 0
        else:
            output = [float(w) for w in string.split('[')[1].split(']')[0].split(',')][index-1]
        return output

    string = 'Trondheim_W48'


    for i in [1,2,3,4]:
        df['w'+str(i)] = df['Analyses'].apply(extract_weights,index=i)
    df['violations'] = df['starvations'] + df['congestions'] 
    df['service_rate'] =  (1-df['violations']/df['trips'])*100
    df['week'] = df['Instance'].apply(lambda x: x.split('_W')[1] )

    df = df.sort_values(by=['service_rate'],ascending=False)


    what_works_best = df.groupby(['w1', 'w2', 'w3','w4']).agg({'service_rate':'mean', 'violations':'mean'})
    what_works_best = what_works_best.sort_values(by=['service_rate'],ascending=False)


    df_OSL = df.loc[df['Instance']=='Oslo']
    df_INSPECT = df.loc[df['w2']==0.2]


    #df2 = df.loc[df['violations']<np.percentile(df['violations'],10)]
    #df3 = df.loc[df['violations']>np.percentile(df['violations'],90)]

    #df_extreme = df.loc[(df['w1']<0.001) | (df['w2']<0.001) | (df['w3']<0.001) | (df['w4']<0.001)] 
    #df_single_measure = df.loc[(df['w1']>0.999) | (df['w2']>0.999) | (df['w3']>0.999) | (df['w4']>0.999)] 
