import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np 
import pandas as pd
import os
from output.visualizer import totime
import csv
from datetime import datetime

#used for branch number
def plot_bar_chart():
    data = [7429,4827,4383,3670,3355,2346,2287,2682,2690,2577,2144,2025,2219,1701,1278,1672,1661,1649,1509,1496]
    # data= [56480,43783,40315,34020,29371,30084,26740,26370,23897,25395,22199,21074,20400,19494,17802,22068,19632,20724,18716,18436]
    total = sum(data)

    fig, ax = plt.subplots()
    ax.bar([i for i in range(1,len(data)+1)], data, color='#2F5597')

    # Add labels and title
    ax.set_xlabel('Branch', fontsize=13)
    ax.set_ylabel('# scenarios', fontsize=13)
    ax.set_title('Branch selection', fontsize=15)
    plt.xticks([i for i in range(1,len(data)+1)])

    # Add percentage values as text over the bars
    for i, val in enumerate(data):
        percentage = (val / total) * 100
        ax.text(i+1, val+30, f"{percentage:.1f}%", ha='center')
    # Show the plot
    plt.show()

def violin_plot():
    # Example data
    category_names = ['Category A', 'Category B', 'Category C']
    y_values = [[random.randint(30, 90) for _ in range(20)] for _ in range(3)]

    # Calculate mean values for each category
    mean_values = [np.mean(y) for y in y_values]

    # Create violin plot
    fig, ax = plt.subplots()
    ax.violinplot(y_values)

    # Add mean lines
    for i, mean in enumerate(mean_values):
        ax.axhline(y=mean, linestyle='--', color='r', label=f'Mean value for {category_names[i]}')

    # Set labels and titles
    ax.set_xticks(range(1, len(category_names) + 1))
    ax.set_xticklabels(category_names)
    ax.set_xlabel('Category')
    ax.set_ylabel('Number of violations')
    ax.set_title('Violin plot for different categories')

    # Add legend
    ax.legend()

    # Show plot
    plt.show()

def roaming_shares():
    # Define the data for the plot
    distances = np.linspace(0, 600, 100) / 1000  # in km
    share_of_users_willing_to_roam = [-1.65*x**2 - 0.70*x + 1 for x in distances]

    # Define the data for the scatter plot
    scatter_distances = [0, 100, 200, 300, 400, 500, 600]  # in m
    scatter_share_of_users_willing_to_roam = [1, 0.93, 0.85, 0.63, 0.44, 0.26, 0.08]

    # Create the plot
    plt.plot(distances, share_of_users_willing_to_roam, label='p(x) = −1.65x^2 − 0.70x + 1', color='#ED7D31', linewidth=2)
    plt.scatter(np.array(scatter_distances) / 1000, scatter_share_of_users_willing_to_roam, marker='o', alpha=1, label='Survey results', color='#2F5597')
    
    # Add labels and title to the plot
    plt.xlabel('Distance (km)', fontsize=12)
    plt.ylabel('Share of users', fontsize=12)
    plt.title('Willingness to roam',fontsize=15)

    # Add numeric values to the scatter plot
    for i in range(len(scatter_distances)):
        plt.text(scatter_distances[i]/1000, scatter_share_of_users_willing_to_roam[i]+0.015, f'{scatter_share_of_users_willing_to_roam[i]:.2f}', fontsize=11)

    # Move legend in front of the scatter plot
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [1, 0]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    plt.grid(axis='y')

    # Show the plot
    plt.show()

def solution_times():
    times_1 = [0.286910127658834,0.506116209212262,0.68730214648737,0.909322110457106,1.21355056572913]
    times_2 =[0.296412638190561,0.74995476559135,1.10196163457581,5.54463300475282,9.817172741]
    times_3 =[0.310755980872849,0.832518047234595,1.22950507014803,9.216695174,16.2211978]
    times_4 = [0.322126505886221,0.908936786108845,1.64438653582492,13.99606311,23.38967546]
    times_5 =[0.349797594891064,1.03106718980004,1.54947187919452,17.4451748228995,30.5476270707817]
    times_6 = [0.354972953505799,1.09330931905914,1.64292984683035,19.9590297774495,35.6893297259586]
    times_7 =[0.371898019323232,1.17172450812142,1.79121832686287,22.7327883358421,39.766816543801]

    betas = [1, 3, 5, 7, 10] 

    # Create the plot
    # plt.yscale("log")
    max_limit = [10, 10, 10, 10, 10] 

    # plt.plot(betas, max_limit, label="max solution time", color='gray', linewidth=2, ls='--')
    plt.plot(betas, times_1, label= r'$\alpha=1$', color='#ED7D31', linewidth=2, marker='o')
    plt.plot(betas, times_2, label=r'$\alpha=2$', color='#2F5597', linewidth=2, marker='o')
    plt.plot(betas, times_3, label=r'$\alpha=3$', color='green', linewidth=2, marker='o')
    plt.plot(betas, times_4, label=r'$\alpha=4$', color='purple', linewidth=2, marker='o')
    plt.plot(betas, times_5, label=r'$\alpha=5$', color='pink', linewidth=2, marker='o')
    plt.plot(betas, times_6, label=r'$\alpha=6$', color='red', linewidth=2, marker='o')
    plt.plot(betas, times_7, label=r'$\alpha=7$', color='brown', linewidth=2, marker='o')

    # Add labels and title to the plot
    plt.xlabel(r'$\beta$', fontsize=13)
    plt.ylabel('Solution time (s)', fontsize=13)
    
    plt.title('Solution time for various combinations of '+r'$\alpha$'+' and '+ r'$\beta$',fontsize=15)
    
    plt.legend()

    plt.grid(axis='y', alpha=0.5)
    plt.yticks(np.arange(0, 45, 5))
    plt.xticks([1,3,5,7,10])
    # Show the plot
    plt.show()

def solution_quality():
    q0 = [2876.8,2618.8,2653.7,2673.1,2635,2891.5,2906.6]
    q1 =[2924.8,2709.7,2666.1,2653.2,2794.6,2861.4,2894.5]
    q2 =[2834.3,2600.1,2529.5,2542.1,2600,2661.5]
    q3 = [2865.5,2635.3,2566.1,2678.5,2711.3]
    q4 =[2881.7,2582.4,2607.6,2698.8,2753]
    q5 = [2884.1,2592.2,2603.1,2697.5,2829.5]
    q6 =[2841.3,2612.2,2641,2680.2,2706.1]

    betas1_20 = [1, 3, 5, 7, 9, 11, 13]
    betas1_15 = [1, 3, 5, 7, 9, 11] 
    betas1_10 = [1, 3, 5, 7, 9]

    # Create the plot
    # plt.yscale("log")
    # max_limit = [10, 10, 10, 10, 10]

    # plt.plot(betas, max_limit, label="max solution time", color='gray', linewidth=2, ls='--')
    plt.plot(betas1_20, q0, label= r'$\alpha=1$', color='#ED7D31', linewidth=1.5, marker='o')
    plt.plot(betas1_20, q1, label=r'$\alpha=2$', color='#2F5597', linewidth=1.5, marker='o')
    plt.plot(betas1_15, q2, label=r'$\alpha=3$', color='green', linewidth=1.5, marker='o')
    plt.plot(betas1_10, q3, label=r'$\alpha=4$', color='purple', linewidth=1.5, marker='o')
    plt.plot(betas1_10, q4, label=r'$\alpha=5$', color='pink', linewidth=1.5, marker='o')
    plt.plot(betas1_10, q5, label=r'$\alpha=6$', color='red', linewidth=1.5, marker='o')
    plt.plot(betas1_10, q6, label=r'$\alpha=7$', color='brown', linewidth=1.5, marker='o')

    # Add labels and title to the plot
    plt.xlabel(r'$\beta$', fontsize=13)
    plt.ylabel('Failed events', fontsize=13)
    plt.title('Number of failed events for various combinations of '+r'$\alpha$'+' and '+ r'$\beta$',fontsize=15)
    plt.legend(loc='lower right')

    plt.grid(axis='y', alpha=0.5)
    plt.yticks(np.arange(2500, 3000, 50))
    plt.xticks([1,3,5,7,9,11,13], [1,3,5,7,10,15,20])

    # Show the plot
    plt.show()

def box_plot():
    data_normal = [[10711,11064,10808,9781,11248,9652,9974,11537,11067,10449,10418,11422,11404,11415,10992,11697,10513,10133,10219,11039,9906,10051,11064,12085,10824,10713,11022,10940,11232,9855,11756,10773,10894,11732,11070,10872,10153,11522,11324,10343],
            [10721,10622,11018,10059,11034,9771,10161,10729,11071,10643,10198,10763,10453,11582,11065,11337,10619,9499,9662,10724,9581,10234,10883,12156,11606,10313,10544,10684,10560,9885,11689,10660,10925,10908,10519,10849,9740,10973,11128,10365],
            [10652,10588,11050,10042,11235,10538,9929,10930,11037,10689,10096,10913,10440,11788,10831,11273,10757,9378,9562,10841,9623,9739,11083,11997,11309,10542,10609,10803,10593,9740,11507,10504,11088,10869,10370,10913,9632,10962,11186,10610],
            [10656,10598,10859,10020,11680,9711,9919,11386,10497,10181,10186,11332,10180,11331,10711,11024,10742,9828,9977,11141,9580,10467,10813,11528,11574,10600,10545,10594,10770,9960,11842,10001,11153,11383,10802,11082,9910,10714,11019,10544],
            [10962,10497,10786,10374,11176,10193,10442,10907,10579,10494,10343,10730,10835,11442,11072,11537,11315,9527,10148,10464,9277,9912,10747,11882,11703,10613,10441,10748,10675,9819,10990,10279,10834,11050,10705,11166,10033,10857,11268,10100],
            [10305,10558,10953,9986,11341,9675,10024,10859,10756,10020,10704,11342,10379,11374,10596,10678,10874,9978,9888,10638,9567,9957,10587,11860,11495,10506,10581,10715,10939,9507,11317,10426,10587,11315,10714,11004,10234,10755,11291,10762],
            [10455,10676,10404,9975,10733,9950,9904,10593,10409,10507,9928,10513,10634,11280,10812,11361,10467,9767,10117,10665,9235,10308,11348,11938,11879,11089,10715,10844,11236,9920,11095,10465,10820,11160,10683,10568,9845,10980,11155,9920]
            ] 
    
    data_poisson = [[10656,10396,10737,10313,11464,10313,9714,10934,10106,10829,10166,11011,10219,11289,10333,11206,10310,9994,10058,10698,9140,10072,10882,11559,12129,10450,10801,10863,11292,9491,10992,10056,10589,11103,11070,10808,10031,10691,11444,10157],
            [10445,10386,11360,10231,11340,9581,10373,11255,10653,10514,10350,11084,10800,11937,10933,11336,10669,9753,9843,10955,9657,10045,10909,12089,12071,11225,10809,11141,10799,9558,10981,10781,10941,11306,11172,11149,10395,11358,11015,10470],
            [10745,10287,10978,9802,11368,9859,10488,11147,10770,10412,10057,10866,11113,11495,10467,10770,10564,9594,10098,11089,9340,10496,11097,11687,11642,10739,10389,10841,11189,9568,11374,10673,10681,10655,10998,10730,10047,11122,11221,10461],
            [10317,10546,10928,9911,11028,9793,10056,11166,10506,10750,9927,10770,10592,11343,11077,11277,10508,9884,10156,10381,9586,10294,11197,11297,11965,10582,10535,11090,10881,9504,10837,10508,10659,11170,10852,10862,10083,11109,11145,10117],
            [10479,10420,10853,9867,11173,10059,9824,11152,10568,10326,9909,11117,10598,11023,10039,11371,10527,9877,9905,10897,9336,10083,11206,11372,11766,10192,10657,10561,10771,9780,10906,10686,10551,11115,10504,11317,9955,10739,11120,10533],
            [10290,10425,10765,9971,11044,10057,10227,10711,10473,10317,10278,10544,10398,11047,10703,11208,10843,9669,9773,11291,9133,10290,10925,11592,11417,10687,10270,10971,10754,9638,11256,10320,10350,11165,10928,10987,10028,11133,11023,10005],
            [10449,10455,10397,9783,11230,9583,10155,10836,10642,10466,10320,10511,10305,11180,10801,11350,10644,9827,9979,10620,9383,10184,10792,11938,11546,10358,10240,10800,10745,9456,10725,10015,10684,11105,11014,10658,10101,10671,10917,10324]
            ] 
    
    # Create a figure and axes
    fig, ax = plt.subplots()

    boxplots = ax.boxplot(data_poisson, vert=True, showmeans=True, meanline=True)

    # Set labels and title
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman"})
    ax.set_xticklabels(['Expected demand','1', '10', '100', '500', '1000', '2000'])
    ax.set_ylabel('Failed events',fontsize=13)
    ax.set_xlabel('# scenarios', fontsize=13)
    ax.set_title('Failed events for different number of scenarios', fontsize=15)

    # Display the plot
    
    plt.show()

def different_policies():
    # Define the directory where the CSV files are located
    directory = 'policies/inngjerdingen_moeller/simulation_results/different_policies'

    # Get the list of CSV files in the directory
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

    # Create a color map for the policies
    colors = ['blue', 'green', 'red', 'orange', 'purple']  # Add more colors if needed
    color_map = {file: color for file, color in zip(csv_files, colors)}

    # Create an empty list to store the dataframes for each policy
    dfs = []

    # Read the CSV files and store them in the list
    for file in csv_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path, sep=';', usecols=['Time', 'Failed events'])
        dfs.append(df)

    # Create the line chart
    fig, ax =plt.subplots()
       
    # Plot the data for each policy
    for i, df in enumerate(dfs):
        policy = csv_files[i]
        color = color_map[policy]
        ax.plot(df['Time'], df['Failed events'], color=color, label=policy)

    # # Set a fixed number of ticks on the x-axis
    # num_ticks = 8
    # plt.locator_params(axis='x', nbins=num_ticks)

    # Set the labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Accumulated Number of Failed Events')
    ax.set_title('Failed Events by Policy')
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%a %H:%M"))

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()
    

def different_policies2():
    directory = 'policies/inngjerdingen_moeller/simulation_results/different_policies'

    # Get the list of CSV files in the directory
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

    time_stamps = ["Mon 08:00.000000", "Tue 08:00.000000", "Wed 08:00.000000", "Thu 08:00.000000", "Fri 08:00.000000"]
    # x_datetime_labels = [datetime.strptime(x, "%a %H:%M.%f") for x in time_stamps]

    tick_labels = ["Mon 08:00", "Tue 08:00", "Wed 08:00", "Thu 08:00", "Fri 08:00"]
    colors = ["blue", "red", "green", "purple", "yellow"]

    fig, ax =plt.subplots()
    ax.set_xlabel('Time')
    ax.set_ylabel('Accumulated Number of Failed Events')
    ax.set_title('Failed Events by Policy')

    # Open the CSV file
    for i, csv_file in enumerate(csv_files):
        with open(directory+'/'+csv_file, 'r') as csvfile:
            # Create a CSV reader object
            reader = csv.reader(csvfile, delimiter=";")
            
            times = []
            failures = []
            # Skip the header row if it exists
            header = next(reader, None)

            # Read the data row by row
            for row in reader:
                # Extract data from each column
                times.append(row[0])

                if row[1] == "":
                    failures.append(0)
                else:
                    failures.append(int(row[1]))
            
            # x_datetime = [datetime.strptime(x, "%a %H:%M.%f") for x in times]

            ax.plot(times, failures, color=colors[i], label=csv_file[:-4])

    plt.xticks(time_stamps, labels=tick_labels)

    # ax.set_xticklabels([x.strftime("%a %H:%M.%f") for x in x_datetime_labels])


    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()
            

# roaming_shares()
# solution_times()
# solution_quality()
# branch_number()
# plot_bar_chart()
box_plot()
# different_policies2()

