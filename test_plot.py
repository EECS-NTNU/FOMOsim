import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np 
import pandas as pd
import os
from output.visualizer import totime
import csv
from datetime import datetime, timedelta
from dateutil.parser import parse

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
    q0 = [2876.8, 2618.8, 2653.7, 2673.1, 2635, 2891.5, 2906.6]
    q1 =[2865.7, 2752.9, 2660.8, 2717.9, 2732.7, 2861.4, 2894.5]
    q2 =[2834.3, 2600.1, 2529.5, 2542.1, 2600]
    q3 = [2865.5, 2635.3, 2566.1, 2678.5, 2711.3]
    q4 =[2881.7, 2582.4, 2607.6, 2698.8, 2753]
    q5 = [2884.1, 2592.2, 2603.1, 2697.5, 2829.5]
    q6 =[2841.3, 2612.2, 2641, 2680.2, 2706.1]

    betas1_20 = [1, 3, 5, 7, 9, 11, 13] 
    betas1_10 = [1, 3, 5, 7, 9] 

    # Create the plot
    # plt.yscale("log")
    # max_limit = [10, 10, 10, 10, 10]

    # plt.plot(betas, max_limit, label="max solution time", color='gray', linewidth=2, ls='--')
    plt.plot(betas1_20, q0, label= r'$\alpha=1$', color='#ED7D31', linewidth=1.5, marker='o')
    plt.plot(betas1_20, q1, label=r'$\alpha=2$', color='#2F5597', linewidth=1.5, marker='o')
    plt.plot(betas1_10, q2, label=r'$\alpha=3$', color='green', linewidth=1.5, marker='o')
    plt.plot(betas1_10, q3, label=r'$\alpha=4$', color='purple', linewidth=1.5, marker='o')
    plt.plot(betas1_10, q4, label=r'$\alpha=5$', color='pink', linewidth=1.5, marker='o')
    plt.plot(betas1_10, q5, label=r'$\alpha=6$', color='red', linewidth=1.5, marker='o')
    plt.plot(betas1_10, q6, label=r'$\alpha=7$', color='brown', linewidth=1.5, marker='o')

    # Add labels and title to the plot
    plt.xlabel(r'$\beta$', fontsize=13)
    plt.ylabel('Failed events', fontsize=13)
    plt.title('Number of failed events for various combinations of '+r'$\alpha$'+' and '+ r'$\beta$',fontsize=15)
    plt.legend()

    plt.grid(axis='y', alpha=0.5)
    plt.yticks(np.arange(2500, 3000, 50))
    plt.xticks([1,3,5,7])
    # Show the plot
    plt.show()

def box_plot():
    data_normal = [[2797,2520,2589,2736,2935,2548,2405,2470,2355,2429],
            [2734,2503,2713,2621,2681,2563,2469,2565,2812,2276],
            [2800,2483,2326,2632,2523,2568,2568,2501,2603,2480],
            [2703,2497,2386,2533,2654,2352,2196,2583,2441,2270],
            [2644,2617,2669,2370,2684,2665,2299,2307,2495,2251],
            [2656,2706,2364,2451,2313,2641,2129,2242,2434,2288]
            ] 
    
    data_poisson = [[2645,2506,2551,2687,2632,2492,2353,2318,2573,2230],
            [2545,2453,2341,2572,2557,2695,2492,2382,2478,2392],
            [2740,2369,2248,2681,2364,2599,2215,2308,2296,2330],
            [2637,2427,2609,2639,2566,2610,2360,2407,2440,2275],
            [2676,2505,2574,2460,2531,2511,2572,2165,2593,2201],
            [2388,2333,2534,2661,2599,2628,2143,2513,2644,2168]
            ] 
    
    # Create a figure and axes
    fig, ax = plt.subplots()

    boxplots = ax.boxplot(data_poisson, vert=True, showmeans=True, meanline=True)

    # Set labels and title
    ax.set_xticklabels(['1', '10', '100', '500', '1000', '2000'])
    ax.set_ylabel('Failed events')
    ax.set_xlabel('# scenarios')
    ax.set_title('Boxplot')

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

    time_stamps = ["Mon 07:00.000000", "Tue 07:00.000000", "Wed 07:00.000000", "Thu 07:00.000000", "Fri 07:00.000000", "Sat 07:00.000000", "Sun 07:00.000000", "Mon 07:00.000000", "Tue 07:00.000000", "Wed 07:00.000000", "Thu 07:00.000000"]
    x_datetime_labels = [parse(x) for x in time_stamps]

    # tick_labels = ["Mon 07:00", "Tue 07:00", "Wed 07:00", "Thu 07:00", "Fri 07:00", "Sat 07:00", "Sun 07:00"]
    colors = ["blue", "red", "green", "purple", "yellow"]

    fig, ax =plt.subplots()
    ax.set_xlabel('Time', fontweight='bold')
    ax.set_ylabel('Accumulated Failed Events', fontweight='bold')
    ax.set_title('Failed Events by Policy', fontweight='bold')

    current_year = 2023
    current_month = 5
    current_day = 22

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
            
            x_datetime = [parse(x, default=datetime(current_year, current_month, current_day)) for x in times]
            x_datetime_new = []
            new_week = False
            current_weekday = x_datetime[0].weekday()
            for dt in x_datetime:
                if new_week == True:
                    dt += timedelta(days=7)
                elif dt.weekday() < current_weekday:
                    new_week = True
                    dt += timedelta(days=7)
                x_datetime_new.append(dt)
                current_weekday = dt.weekday()

            ax.plot(x_datetime_new, failures, color=colors[i], label=csv_file[:-4])

    # plt.xticks(time_stamps, labels=tick_labels)

    ax.set_xticklabels([x.strftime("%a") for x in x_datetime_labels])
    ax.tick_params(axis='x', color='lightgray')
    ax.tick_params(axis='y', color='lightgray')

    # Add a legend
    ax.legend()
    ax.yaxis.grid(True, color='lightgray', linewidth=0.5)

    # Show the plot
    plt.show()
            

# roaming_shares()
# solution_times()
# branch_number()
# plot_bar_chart()
# box_plot()
different_policies2()

