import random
import matplotlib.pyplot as plt
import numpy as np 


#used for branch number
def plot_bar_chart():

    data = [20456,10783,6785,5897,5667,4378,3344,2234,1234,1133]
    total = sum(data)

    fig, ax = plt.subplots()
    ax.bar([i for i in range(1,len(data)+1)], data, color='#2F5597')

    # Add labels and title
    ax.set_xlabel('Branch number', fontsize=12)
    ax.set_ylabel('# scenarios', fontsize=12)
    ax.set_title('Bar Plot with Percentage Values', fontsize=15)
    plt.xticks([i for i in range(1,len(data)+1)])

    # Add percentage values as text over the bars
    for i, val in enumerate(data):
        percentage = (val / total) * 100
        ax.text(i+1, val+100, f"{percentage:.1f}%", ha='center')
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
    title = 'Solution time for various combinations of '+r'$\alpha$'+' and '+ r'$\beta$'
    plt.title(title,fontsize=15)
    
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
    q6 =[2827.1, 2554.4, 2705.7, 2631.8, 2696.3]

    betas1_20 = [1, 3, 5, 7, 9, 11, 13] 
    betas1_10 = [1, 3, 5, 7, 9] 

    # Create the plot
    # plt.yscale("log")
    # max_limit = [10, 10, 10, 10, 10]

    # plt.plot(betas, max_limit, label="max solution time", color='gray', linewidth=2, ls='--')
    plt.plot(betas1_20, q0, label= r'$\alpha=1$', color='#ED7D31', linewidth=2, marker='o')
    plt.plot(betas1_20, q1, label=r'$\alpha=2$', color='#2F5597', linewidth=2, marker='o')
    plt.plot(betas1_10, q2, label=r'$\alpha=3$', color='green', linewidth=2, marker='o')
    plt.plot(betas1_10, q3, label=r'$\alpha=4$', color='purple', linewidth=2, marker='o')
    plt.plot(betas1_10, q4, label=r'$\alpha=5$', color='pink', linewidth=2, marker='o')
    plt.plot(betas1_10, q5, label=r'$\alpha=6$', color='red', linewidth=2, marker='o')
    plt.plot(betas1_10, q6, label=r'$\alpha=7$', color='brown', linewidth=2, marker='o')

    # Add labels and title to the plot
    plt.xlabel(r'$\beta$', fontsize=13)
    plt.ylabel('Failed events', fontsize=13)
    plt.title('Number of failed events for varying PILOT parameters',fontsize=15)
    
    plt.legend()

    plt.grid(axis='y', alpha=0.5)
    plt.yticks(np.arange(2500, 3000, 50))
    plt.xticks([1,3,5,7,10])
    # Show the plot
    plt.show()

# roaming_shares()
# solution_times()
# branch_number()
# plot_bar_chart()
