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
    times_0 = [0.246, 0.380, 0.489, 0.616, 0.808]
    times_1 =[0.262, 0.566, 0.810, 3.309, 5.696]
    times_2 =[0.280, 0.646, 0.948, 5.094, 8.806]
    times_3 = [0.302, 0.777, 1.106, 6.362, 11.458]
    times_4 =[0.306, 0.828, 1.218, 7.618, 13.500]
    times_5 = [0.321, 0.912, 1.366, 8.973, 17.569]
    times_6 =[0.343, 1.182, 2.415, 10.651, 18.265]

    betas = [1, 3, 5, 7, 10] 

    # Create the plot
    # plt.yscale("log")
    max_limit = [10, 10, 10, 10, 10]

    plt.plot(betas, max_limit, label="max solution time", color='gray', linewidth=2, ls='--')
    plt.plot(betas, times_0, label= r'$\alpha=1$', color='#ED7D31', linewidth=2, marker='o')
    plt.plot(betas, times_1, label=r'$\alpha=2$', color='#2F5597', linewidth=2, marker='o')
    plt.plot(betas, times_2, label=r'$\alpha=3$', color='green', linewidth=2, marker='o')
    plt.plot(betas, times_3, label=r'$\alpha=4$', color='purple', linewidth=2, marker='o')
    plt.plot(betas, times_4, label=r'$\alpha=5$', color='pink', linewidth=2, marker='o')
    plt.plot(betas, times_5, label=r'$\alpha=6$', color='red', linewidth=2, marker='o')
    plt.plot(betas, times_6, label=r'$\alpha=7$', color='brown', linewidth=2, marker='o')

    # Add labels and title to the plot
    plt.xlabel(r'$\beta$', fontsize=13)
    plt.ylabel('Solution time (s)', fontsize=13)
    plt.title('Solution time for varying PILOT parameters',fontsize=15)
    
    plt.legend()

    plt.grid(axis='y', alpha=0.5)
    plt.yticks(np.arange(0, 19, 2))
    plt.xticks([1,3,5,7,10])
    # Show the plot
    plt.show()

# roaming_shares()
# solution_times()
# branch_number()
plot_bar_chart()