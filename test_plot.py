import random
import matplotlib.pyplot as plt
import numpy as np

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



roaming_shares()