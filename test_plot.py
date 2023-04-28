import random
import matplotlib.pyplot as plt
import numpy as np

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