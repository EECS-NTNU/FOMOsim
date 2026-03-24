import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the bike movements CSV
csv_path = "policies/sjovik_sund/simulation_results/csv/DoNothing_baseline_TD_W34_old_V1_D168h_03241609_seed1_results_bike_movements_seed_1.csv"
df = pd.read_csv(csv_path)

# Each row represents one trip
# Aggregate demand per hour per day
demand_per_hour = df.groupby(['Day', 'Hour']).size()

# Create continuous time axis (hour 0-167)
hours = []
demands = []
for day in range(7):
    for hour in range(24):
        if (day, hour) in demand_per_hour.index:
            hours.append(day * 24 + hour)
            demands.append(demand_per_hour[(day, hour)])

# Map day numbers to day names
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Plot
plt.figure(figsize=(16, 6))
plt.plot(hours, demands, marker='o', linewidth=2, markersize=4)
plt.xlabel('Hour (0-168)')
plt.ylabel('Number of Trips')
plt.title('Demand Pattern Across the Week (Monday - Sunday)')
plt.grid(True, alpha=0.3)

# Add vertical lines and labels for day boundaries
for day in range(1, 7):
    plt.axvline(x=day * 24, color='red', linestyle='--', alpha=0.3)
    
# Add day labels at the center of each day
for day in range(7):
    plt.text(day * 24 + 12, plt.ylim()[1] * 0.95, day_names[day], ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('policies/sjovik_sund/output/demand_per_hour.png', dpi=150)
plt.show()

print(f"Demand data points: {len(demands)}")
