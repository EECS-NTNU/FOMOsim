"""
Add a central depot to TD_W34_old.json.gz instance file
"""
import gzip
import json
import numpy as np

# Load the instance
print("Loading TD_W34_old.json.gz...")
with gzip.open('instances/TD_W34_old.json.gz', 'rt') as f:
    data = json.load(f)

# Extract station locations
stations = data['stations']
locations = np.array([[s['location'][0], s['location'][1]] for s in stations])

# Calculate geographic center
center_lat = np.mean(locations[:, 0])
center_lon = np.mean(locations[:, 1])
print(f"Station location range:")
print(f"  lat [{locations[:, 0].min():.4f}, {locations[:, 0].max():.4f}]")
print(f"  lon [{locations[:, 1].min():.4f}, {locations[:, 1].max():.4f}]")
print(f"Center: [{center_lat:.6f}, {center_lon:.6f}]")

# Create a depot at the center
depot = {
    "id": len(stations),  # next ID after all stations (67)
    "location": [center_lat, center_lon],
    "is_depot": True,
    "capacity": 100,  # large repair capacity
    "num_bikes": 0,   # starts empty
    "leave_intensities": [[0.0] * 24 for _ in range(7)],
    "arrive_intensities": [[0.0] * 24 for _ in range(7)],
    "leave_intensities_stdev": [[0.0] * 24 for _ in range(7)],
    "arrive_intensities_stdev": [[0.0] * 24 for _ in range(7)],
}

# Add depot to stations
data['stations'].append(depot)

print(f"\nAdded depot:")
print(f"  ID: {depot['id']}")
print(f"  Location: {depot['location']}")
print(f"  Capacity: {depot['capacity']}")
print(f"  Total stations now: {len(data['stations'])}")

# Save back to file
print("Saving to instances/TD_W34_old.json.gz...")
with gzip.open('instances/TD_W34_old.json.gz', 'wt') as f:
    json.dump(data, f, indent=2)

print("✓ Done! Depot added successfully.")
