import os
import json
import gzip

# 1. Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
instances_dir = os.path.abspath(os.path.join(script_dir, '../../../instances'))

# Source filename
input_gz_filename = 'TD_W34_old.json.gz'
input_path = os.path.join(instances_dir, input_gz_filename)

# Output filenames
output_json_path = os.path.join(script_dir, 'TD_W34_67.json')
output_gz_path = os.path.join(script_dir, 'TD_W34_67.json.gz')

# 2. Read and Modify
if not os.path.exists(input_path):
    print(f"Error: Could not find file at: {input_path}")
else:
    print(f"Reading from: {input_path}")
    
    with gzip.open(input_path, 'rt', encoding='utf-8') as f_in:
        data = json.load(f_in)

    print("Modifying 'num_bikes'...")
    
    # Loop through all stations and update num_bikes
    for station in data['stations']:
        # Get capacity, default to 0 if missing
        capacity = station.get('capacity', 0)
        
        # Calculate half capacity (integer division)
        new_bikes = int(capacity / 2)
        
        # Update the value
        station['num_bikes'] = new_bikes

    # 3. Save as normal .json
    print(f"Saving to .json: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, indent=4)

    # 4. Save as compressed .json.gz
    print(f"Saving to .json.gz: {output_gz_path}")
    with gzip.open(output_gz_path, 'wt', encoding='utf-8') as f_out_gz:
        json.dump(data, f_out_gz)

    print("\nSuccess! Both files created.")