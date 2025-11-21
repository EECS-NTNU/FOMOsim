# read the file TD_W34.json.gz and extract demand information

import json
import gzip
import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from settings import SB_INSTANCE_FILE

def read_demand():
    # Use repository root when reading the instance so tests can be executed from other
    # folders without breaking path resolution
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    instance_path = os.path.join(repo_root, SB_INSTANCE_FILE + '.json.gz')
    print(f"Reading demand from instance file: {instance_path}")
    
    with gzip.open(instance_path, 'rt') as f:
        data = json.load(f)
    
    demand_info = {}
    
    for station in data['stations']:
        station_id = station['id']
        demand_info[station_id] = {
            'arrive_intensity': station.get('arrive_intensity', {}),
            'leave_intensity': station.get('leave_intensity', {})
        }
    
    return demand_info

if __name__ == "__main__":
    demand = read_demand()
    for station_id, info in demand.items():
        print(f"Station {station_id}:")
        print(f"  Arrive Intensity: {info['arrive_intensity']}")
        print(f"  Leave Intensity: {info['leave_intensity']}")
        
        