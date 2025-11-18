# read the file TD_W34.json.gz and extract demand information

import json
import gzip
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from settings import SB_INSTANCE_FILE

def read_demand():
    instance_path = os.path.join(os.path.dirname(__file__), '..', '..', SB_INSTANCE_FILE + '.json.gz')
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
        
        