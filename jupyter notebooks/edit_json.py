###### Make ebike .json-files

import os
import gzip
import shutil
import json


WITH_DEPOT = True
instances_path = os.getcwd() + "/instances"
ebike_folder = instances_path + "/ebike" + ("_with_depot" if WITH_DEPOT else "")
    
def fix_json(filename):
    with gzip.open(instances_path + '/' + filename, 'rt', encoding='utf-8') as gz_file:
        data = json.loads(gz_file.read())

        if 'bike_class' in data:
            data['bike_class'] = 'EBike'

        for index, station in enumerate(data.get('stations', [])):
            # Set capacity to 'inf' for all stations
            station['capacity'] = 'inf'
        
            # Change 'is_depot' to true only for the first station
            if index == 0 and WITH_DEPOT:
                station['is_depot'] = True

    with gzip.open(ebike_folder + '/' + filename, 'wt', encoding='utf-8') as gz_file:
        json.dump(data, gz_file, indent=4)


def move_png(filename):
    shutil.copy(instances_path + '/' + filename, ebike_folder + '/' + filename)


def move_png(filename):
    shutil.copy(instances_path + '/' + filename, ebike_folder + '/' + filename)


if __name__ == "__main__":
    for filename in os.listdir(instances_path):
        if filename.endswith('.json.gz'):
            fix_json(filename)
        elif filename.endswith('.png'):
            move_png(filename)