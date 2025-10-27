import jsonpickle
import json
import hashlib
import os
import math
import sys
import gzip
import contextily as ctx
import geopy.distance
import geopandas
import rasterio
from rasterio.plot import show as rioshow
from shapely.geometry import box
import matplotlib.pyplot as plt
from PIL import Image as im

import sim
import settings
from helpers import lock, unlock

def createMap(instance_directory, statedata):
    filename = statedata["name"] + ".png"
    filepath = instance_directory + "/" + filename

    # find bounding box of all stations
    east = -180
    west = 180
    north = -180
    south = 180
    for st in statedata["stations"]:
        lat, lon = st["location"]
        if east < lon: east = lon
        if west > lon: west = lon
        if north < lat: north = lat
        if south > lat: south = lat

    # add some space around the outer stations
    north = geopy.distance.distance(meters=500).destination((north, 0), bearing=0)[0]
    south = geopy.distance.distance(meters=500).destination((south, 0), bearing=180)[0]
    east = geopy.distance.distance(meters=500).destination((0, east), bearing=90)[1]
    west = geopy.distance.distance(meters=500).destination((0, west), bearing=270)[1]

    # create map
    img, _ = ctx.bounds2img(west, south, east, north, source=ctx.providers.OpenStreetMap.Mapnik, ll=True)
    data = im.fromarray(img)
    data.save(filepath)

    # add map to statedata
    statedata["map"] = filename
    statedata["map_boundingbox"] = (west, east, south, north)

def get_initial_state(name, source, number_of_stations=None, number_of_bikes=None, city=None, **kwargs):
    if city is None: city = name

    # create initial state
    statedata = {
        "name" : name,
        "city" : city,
    }

    statedata.update(source.get_initial_state(city=city, **kwargs))

    # create subset of stations
    if number_of_stations is not None:
        create_station_subset(statedata, number_of_stations)

    # override number of bikes
    if number_of_bikes is not None:
        set_num_bikes(statedata, number_of_bikes)

    state = sim.State.get_initial_sb_state(statedata)

    return state

def read_initial_state(sb_jsonFilename=None, ff_jsonFilename=None, use_bikes=True, use_escooters=True,
                       number_of_stations=None, number_of_bikes=None):
    if use_bikes:
        if sb_jsonFilename:
            with gzip.open(f"{sb_jsonFilename}.json.gz", "r") as sb_infile:
                dirname = os.path.dirname(sb_jsonFilename);

                # load json state
                sb_statedata = json.load(sb_infile)
        else:
            sb_statedata = None
    else:
        sb_statedata = None
    
    if use_escooters:
        if ff_jsonFilename:
            with gzip.open(f"{ff_jsonFilename}.json.gz", "r") as ff_infile:
                dirname = os.path.dirname(ff_jsonFilename);

                # load json state
                ff_statedata = json.load(ff_infile)
        else:
            ff_statedata = None
    else:
        ff_statedata = None

    # create subset of stations
    if number_of_stations is not None:
        create_station_subset(sb_statedata, number_of_stations)

    # override number of bikes
    if number_of_bikes is not None:
        set_num_bikes(sb_statedata, number_of_bikes)
    
    # set path to map - bilde (TD_W34.png)
    if(sb_statedata is not None):
        if("map" in sb_statedata): sb_statedata["map"] = dirname + "/" + sb_statedata["map"]

    state = sim.State.get_initial_state(sb_statedata, ff_statedata)

    return state

def create_and_save_state(name, instance_directory, source, number_of_stations=None, number_of_bikes=None, city=None, **kwargs):
    if city is None: city = name

    # create initial state
    statedata = {
        "name" : name,
        "city" : city,
    }

    statedata.update(source.get_initial_state(city=city, **kwargs))

    # create subset of stations
    if number_of_stations is not None:
        create_station_subset(statedata, number_of_stations)

    # override number of bikes
    if number_of_bikes is not None:
        set_num_bikes(statedata, number_of_bikes)

    createMap(instance_directory, statedata)

    # save to json
    filename=instance_directory + "/" + name + ".json.gz"
    with gzip.open(filename, "w") as outfile:
        outfile.write(json.dumps(statedata, indent=4).encode())

def set_num_bikes(statedata, n):
    # find total capacity
    total_capacity = 0
    for station in statedata["stations"]:
        total_capacity += station["capacity"]

    # don't place more bikes than capacity
    if n > total_capacity:
        n = total_capacity 

    # place bikes at stations, scaled for capacity
    scale = n / total_capacity
    counter = 0
    for station in statedata["stations"]:
        num_bikes = int(station["capacity"] * scale)
        station["num_bikes"] = num_bikes
        counter += num_bikes

    # place remaining bikes
    biggest_stations = sorted(statedata["stations"], key = lambda station: station["capacity"], reverse=True)
    for i in range(n - counter):
        station = biggest_stations[i % len(biggest_stations)]
        station["num_bikes"] += 1

def create_score(station):
    # make sure depots get high score
    if station["is_depot"]:
        return 1000 + station["depot_capacity"]

    # return average of leave intensity
    leave = 0
    for day in range(7):
        for hour in range(24):
            leave += station["leave_intensities"][day][hour]
    leave /= 7*24
    return leave

def create_station_subset(statedata, n):
    subset = sorted(statedata["stations"], key=create_score, reverse=True)[:n]

    for sid, station in enumerate(subset):
        for day in range(7):
            for hour in range(24):
                # move probabilities subset
                move_prob = []
                for dest in subset:
                    move_prob.append(station["move_probabilities"][day][hour][dest["id"]])
                station["move_probabilities"][day][hour] = move_prob

                # scale move probabilities
                subset_prob = 0
                for prob in station["move_probabilities"][day][hour]:
                    subset_prob += prob
                if subset_prob == 0:
                    for i in range(len(subset)):
                        station["move_probabilities"][day][hour][i] = 1 / len(subset)
                else:
                    for i in range(len(subset)):
                        station["move_probabilities"][day][hour][i] /= subset_prob

    # calculate arrive intensity (should match leave intensities)
    for day in range(7):
        for hour in range(24):
            for station_id,station in enumerate(subset):
                incoming = 0
                for from_station in subset:
                    incoming += from_station["leave_intensities"][day][hour] * from_station["move_probabilities"][day][hour][station_id]
                station["arrive_intensities"][day][hour] = incoming
                station["arrive_intensities_stdev"][day][hour] = 0

    statedata["stations"] = subset
