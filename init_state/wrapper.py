import jsonpickle
import json
import hashlib
import os
import math
import sys
import gzip

import sim
import settings
from helpers import lock, unlock

savedStatesDirectory = "saved_states"

def read_initial_state(jsonFilename, target_state=None, load_from_cache=True):
    # create filename
    all_args = {"target_state" : target_state, "jsonFilename" : jsonFilename}
    checksum = hashlib.sha256(jsonpickle.encode(all_args).encode('utf-8')).hexdigest()
    stateFilename = f"{savedStatesDirectory}/{checksum}.pickle.gz"

    if not os.path.isdir(savedStatesDirectory):
        os.makedirs(savedStatesDirectory, exist_ok=True)

    lock_handle = lock(stateFilename)

    if load_from_cache:
        if os.path.isfile(stateFilename):
            # load from cache
            print("Loading state from file")
            state = sim.State.load(stateFilename)
            unlock(lock_handle)
            return state

    with gzip.open(f"{jsonFilename}.json.gz", "r") as infile:
        # load json state
        statedata = json.load(infile)
        state = sim.State.get_initial_state(statedata)

        # calculate target state
        if target_state is not None:
            tstate = target_state(state)
            state.set_target_state(tstate)

        # save to cache
        print("Saving state to file")
        state.save(stateFilename)

        unlock(lock_handle)
        return state

    return None

def create_and_save_state(filename, source, number_of_stations=None, number_of_bikes=None, bike_class="Bike", **kwargs):
    # create initial state
    statedata = source.get_initial_state(**kwargs)

    # create subset of stations
    if number_of_stations is not None:
        create_station_subset(statedata, number_of_stations)

    # override number of bikes
    if number_of_bikes is not None:
        set_num_bikes(statedata, number_of_bikes, bike_class)

    # save to json
    with gzip.open(f"{filename}.json.gz", "w") as outfile:
        outfile.write(json.dumps(statedata, indent=4).encode())

def set_num_bikes(statedata, n):
    # find total capacity
    total_capacity = 0
    for cap in statedata["capacities"]:
        total_capacity += cap

    # don't place more bikes than capacity
    if n > total_capacity:
        n = total_capacity

    # place bikes at stations, scaled for capacity
    scale = n / total_capacity
    counter = 0
    for i in range(statedata["num_stations"]):
        add = int(statedata["capacities"] * scale)
        statedata["bikes_per_station"][i] = add
        counter += add

    # place remaining bikes
    for i in range(n - counter):
        station = i % statedata["num_stations"]
        statedata["bikes_per_station"][i] += 1

def create_score(leave_intensities):
    # return average of leave intensity
    leave = 0
    for day in range(7):
        for hour in range(24):
            leave += leave_intensities[day][hour]
    leave /= 7*24
    return leave

def create_station_subset(statedata, n):
    station_ids = list(range(statedata["num_stations"]))
    station_score = [ create_score(x) for x in statedata["leave_intensities"] ]
    for depot in statedata["depots"]:
        station_score[depot] = 1000

    zipped = sorted(zip(station_ids, station_score), key=lambda y : y[1], reverse=True)[:n]
    subset = [ x[0] for x in zipped ]

    for station_id in subset:
        for day in range(7):
            for hour in range(24):
                # move probabilities subset
                move_prob = []
                subset_prob = 0
                for dest_id in subset:
                    prob = stationdata["move_probabilities"][day][hour][dest.id]
                    move_prob.append(prob)
                    subset_prob += prob
                statedata[station_id]["move_probabilities"][day][hour] = move_prob

                # scale move probabilities
                for i in range(len(subset)):
                    if subset_prob == 0:
                        statedata[station_id]["move_probabilities"][day][hour][i] = 1 / len(subset)
                    else:
                        statedata[station_id]["move_probabilities"][day][hour][i] /= subset_prob

    # calculate arrive intensity (should match leave intensities)
    for day in range(7):
        for hour in range(24):
            for station_id in subset:
                incoming = 0
                for from_station_id in subset:
                    incoming += statedata[from_station_id]["leave_intensities"][day][hour] * statedata[from_station_id]["move_probabilities"][day][hour][station.id]
                statedata[station_id]["arrive_intensity"][day][hour] = incoming
