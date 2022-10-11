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

def create_and_save_state(name, filename, source, number_of_stations=None, number_of_bikes=None, bike_class="Bike", **kwargs):
    # create initial state
    statedata = { "name" : name }
    statedata.update(source.get_initial_state(**kwargs))

    # create subset of stations
    if number_of_stations is not None:
        create_station_subset(statedata, number_of_stations)

    # override number of bikes
    if number_of_bikes is not None:
        set_num_bikes(statedata, number_of_bikes)

    # save to json
    with gzip.open(f"{filename}.json.gz", "w") as outfile:
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

    statedata["stations"] = subset
