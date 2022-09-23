import jsonpickle
import hashlib
import os
import math
import sys

import sim
import settings
from helpers import lock, unlock

savedStatesDirectory = "saved_states"

def get_initial_state(source, target_state=None, number_of_stations=None, number_of_bikes=None, bike_class="Bike", load_from_cache=True, **kwargs):
    if not os.path.isdir(savedStatesDirectory):
        os.makedirs(savedStatesDirectory, exist_ok=True)

    # create filename
    all_args = {"source" : source, "target_state" : target_state, "number_of_stations" : number_of_stations, "number_of_bikes" : number_of_bikes, "bike_class" : bike_class}
    all_args.update(kwargs)
    checksum = hashlib.sha256(jsonpickle.encode(all_args).encode('utf-8')).hexdigest()
    stateFilename = f"{savedStatesDirectory}/{checksum}.pickle.gz"

    # if exists, load from cache
    print("Waiting for lock")
    lock_handle = lock(stateFilename)
    print("Got lock")

    if load_from_cache:
        if os.path.isdir(savedStatesDirectory):
            # directory with saved states exists
            if os.path.isfile(stateFilename):
                print("Loading state from file")
                state = sim.State.load(stateFilename)
                unlock(lock_handle)
                return state

    # create initial state
    state = source.get_initial_state(**kwargs)

    # create subset of stations
    if number_of_stations is not None:
        create_station_subset(state, number_of_stations)

    # override number of bikes
    if number_of_bikes is not None:
        set_num_bikes(state, number_of_bikes, bike_class)

    # calculate target state
    if target_state is not None:
        tstate = target_state(state)
        state.set_target_state(tstate)

    # save to cache
    print("Saving state to file")
    state.save(stateFilename)

    unlock(lock_handle)

    return state

def set_num_bikes(state, n, bike_class):
    # find total capacity
    total_capacity = 0
    for station in state.locations:
        total_capacity += station.capacity

    # don't place more bikes than capacity
    if n > total_capacity:
        n = total_capacity

    # create and place bikes at stations, scaled for capacity
    id_counter = 0
    scale = n / total_capacity
    for station in state.locations:
        bikes = []
        for bike_id in range(int(station.capacity * scale)):
            if bike_class == "EBike":
                bikes.append(sim.EBike(bike_id=id_counter, battery=100))
            else:
                bikes.append(sim.Bike(bike_id=id_counter))
            id_counter += 1
        station.set_bikes(bikes)

    # place remaining bikes
    stations = sorted(state.locations, key = lambda station: station.capacity, reverse=True)
    for i in range(n - id_counter):
        station = stations[i % len(stations)]
        if bike_class == "EBike":
            station.add_bike(state.rng, sim.EBike(bike_id=id_counter, battery=100))
        else:
            station.add_bike(state.rng, sim.Bike(bike_id=id_counter))
        id_counter += 1

    # remove bikes in use
    state.bikes_in_use = {}

    # remove bikes in vehicles
    for vehicle in state.vehicles:
        vehicle.bike_inventory = {}

def station_sort(station):
    # make sure depots get high score
    if isinstance(station, sim.Depot):
        return 1000 + station.depot_capacity

    # return average of leave intensity
    leave = 0
    for day in range(7):
        for hour in range(24):
            leave += station.get_leave_intensity(day, hour)
    leave /= 7*24
    return leave

def create_station_subset(state, n):
    subset = sorted(state.locations, key=station_sort, reverse=True)[:n]

    for sid, station in enumerate(subset):
        for day in range(7):
            for hour in range(24):
                # move probabilities subset
                move_prob = []
                for dest in subset:
                    move_prob.append(station.move_probabilities[day][hour][dest.id])
                station.move_probabilities[day][hour] = move_prob

                # scale move probabilities
                subset_prob = 0
                for prob in station.move_probabilities[day][hour]:
                    subset_prob += prob
                if subset_prob == 0:
                    for i in range(len(subset)):
                        station.move_probabilities[day][hour][i] = 1 / len(subset)
                else:
                    for i in range(len(subset)):
                        station.move_probabilities[day][hour][i] /= subset_prob

        station.id = sid

    # calculate arrive intensity (should match leave intensities)
    for day in range(7):
        for hour in range(24):
            for station in subset:
                incoming = 0
                for from_station in subset:
                    incoming += from_station.leave_intensity_per_iteration[day][hour] * from_station.move_probabilities[day][hour][station.id]
                station.arrive_intensity_per_iteration[day][hour] = incoming

    state.set_locations(subset)
