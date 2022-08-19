import jsonpickle
import hashlib
import os

import sim
import settings

savedStatesDirectory = "saved_states/"

def get_initial_state(source, target_state=None, number_of_stations=None, number_of_bikes=None, load_from_cache=True, **kwargs):
    # create filename
    all_args = {"source" : source, "target_state" : target_state, "number_of_stations" : number_of_stations, "number_of_bikes" : number_of_bikes}
    all_args.update(kwargs)
    checksum = hashlib.sha256(jsonpickle.encode(all_args).encode('utf-8')).hexdigest()
    stateFilename = f"{savedStatesDirectory}/{checksum}.pickle.gz"

    # if exists, load from cache
    if load_from_cache:
        if os.path.isdir(savedStatesDirectory):
            # directory with saved states exists
            if os.path.isfile(stateFilename):
                print("Loading state from file")
                state = sim.State.load(stateFilename)
                return state

    # create initial state
    state = source.get_initial_state(**kwargs)

    # create subset of stations
    if number_of_stations is not None:
        create_subset(state, number_of_stations)

    if number_of_bikes is not None:
        adjust_num_bikes(state, number_of_bikes)

    # calculate target state
    if target_state is not None:
        tstate = target_state(state)
        state.set_target_state(tstate)

    # save to cache
    if not os.path.isdir(savedStatesDirectory):
        os.makedirs(savedStatesDirectory, exist_ok=True) # first time
    print("Saving state to file")
    state.save(stateFilename)

    return state

def adjust_num_bikes(state, n):
    stations = sorted(state.locations, key = lambda station: len(station.get_bikes()), reverse=True)
    statCounter = 0

    while(len(state.get_all_bikes()) > n):
        # first, remove from bikes in use
        if not settings.FULL_TRIP and len(state.bikes_in_use) > 0:
            bike_id = list(state.bikes_in_use.values())[0].id
            state.remove_used_bike(bike_id)
        else:
            # then, remove from stations
            if len(stations[statCounter].get_bikes()) > 0:
                bike_id = list(stations[statCounter].get_bikes())[0].id
                del stations[statCounter].bikes[bike_id]

            statCounter += 1
            if statCounter >= len(stations):
                statCounter = 0

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

def create_subset(state, n):
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
