import datetime

from google.cloud import bigquery as bq
import pickle

import sim
import settings

def setup_stations_students(clientName, init_hour, number_of_vehicles, random_seed, pull_data_server=False):
    """
    function used to setup stations for students without access to all BQ data
    """

    if pull_data_server:
        client = bq.Client(clientName)
    else:
        client = None

    demand_query = "SELECT * FROM `uip-students.loaded_data.simulation_demand_prediction`"
    snapshot_query = "SELECT * FROM `uip-students.loaded_data.simulation_dockgroup_snapshots`"
    dockgroup_movement_query = "SELECT * FROM `uip-students.loaded_data.simulation_station_movement_info`"
    driving_time_query = "SELECT * FROM `uip-students.loaded_data.simulation_driving_times`"
    coordinate_query = "SELECT DISTINCT dock_group_id, dock_group_coords.latitude, dock_group_coords.longitude FROM `uip-students.loaded_data.stations_snapshots`"

    save_data = False

    if pull_data_server: #pull from server
        coordinates = get_data_from_bq(coordinate_query, client)
        demand_df = get_data_from_bq(demand_query, client)
        snapshot_df = get_data_from_bq(snapshot_query, client)
        movement_df = get_data_from_bq(dockgroup_movement_query, client)
        car_movement_df = get_data_from_bq(driving_time_query, client)
        if save_data:
            all_data_bigquery = [coordinates,demand_df,snapshot_df,movement_df,car_movement_df]
            dbfile = open("Input//AllDataBigQuery", "ab")   #Or put AllDataBigQuery
            pickle.dump(all_data_bigquery, dbfile)                     
            dbfile.close()
        #
        #all_stations = pickle.load(data_file)
        #len(all_stations) #237
    else:
        data_file = open("init_state//fosen_haldorsen//AllDataBigQuery", "rb")    #open("AllDataBigQuery", "rb")   
        all_data_bigquery = pickle.load(data_file)
        coordinates = all_data_bigquery[0]
        demand_df = all_data_bigquery[1]
        snapshot_df = all_data_bigquery[2]
        movement_df = all_data_bigquery[3]
        car_movement_df = all_data_bigquery[4]

    datestring = "2019-09-17"
    coordinate_dict = get_input_data_from_coordinate_df(coordinates)
    snapshot_input = get_input_data_from_snapshot_df(snapshot_df, datestring)
    movement_input = get_input_data_from_movement_df(movement_df, datestring, snapshot_keys=snapshot_input.keys())
    demand_input = get_input_data_from_demand_df(demand_df, snapshot_keys=snapshot_input.keys())
    car_movement_input = get_input_data_from_car_movement_df(car_movement_df,
            snapshot_keys=snapshot_input.keys())

    # search for missing station_ids and add them to movement_input
    snap_keys = list(snapshot_input.keys())
    car_missing = set(snap_keys).difference(set(car_movement_input.keys()))

    for missing_station_id in car_missing:
        car_movement_input[missing_station_id] = {id: 3*time for id, time in movement_input[missing_station_id]["avg_trip_duration"].items()}

    ###############################################################################

    traveltime_matrix = []
    traveltime_vehicle_matrix = []
    number_of_scooters = []
    capacities = []

    arrive_intensities = []
    leave_intensities = []
    move_probabilities = []

    original_ids = []
    charging_stations = []

    stations_to_include = []

    for station_id in snapshot_input.keys():
        add = True
        for value in car_movement_input[station_id].values():
            if value > 100:
                add = False

        if add:
            stations_to_include.append(station_id)

    num_stations = len(stations_to_include)

    toid = {}
    for i, stationid in enumerate(stations_to_include):
        toid[i] = stationid

    for i in range(num_stations):
        station_id = toid[i]
        next_station_probabilities = movement_input[station_id]["movement_probabilities"]
        station_travel_time = movement_input[station_id]["avg_trip_duration"]
        max_capacity = snapshot_input[station_id]["max_capacity"]
        demand_per_hour = demand_input[station_id] if station_id in demand_input else {i: 0 for i in range(6, 24)}
        actual_num_bikes = snapshot_input[station_id]["bikes"]
        station_car_travel_time = car_movement_input[station_id]

        move_p = [next_station_probabilities[toid[i]] for i in range(num_stations)]

        if (int(station_id) % 10) == 0:
            charging_stations.append(i)

        traveltime_matrix.append([ station_car_travel_time.get(toid[j], 0) * 1.3 for j in range(num_stations) ])
        traveltime_vehicle_matrix.append([ station_car_travel_time.get(toid[j], 0) for j in range(num_stations) ])

        number_of_scooters.append(actual_num_bikes[init_hour])
        capacities.append(max_capacity)

        original_ids.append(station_id)

        arrive_intensities.append([])
        leave_intensities.append([])
        move_probabilities.append([])
        for day in range(7):
            arrive_intensities[i].append([])
            leave_intensities[i].append([])
            move_probabilities[i].append([])
            for hour in range(24):
                arrive_intensities[i][day].append(0) # fixed in preprocess.py
                leave_intensities[i][day].append(demand_per_hour[hour])
                move_probabilities[i][day].append(move_p)

    ###############################################################################

    stations = sim.State.create_stations(num_stations=len(capacities), capacities=capacities, charging_stations=charging_stations, original_ids=original_ids, depots=[4])
    sim.State.create_bikes_in_stations(stations, "Scooter", number_of_scooters)
    sim.State.set_customer_behaviour(stations, leave_intensities, arrive_intensities, move_probabilities)
    return sim.State.get_initial_state(stations, number_of_vehicles, random_seed, traveltime_matrix, traveltime_vehicle_matrix)


def get_input_data_from_movement_df(movement_df, datestring, snapshot_keys):
    """
    Returns a dictionary with data on the form:
    "dock_group_id": {
        "movement_probabilities: {end_dock_id: prob (float)},
        "avg_trip_duration": {end_dock_id: avg_trip_dur_seconds},
        }
    """

    data = dict()
    mf = movement_df
    # remove stations that are not in dockgroup snapshot query
    mf = mf[(mf.start_dock_id.isin(snapshot_keys)) & (mf.end_dock_id.isin(snapshot_keys))]

    g = mf.groupby("start_dock_id")
    all_dock_ids = [data[0] for data in g]

    for info in g:
        id = info[0]
        df = info[1].copy()

        total_trips = df.num_trips.sum()
        df["move_probability"] = df.num_trips/total_trips

        move_prob = dict(zip(df.end_dock_id, df.move_probability))
        avg_trip_duration = dict(zip(df.end_dock_id, df.avg_duration_in_seconds))

        # add 0s for stations that havent been visited
        for dock_id in all_dock_ids:
            if dock_id not in move_prob:
                move_prob[dock_id] = 0
                avg_trip_duration[dock_id] = 60*60

        dock_info = dict()
        dock_info["movement_probabilities"] = move_prob
        dock_info["avg_trip_duration"] = avg_trip_duration

        data[id] = dock_info

    return data


def get_input_data_from_snapshot_df(df, datestring):
    """
    Returns a dictionary with data on the form:
    "dock_group_id": {
        "num_bikes_list": {datetime: num_bikes (int)},
        "max_capacity": num_docks (int),
        "dock_group_title": string,
        }
    """
    date = datetime.datetime.strptime(datestring, "%Y-%m-%d")

    #df = snapshot_df
    # df = df[["hour", "minute", "current_num_bikes", "dock_group_id"]]
    df["date"] = df.apply(lambda row:
        date + datetime.timedelta(minutes = row.minute, hours = row.hour), axis = 1)

    data = dict()
    g = df.groupby("dock_group_id") #[["date", "current_num_bikes"]]

    for dock_id in g:

        id = dock_id[0]
        df2 = dock_id[1]

        dock_info = dict()
        dock_info["num_bikes_list"] = dict(zip(df2.date, df2.current_num_bikes))
        dock_info["max_capacity"] = df2.total_num_docks.mean()   #mean??
        dock_info["dock_group_title"] = df2.iloc[0].dock_group_title
        dock_info["bikes"] = dict(zip(df2.hour, df2.current_num_bikes))

        data[id] = dock_info

    return data



def get_input_data_from_coordinate_df(df):
    data = dict()
    for index, row in df.iterrows():
        data[row[0]] = [row[1], row[2]]
    return data


def get_input_data_from_demand_df(demand_df, snapshot_keys):
    # snapshot_keys = [float(val) for val in snapshot_keys]
    data = dict()

    demand_df = demand_df[demand_df["station_id"].isin(snapshot_keys)]
    for station_id, df in demand_df.groupby("station_id"):
        station_data = dict(zip(df.hour, df.bike_demand_per_hour))

        data[station_id] = station_data

    return data


def get_input_data_from_car_movement_df(car_movement_df, snapshot_keys):
    car_movement_df = car_movement_df[car_movement_df["start_station_id"].isin(snapshot_keys)]
    data = dict()

    for station_id, df in car_movement_df.groupby("start_station_id"):
        station_data = dict(zip(df.end_station_id, df.driving_time))  # Removed sec conversion
        data[station_id] = station_data

    return data
