"""
Main file for clustering module
"""
from sim import Vehicle, State, Bike, EBike
import init_state.entur.methods as methods
import os
from settings import *
import pandas as pd
import numpy as np

def scooter_sample_filter(rng, dataframe: pd.DataFrame, number_of_scooters=None):
    if number_of_scooters:
        dataframe = dataframe.sample(number_of_scooters, random_state=rng.integers(0, 1000))
    return dataframe["id"].tolist()


def get_initial_state(
    entur_data_dir,
    entur_main_file,
    number_of_bikes=None,
    initial_in_use=0,
    number_of_clusters=20,
    initial_location_depot=True,
    number_of_vehicles=4,
    random_seed=None,
) -> State:
    """
    Main method for setting up a state object based on EnTur data from test_data directory.
    This method saves the state objects and reuse them if a function call is done with the same properties.
    :param number_of_bikes: number of e-scooters in the instance
    :param number_of_clusters
    :param initial_location_depot: set the initial current location of the vehicles to the depot
    :param number_of_vehicles: the number of vehicles to use
    :return:
    """
    # If this combination has been requested before we fetch a cached version
    print(
        f"\nSetup initial state from entur dataset with {number_of_clusters} clusters and {number_of_bikes} scooters"
    )
    # Get dataframe from EnTur CSV file within boundary
    entur_dataframe = methods.read_bounded_csv_file(entur_data_dir + "/" + entur_main_file)

    rng=np.random.default_rng(random_seed)

    # Create clusters
    cluster_labels = methods.cluster_data(rng, entur_dataframe, number_of_clusters)

    # generate depots and adding them to clusters list
    depots = methods.generate_depots()

    # Structure data into objects
    clusters = methods.generate_cluster_objects("EBike", entur_dataframe, cluster_labels, number_of_depots=len(depots))

    # Create state object
    initial_state = State(depots + clusters, rng=rng)

    # Sample size filtering. Create list of scooter ids to include
    sample_scooters = scooter_sample_filter(rng, entur_dataframe)

    # Trip intensity analysis
    methods.compute_and_set_trip_intensity(initial_state, sample_scooters, entur_data_dir)

    # Get probability of movement from scooters in a cluster
    probability_matrix = methods.scooter_movement_analysis(initial_state, entur_data_dir)
    initial_state.set_move_probabilities(probability_matrix)

    if number_of_bikes:
        initial_state.sample(number_of_bikes)
        for st in initial_state.stations.values():
            st.capacity = max(DEFAULT_STATION_CAPACITY, len(st.bikes))

    # Choosing a location as starting cluster for all vehicles
    current_location = (
        initial_state.depots[0] if initial_location_depot else initial_state.stations[0]
    )

    # Setting vehicles to initial state
    initial_state.vehicles = [
        Vehicle(i, current_location, VEHICLE_BATTERY_INVENTORY, VEHICLE_BIKE_INVENTORY)
        for i in range(number_of_vehicles)
    ]

    initial_state.scooters_in_use = {}
    if not FULL_TRIP:
        for id in range(number_of_bikes, number_of_bikes+initial_in_use):
            initial_state.scooters_in_use[id] = EBike(0, 0, 100, id)

    return initial_state


if __name__ == "__main__":
    get_initial_state(number_of_bikes=100, number_of_clusters=2)
