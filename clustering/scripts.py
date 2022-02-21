"""
Main file for clustering module
"""
from sim import Vehicle, State, Bike, Scooter
import clustering.methods as methods
import os
from settings import *
import pandas as pd
import numpy as np

def scooter_sample_filter(rng, dataframe: pd.DataFrame, number_of_scooters=None):
    if number_of_scooters:
        dataframe = dataframe.sample(number_of_scooters, random_state=rng.integers(0, 1000))
    return dataframe["id"].tolist()


def get_initial_state(
    classname,
    number_of_scooters=None,
    initial_in_use=0,
    number_of_clusters=20,
    initial_location_depot=True,
    number_of_vans=4,
    random_seed=None,
) -> State:
    """
    Main method for setting up a state object based on EnTur data from test_data directory.
    This method saves the state objects and reuse them if a function call is done with the same properties.
    :param number_of_scooters: number of e-scooters in the instance
    :param number_of_clusters
    :param initial_location_depot: set the initial current location of the vehicles to the depot
    :param number_of_vans: the number of vans to use
    :return:
    """
    # If this combination has been requested before we fetch a cached version
    print(
        f"\nSetup initial state from entur dataset with {number_of_clusters} clusters and {number_of_scooters} scooters"
    )
    # Get dataframe from EnTur CSV file within boundary
    entur_dataframe = methods.read_bounded_csv_file(
        "test_data/0900-entur-snapshot.csv"
    )

    rng=np.random.default_rng(random_seed)

    # Create clusters
    cluster_labels = methods.cluster_data(rng, entur_dataframe, number_of_clusters)

    # Structure data into objects
    clusters = methods.generate_cluster_objects(classname, entur_dataframe, cluster_labels)

    # generate depots and adding them to clusters list
    depots = methods.generate_depots(number_of_clusters=len(clusters))

    # Create state object
    initial_state = State(clusters, depots, rng=rng)

    # Sample size filtering. Create list of scooter ids to include
    sample_scooters = scooter_sample_filter(rng, entur_dataframe)

    # Trip intensity analysis
    methods.compute_and_set_trip_intensity(initial_state, sample_scooters)

    # Get probability of movement from scooters in a cluster
    probability_matrix = methods.scooter_movement_analysis(initial_state)
    initial_state.set_probability_matrix(probability_matrix)

    if number_of_scooters:
        initial_state.sample(number_of_scooters)

    # Choosing a location as starting cluster for all vehicles
    current_location = (
        initial_state.depots[0] if initial_location_depot else initial_state.stations[0]
    )

    # Setting vehicles to initial state
    initial_state.vehicles = [
        Vehicle(i, current_location, VAN_BATTERY_INVENTORY, VAN_SCOOTER_INVENTORY)
        for i in range(number_of_vans)
    ]

    initial_state.scooters_in_use = []
    if not FULL_TRIP:
        if classname == "Bike":
          initial_state.scooters_in_use = [
            Bike(0, 0, index+number_of_scooters)
            for index in range(initial_in_use)
          ]
        else:
          initial_state.scooters_in_use = [
            Scooter(0, 0, 100, index+number_of_scooters)
            for index in range(initial_in_use)
          ]

    return initial_state


if __name__ == "__main__":
    get_initial_state(number_of_scooters=100, number_of_clusters=2)
