"""
Main file for clustering module
"""
from sim import Vehicle, State, Bike, Scooter
import clustering.methods as methods
import os
from settings import *
import pandas as pd

def scooter_sample_filter(dataframe: pd.DataFrame, sample_size=None):
    if sample_size:
        dataframe = dataframe.sample(sample_size)
    return dataframe["id"].tolist()


def get_initial_state(
    classname,
    sample_size=None,
    initial_in_use=0,
    number_of_clusters=20,
    initial_location_depot=True,
    number_of_vans=4,
) -> State:
    """
    Main method for setting up a state object based on EnTur data from test_data directory.
    This method saves the state objects and reuse them if a function call is done with the same properties.
    :param sample_size: number of e-scooters in the instance
    :param number_of_clusters
    :param initial_location_depot: set the initial current location of the vehicles to the depot
    :param number_of_vans: the number of vans to use
    :return:
    """
    # If this combination has been requested before we fetch a cached version
    print(
        f"\nSetup initial state from entur dataset with {number_of_clusters} clusters and {sample_size} scooters"
    )
    # Get dataframe from EnTur CSV file within boundary
    entur_dataframe = methods.read_bounded_csv_file(
        "test_data/0900-entur-snapshot.csv"
    )

    # Create clusters
    cluster_labels = methods.cluster_data(entur_dataframe, number_of_clusters)

    # Structure data into objects
    clusters = methods.generate_cluster_objects(classname, entur_dataframe, cluster_labels)

    # generate depots and adding them to clusters list
    depots = methods.generate_depots(number_of_clusters=len(clusters))

    # Create state object
    initial_state = State(clusters, depots)

    # Sample size filtering. Create list of scooter ids to include
    sample_scooters = scooter_sample_filter(entur_dataframe, sample_size)

    # Trip intensity analysis
    methods.compute_and_set_trip_intensity(initial_state, sample_scooters)

    # Get probability of movement from scooters in a cluster
    probability_matrix = methods.scooter_movement_analysis(initial_state)
    initial_state.set_probability_matrix(probability_matrix)

    if sample_size:
        initial_state.sample(sample_size)

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
            Bike(0, 0, index+sample_size)
            for index in range(initial_in_use)
          ]
        else:
          initial_state.scooters_in_use = [
            Scooter(0, 0, 100, index+sample_size)
            for index in range(initial_in_use)
          ]

    return initial_state


if __name__ == "__main__":
    get_initial_state(sample_size=100, number_of_clusters=2)
