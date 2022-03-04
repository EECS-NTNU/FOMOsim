import copy
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
import os

from sim import State, Bike, Scooter, Station
from sim.Depot import Depot
from settings import (
    GEOSPATIAL_BOUND_NEW,
    MAIN_DEPOT_LOCATION,
    SMALL_DEPOT_LOCATIONS,
    TRIP_INTENSITY_FACTOR,
)
from progress.bar import Bar

def get_moved_scooters(merged_tables):
    return merged_tables[
        (merged_tables["battery_before"] > merged_tables["battery_after"])
        & (abs(merged_tables["lat_before"] - merged_tables["lat_after"]) >= 0.0001)
        & (abs(merged_tables["lon_before"] - merged_tables["lon_after"]) >= 0.0001)
    ].copy()



def merge_scooter_snapshots(state, first_snapshot_data, second_snapshot_data):
    # Join tables on scooter id
    merged_tables = pd.merge(
        left=first_snapshot_data,
        right=second_snapshot_data,
        left_on="id",
        right_on="id",
        how="inner",
        suffixes=("_before", "_after"),
    )

    # Filtering out scooters that has moved during the 20 minutes
    moved_scooters = get_moved_scooters(merged_tables)
    moved_scooters["cluster_before"] = [
        state.get_cluster_by_lat_lon(row["lat_before"], row["lon_before"]).id
        for index, row in moved_scooters.iterrows()
    ]
    moved_scooters["cluster_after"] = [
        state.get_cluster_by_lat_lon(row["lat_after"], row["lon_after"]).id
        for index, row in moved_scooters.iterrows()
    ]
    # Remove the scooters who move, but within the same cluster
    filtered_moved_scooters = moved_scooters[
        moved_scooters["cluster_before"] != moved_scooters["cluster_after"]
    ].copy()

    # Due to the dataset only showing available scooters we need to find out how many scooters leave the zone
    # resulting in a battery percent below 20. To find these scooters we find scooters from the first snapshot
    # that is not in the merge. The "~" symbol indicates "not" in pandas boolean indexing
    disappeared_scooters: pd.DataFrame = first_snapshot_data.loc[
        ~first_snapshot_data["id"].isin(merged_tables["id"])
    ].copy()
    # Find the origin cluster for these scooters
    disappeared_scooters["cluster_id"] = [
        state.get_cluster_by_lat_lon(row["lat"], row["lon"])
        for index, row in disappeared_scooters.iterrows()
    ]
    return moved_scooters, filtered_moved_scooters, disappeared_scooters


def read_bounded_csv_file(
    file_path: str, sample_size=None, separator=",", operator=None
) -> pd.DataFrame:
    """
    Reads csv file from Entur and outputs a dataframe
    with scooters within the given boundary
    :param file_path: filepath to csv file
    :param sample_size: integer number with number of scooters to fetch
    :param separator: how to separate the values in a row of the csv default ";"
    :param operator: Either "voi" or "tier"'
    :return: dataframe with scooter data
    """
    # Get EnTur data from csv file
    raw_data = pd.read_csv(file_path, sep=separator)
    # Hardcoded boundary on data
    lat_min, lat_max, lon_min, lon_max = GEOSPATIAL_BOUND_NEW
    # Filter out data not within boundary
    raw_data = raw_data.loc[
        (
            (lon_min <= raw_data["lon"])
            & (raw_data["lon"] <= lon_max)
            & (lat_min <= raw_data["lat"])
            & (raw_data["lat"] <= lat_max)
        )
    ]

    # Only two operators in the current dataset
    if operator == "voi" or operator == "tier":
        # Reduce number of scooters with a operator filter
        raw_data = raw_data[raw_data["operator"] == operator]

    if sample_size:
        # Reduce number of scooters with a sample size
        raw_data = raw_data.sample(sample_size)

    return raw_data


def cluster_data(rng, data: pd.DataFrame, number_of_clusters: int) -> [int]:
    """
    Uses an clustering algorithm to group together scooters
    :param data: geospatial data containing cols ["lat", "lon"]
    :param number_of_clusters: how many clusters to create
    :return: list of labels for input data
    """
    # Generate numpy array from dataframe
    coords = data[["lat", "lon"]].values
    # Run k-means algorithm to generate clusters
    return KMeans(number_of_clusters, random_state=rng.integers(0, 1000)).fit(coords).labels_


def scooter_movement_analysis(state: State, entur_data_dir) -> np.ndarray:
    def get_probability_matrix(
        initial_state: State,
        first_snapshot_data: pd.DataFrame,
        second_snapshot_data: pd.DataFrame,
    ) -> np.ndarray:
        """
        Based on the clusters created and the two snapshots provided, a matrix corresponding to the
        probability that a scooter will move between two clusters.
        E.g. probability_matrix[3][5] - will return the probability for a scooter of moving from cluster 3 to 5
        probability_matrix[3][3] - will return the probability of a scooter staying in cluster 3
        :param initial_state: state with list of generated clusters
        :param first_snapshot_data: geospatial data for scooters in first snapshot
        :param second_snapshot_data: geospatial data for scooters in first snapshot
        :return: probability matrix
        """
        (_, filtered_moved_scooters, disappeared_scooters) = merge_scooter_snapshots(
            initial_state, first_snapshot_data, second_snapshot_data
        )
        # Get list of cluster ids and find number of clusters for dimensions of arrays
        cluster_labels = [cluster.id for cluster in initial_state.locations]
        number_of_clusters = len(cluster_labels)

        # Initialize probability_matrix with number of scooters in each cluster
        number_of_scooters = np.array(
            [[cluster.number_of_scooters() for cluster in initial_state.locations]]
            * number_of_clusters,
            dtype="float64",
        ).transpose()

        # Create counter for every combination of cluster
        move_count = np.zeros((number_of_clusters, number_of_clusters), dtype="float64")
        # Count how many scooters move
        for _, row in filtered_moved_scooters.iterrows():
            # Increase the counter for every visit if the scooter has moved to a new cluster
            move_count[row["cluster_before"]][row["cluster_after"]] += 1

        # Calculate number of scooters who stayed in each zone
        for cluster_id in cluster_labels:
            # Formula: Number of scooters from beginning - number of scooters leaving - number of disappeared scooters
            move_count[cluster_id][cluster_id] = (
                number_of_scooters[cluster_id][cluster_id]
                - sum(
                    move_count[cluster_id][np.arange(number_of_clusters) != cluster_id]
                )
                # (np.arange(number_of_clusters) != cluster_id) says "all indices except cluster_id"
                - len(
                    disappeared_scooters[
                        disappeared_scooters["cluster_id"] == cluster_id
                    ]
                )
            )

        # Calculate the probability matrix
        np.seterr(divide='ignore', invalid='ignore')
        probability_matrix = move_count / number_of_scooters

        # Normalize non stay distribution - Same as distribute the disappeared scooter with same distribution
        for cluster_id in cluster_labels:
            # Calculate probability of a scooter leaving the cluster
            prob_leave = 1 - probability_matrix[cluster_id][cluster_id]

            # Extract leaving probabilities from move probabilities
            leaving_probabilities = probability_matrix[cluster_id][
                np.arange(number_of_clusters) != cluster_id
            ]

            # Make sure that sum of all leave probabilities equals prob leave
            probability_matrix[cluster_id][
                np.arange(number_of_clusters) != cluster_id
            ] = np.divide(
                prob_leave * leaving_probabilities,
                np.sum(leaving_probabilities),
                out=np.zeros_like(leaving_probabilities),
                where=np.sum(leaving_probabilities) != 0,
            )
        return probability_matrix

    progress = Bar(
        "| Computing MPM",
        max=len(os.listdir(entur_data_dir)),
    )
    # Fetch all snapshots from test data
    probability_matrices = []
    previous_snapshot = None
    for index, file_path in enumerate(sorted(os.listdir(entur_data_dir))):
        progress.next()
        current_snapshot = read_bounded_csv_file(f"{entur_data_dir}/{file_path}")
        if previous_snapshot is not None:
            probability_matrices.append(
                get_probability_matrix(state, previous_snapshot, current_snapshot)
            )
        previous_snapshot = current_snapshot
    progress.finish()
    # Compute mean
    return np.mean(probability_matrices, axis=0)


def generate_cluster_objects(
        classname, scooter_data: pd.DataFrame, cluster_labels: list, number_of_depots,
) -> [Station]:
    """
    Based on cluster labels and scooter data create Scooter and Station objects.
    Station class generates cluster center
    :param scooter_data: geospatial data for scooters
    :param cluster_labels: list of labels for scooter data
    :return: list of clusters
    """
    # Add cluster labels as a row to the scooter data dataframe
    scooter_data_w_labels = scooter_data.copy()
    scooter_data_w_labels["cluster_labels"] = cluster_labels
    # Generate series of scooters belonging to each cluster
    clusters = []
    for cluster_label in np.unique(cluster_labels):
        # Filter out scooters within cluster
        cluster_scooters = scooter_data_w_labels[
            scooter_data_w_labels["cluster_labels"] == cluster_label
        ]
        # Generate scooter objects, using index as ID
        scooters = []
        if classname == "Bike":
          scooters = [
            Bike(row["lat"], row["lon"], index)
            for index, row in cluster_scooters.iterrows()
          ]
        else:
          scooters = [
            Scooter(row["lat"], row["lon"], row["battery"], index)
            for index, row in cluster_scooters.iterrows()
          ]
        # Adding all scooters to cluster to find center location
        clusters.append(Station(cluster_label + number_of_depots, scooters))
    return sorted(clusters, key=lambda cluster: cluster.id)


def compute_and_set_trip_intensity(state: State, sample_scooters: list, entur_data_dir):
    """
    Counts the number of e-scooters leaving each cluster and average over all snapshots to calculate the trip intensity
    :param state: state object
    :param sample_scooters: ids of e-scooters in the sample used
    """
    progress = Bar(
        "| Computing trip intensity",
        max=len(os.listdir(entur_data_dir)),
    )
    # Fetch all snapshots from test data
    trip_counter_leave = np.zeros((len(state.locations), len(os.listdir(entur_data_dir))))
    trip_counter_arrive = np.zeros((len(state.locations), len(os.listdir(entur_data_dir))))
    previous_snapshot = None
    for index, file_path in enumerate(sorted(os.listdir(entur_data_dir))):
        progress.next()
        current_snapshot = read_bounded_csv_file(f"{entur_data_dir}/{file_path}")
        current_snapshot = current_snapshot[
            current_snapshot["id"].isin(sample_scooters)
        ]
        if previous_snapshot is not None:
            (
                moved_scooters,
                filtered_moved_scooters,
                disappeared_scooters,
            ) = merge_scooter_snapshots(state, previous_snapshot, current_snapshot)
            for cluster in state.stations:
                scooters_leaving_the_cluster = filtered_moved_scooters[
                    filtered_moved_scooters["cluster_before"] == cluster.id
                ]
                scooters_arriving_to_the_cluster = filtered_moved_scooters[
                    filtered_moved_scooters["cluster_after"] == cluster.id
                ]
                # These are all the scooters that move, both within and to new clusters
                scooters_moving_in_the_cluster = moved_scooters[
                    moved_scooters["cluster_before"] == cluster.id
                ]

                # Number of scooters leaving the cluster
                # + number of disappeared scooters likely to leave ( # of disappeared * ratio of leaving)
                trip_counter_leave[cluster.id][index] = len(
                    scooters_leaving_the_cluster
                ) + round(
                    len(
                        disappeared_scooters[
                            disappeared_scooters["cluster_id"] == cluster.id
                        ]
                    )
                    * (
                        (
                            len(scooters_leaving_the_cluster)
                            / len(scooters_moving_in_the_cluster)
                        )
                        if len(scooters_moving_in_the_cluster)
                        else 1
                    )
                )
                # Number of scooters arriving to the cluster
                trip_counter_arrive[cluster.id][index] = len(scooters_arriving_to_the_cluster)

        previous_snapshot = current_snapshot

    cluster_leave_intensities = np.mean(trip_counter_leave, axis=1)
    cluster_arrive_intensities = np.mean(trip_counter_arrive, axis=1)
    for cluster in state.stations:
        cluster.leave_intensity_per_iteration = cluster_leave_intensities[cluster.id] * TRIP_INTENSITY_FACTOR
        cluster.arrive_intensity_per_iteration = cluster_arrive_intensities[cluster.id] * TRIP_INTENSITY_FACTOR

    progress.finish()


def generate_depots():
    """
    Generate depot objects
    :return: depot objects
    """
    main_depot_lat, main_depot_lon = MAIN_DEPOT_LOCATION
    depots = [
        Depot(main_depot_lat, main_depot_lon, 0, main_depot=True)
    ]

    for i, (lat, lon) in enumerate(SMALL_DEPOT_LOCATIONS):
        depots.append(Depot(lat, lon, i + 1, main_depot=False))

    return depots

