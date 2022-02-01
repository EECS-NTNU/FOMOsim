import copy

from sklearn.cluster import KMeans
import os

from classes import State, Scooter, Cluster
from classes.Depot import Depot
from globals import (
    GEOSPATIAL_BOUND_NEW,
    TEST_DATA_DIRECTORY,
    MAIN_DEPOT_LOCATION,
    SMALL_DEPOT_LOCATIONS,
)
from progress.bar import Bar
from .helpers import *


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


def cluster_data(data: pd.DataFrame, number_of_clusters: int) -> [int]:
    """
    Uses an clustering algorithm to group together scooters
    :param data: geospatial data containing cols ["lat", "lon"]
    :param number_of_clusters: how many clusters to create
    :return: list of labels for input data
    """
    # Generate numpy array from dataframe
    coords = data[["lat", "lon"]].values
    # Run k-means algorithm to generate clusters
    return KMeans(number_of_clusters).fit(coords).labels_


def scooter_movement_analysis(state: State) -> np.ndarray:
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
        cluster_labels = [cluster.id for cluster in initial_state.clusters]
        number_of_clusters = len(cluster_labels)

        # Initialize probability_matrix with number of scooters in each cluster
        number_of_scooters = np.array(
            [[cluster.number_of_scooters() for cluster in initial_state.clusters]]
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
        max=len(os.listdir(TEST_DATA_DIRECTORY)),
    )
    # Fetch all snapshots from test data
    probability_matrices = []
    previous_snapshot = None
    for index, file_path in enumerate(sorted(os.listdir(TEST_DATA_DIRECTORY))):
        progress.next()
        current_snapshot = read_bounded_csv_file(f"{TEST_DATA_DIRECTORY}/{file_path}")
        if previous_snapshot is not None:
            probability_matrices.append(
                get_probability_matrix(state, previous_snapshot, current_snapshot)
            )
        previous_snapshot = current_snapshot
    progress.finish()
    # Compute mean
    return np.mean(probability_matrices, axis=0)


def generate_cluster_objects(
    scooter_data: pd.DataFrame, cluster_labels: list
) -> [Cluster]:
    """
    Based on cluster labels and scooter data create Scooter and Cluster objects.
    Cluster class generates cluster center
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
        # Generate scooter objets, using index as ID
        scooters = [
            Scooter(row["lat"], row["lon"], row["battery"], index)
            for index, row in cluster_scooters.iterrows()
        ]
        # Adding all scooters to cluster to find center location
        clusters.append(Cluster(cluster_label, scooters))
    return sorted(clusters, key=lambda cluster: cluster.id)


def compute_and_set_ideal_state(state: State, sample_scooters: list):
    progressbar = Bar(
        "| Computing ideal state", max=len(os.listdir(TEST_DATA_DIRECTORY))
    )
    number_of_scooters_counter = np.zeros(
        (len(state.clusters), len(os.listdir(TEST_DATA_DIRECTORY)))
    )
    for index, file_path in enumerate(sorted(os.listdir(TEST_DATA_DIRECTORY))):
        progressbar.next()
        current_snapshot = read_bounded_csv_file(f"{TEST_DATA_DIRECTORY}/{file_path}")
        current_snapshot = current_snapshot[
            current_snapshot["id"].isin(sample_scooters)
        ]
        current_snapshot["cluster"] = [
            state.get_cluster_by_lat_lon(row["lat"], row["lon"]).id
            for index, row in current_snapshot.iterrows()
        ]
        for cluster in state.clusters:
            number_of_scooters_counter[cluster.id][index] = len(
                current_snapshot[current_snapshot["cluster"] == cluster.id]
            )
    cluster_ideal_states = np.mean(number_of_scooters_counter, axis=1)
    normalized_cluster_ideal_states = normalize_to_integers(
        cluster_ideal_states, sum_to=len(sample_scooters)
    )

    for cluster in state.clusters:
        cluster.ideal_state = normalized_cluster_ideal_states[cluster.id]

    # setting number of scooters to ideal state
    state_rebalanced_ideal_state = idealize_state(state)

    # adjusting ideal state by average cluster in- and outflow
    simulate_state_outcomes(state_rebalanced_ideal_state, state)

    progressbar.finish()


def compute_and_set_trip_intensity(state: State, sample_scooters: list):
    """
    Counts the number of e-scooters leaving each cluster and average over all snapshots to calculate the trip intensity
    :param state: state object
    :param sample_scooters: ids of e-scooters in the sample used
    """
    progress = Bar(
        "| Computing trip intensity",
        max=len(os.listdir(TEST_DATA_DIRECTORY)),
    )
    # Fetch all snapshots from test data
    trip_counter = np.zeros((len(state.clusters), len(os.listdir(TEST_DATA_DIRECTORY))))
    previous_snapshot = None
    for index, file_path in enumerate(sorted(os.listdir(TEST_DATA_DIRECTORY))):
        progress.next()
        current_snapshot = read_bounded_csv_file(f"{TEST_DATA_DIRECTORY}/{file_path}")
        current_snapshot = current_snapshot[
            current_snapshot["id"].isin(sample_scooters)
        ]
        if previous_snapshot is not None:
            (
                moved_scooters,
                filtered_moved_scooters,
                disappeared_scooters,
            ) = merge_scooter_snapshots(state, previous_snapshot, current_snapshot)
            for cluster in state.clusters:
                scooters_leaving_the_cluster = filtered_moved_scooters[
                    filtered_moved_scooters["cluster_before"] == cluster.id
                ]
                # These are all the scooters that move, both within and to new clusters
                scooters_moving_in_the_cluster = moved_scooters[
                    moved_scooters["cluster_before"] == cluster.id
                ]
                # Number of scooters leaving the cluster
                # + number of disappeared scooters likely to leave ( # of disappeared * ratio of leaving)
                trip_counter[cluster.id][index] = len(
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
        previous_snapshot = current_snapshot
    cluster_trip_intensities = np.mean(trip_counter, axis=1)
    for cluster in state.clusters:
        cluster.trip_intensity_per_iteration = cluster_trip_intensities[cluster.id]
    progress.finish()


def generate_depots(number_of_clusters):
    """
    Generate depot objects
    :param number_of_clusters: the number of clusters in the state created
    :return: depot objects
    """
    main_depot_lat, main_depot_lon = MAIN_DEPOT_LOCATION
    depots = [
        Depot(main_depot_lat, main_depot_lon, number_of_clusters, main_depot=True)
    ]

    for i, (lat, lon) in enumerate(SMALL_DEPOT_LOCATIONS):
        depots.append(Depot(lat, lon, i + number_of_clusters + 1, main_depot=False))

    return depots


def generate_scenarios(state: State, number_of_scenarios=10000):
    """
    Generate system simulation scenarios. This is used to speed up the training simulation
    :param state: new state
    :param number_of_scenarios: how many scenarios to generate
    :return: the scenarios list of (cluster id, number of trips, list of end cluster ids)
    """
    scenarios = []
    cluster_indices = np.arange(len(state.clusters))
    for i in range(number_of_scenarios):
        one_scenario = []
        for cluster in state.clusters:
            number_of_trips = round(
                np.random.poisson(cluster.trip_intensity_per_iteration)
            )
            end_cluster_indices = np.random.choice(
                cluster_indices,
                p=cluster.get_leave_distribution(),
                size=number_of_trips,
            ).tolist()
            one_scenario.append((cluster.id, number_of_trips, end_cluster_indices))

        scenarios.append(one_scenario)
    return scenarios
