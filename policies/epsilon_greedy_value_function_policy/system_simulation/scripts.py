"""
File containing the system simulation. The system simulation simulate customer behavior by generating trips
"""

import numpy as np


def system_simulate(state):
    """
    Simulation of poisson process on the system
    Poisson distributed number of trips out of each cluster, markov chain decides where the trip goes
    :param state: current world
    :return: flows generated by the system simulation
    """
    flow_counter = {
        (start, end): 0
        for start in np.arange(len(state.stations))
        for end in np.arange(len(state.stations))
        if start != end
    }
    trips = []
    lost_demand = []
    scenario = state.rng.choice(state.simulation_scenarios)
    for start_cluster_id, number_of_trips, end_cluster_indices in scenario:
        start_cluster = state.get_location_by_id(start_cluster_id)
        # if there is more trips than scooters available, the system has lost demand
        valid_scooters = start_cluster.get_available_scooters()
        if number_of_trips > len(valid_scooters):
            lost_demand.append(
                (number_of_trips - len(valid_scooters), start_cluster_id)
            )
            end_cluster_indices = end_cluster_indices[: len(valid_scooters)]

        # loop to generate trips from the cluster
        for j, end_cluster_index in enumerate(end_cluster_indices):
            trips.append(
                (
                    start_cluster,
                    state.get_location_by_id(end_cluster_index),
                    valid_scooters.pop(0),
                )
            )
            flow_counter[(start_cluster.id, end_cluster_index)] += 1

    # compute trip after all trips are generated to avoid handling inflow in cluster
    for start_cluster, end_cluster, scooter in trips:
        start_cluster.scooters.remove(scooter)
        trip_distance = state.get_distance(start_cluster.id, end_cluster.id)
        scooter.travel(trip_distance)
        end_cluster.add_scooter(state.rng, scooter)

    return (
        [(start, end, flow) for (start, end), flow in list(flow_counter.items())],
        trips,
        lost_demand,
    )
