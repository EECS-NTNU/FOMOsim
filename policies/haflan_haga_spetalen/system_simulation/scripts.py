"""
File containing the system simulation. The system simulation simulate customer behavior by generating trips
"""

import numpy as np

def fast_choice(rng, values):
    L = len(values)
    i = rng.integers(0, L)
    return values[i]

def system_simulate(state, day, hour):
    """
    Simulation of poisson process on the system
    Poisson distributed number of trips out of each cluster, markov chain decides where the trip goes
    :param state: current world
    :return: flows generated by the system simulation
    """
    flow_counter = {
        (start, end): 0
        for start in np.arange(len(state.locations))
        for end in np.arange(len(state.locations))
    }
    trips = []
    lost_demand = []
    scenario = fast_choice(state.rng, state.simulation_scenarios[day][hour])

    for start_cluster_id, number_of_trips, end_cluster_indices in scenario:
        start_cluster = state.get_location_by_id(start_cluster_id)
        # if there is more trips than scooters available, the system has lost demand
        valid_scooters = start_cluster.get_available_bikes()
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
        start_cluster.remove_scooter(scooter)
        travel_time = state.get_travel_time(start_cluster.id, end_cluster.id)
        scooter.travel(travel_time)
        end_cluster.add_bike(scooter)

    return (
        [(start, end, flow) for (start, end), flow in list(flow_counter.items())],
        trips,
        lost_demand,
    )
