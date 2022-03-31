import clustering.scripts
import sim
from progress.bar import Bar
import os
import numpy as np
import copy
import settings
import math

SHIFT_DURATION=960

def haflan_haga_spetalen_ideal_state(state):
    entur_main_file = "0900-entur-snapshot.csv"
    entur_data_dir = "test_data"

    state.simulation_scenarios = generate_scenarios(state)

    entur_dataframe = clustering.methods.read_bounded_csv_file(entur_data_dir + "/" + entur_main_file)
    sample_scooters = clustering.scripts.scooter_sample_filter(state.rng, entur_dataframe, len(state.get_all_scooters()))

    compute_and_set_ideal_state(state, sample_scooters, entur_data_dir)

def fast_choice(rng, values):
    L = len(values)
    i = rng.integers(0, L)
    return values[i]

def system_simulate(state):
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
        if start != end
    }
    trips = []
    lost_demand = []
    scenario = fast_choice(state.rng, state.simulation_scenarios)

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

def generate_scenarios(state: sim.State, number_of_scenarios=10000):
    """
    Generate system simulation scenarios. This is used to speed up the training simulation
    :param state: new state
    :param number_of_scenarios: how many scenarios to generate
    :return: the scenarios list of (cluster id, number of trips, list of end cluster ids)
    """
    scenarios = []
    cluster_indices = np.arange(len(state.locations))
    for i in range(number_of_scenarios):
        one_scenario = []
        for cluster in state.stations:
            number_of_trips = round(
                state.rng.poisson(cluster.get_leave_intensity(0, 0))
            )
            end_cluster_indices = state.rng.choice(
                cluster_indices,
                p=cluster.get_leave_distribution(state, 0, 0),
                size=number_of_trips,
            ).tolist()

            one_scenario.append((cluster.id, number_of_trips, end_cluster_indices))

        scenarios.append(one_scenario)
    return scenarios

def normalize_to_integers(array, sum_to=1):
    normalized_cluster_ideal_states = sum_to * array / sum(array)
    rests = normalized_cluster_ideal_states - np.floor(normalized_cluster_ideal_states)
    number_of_ones = int(round(sum(rests)))
    sorted_rests = np.sort(rests)
    return np.array(
        np.floor(normalized_cluster_ideal_states)
        + [1 if rest in sorted_rests[-number_of_ones:] else 0 for rest in rests],
        dtype="int32",
    ).tolist()

def idealize_state(state):
    state_rebalanced_ideal_state = copy.deepcopy(state)

    # Set all clusters to ideal state
    excess_scooters = []
    for cluster in state_rebalanced_ideal_state.stations:
        # Swap all scooters under 50% battery
        for scooter in cluster.scooters:
            if isinstance(scooter, sim.Scooter) and scooter.battery < 50:
                scooter.swap_battery()
        # Find scooters possible to pick up
        positive_deviation = len(cluster.get_available_scooters()) - cluster.get_ideal_state(0, 0)
        if positive_deviation > 0:
            # Add scooters possible to pick up
            excess_scooters += [
                (scooter, cluster) for scooter in cluster.scooters[:positive_deviation]
            ]

    # Add excess scooters to clusters in need of scooters
    for cluster in state_rebalanced_ideal_state.stations:
        # Find out how many scooters to add to cluster
        number_of_scooters_to_add = cluster.get_ideal_state(0, 0) - len(
            cluster.get_available_scooters()
        )
        # Add scooters to the cluster only if the number of available scooter is lower than ideal state
        if number_of_scooters_to_add > 0:
            for _ in range(number_of_scooters_to_add):
                if len(excess_scooters) > 0:
                    # fetch and remove a scooter from the excess scooters
                    scooter, origin_cluster = excess_scooters.pop()
                    # Remove scooter from old cluster
                    origin_cluster.remove_scooter(scooter)
                    # Add scooter to new cluster
                    cluster.add_scooter(state.rng, scooter)

    return state_rebalanced_ideal_state

SIMULATIONS = 100

def simulate_state_outcomes(state_rebalanced_ideal_state, state, day, hour):
    # dict to record the outcomes of available scooters in a cluster after simulation
    simulating_outcomes = {
        cluster_id: []
        for cluster_id in range(len(state_rebalanced_ideal_state.locations))
    }

    progressbar = Bar("| Simulating state outcomes", max=SIMULATIONS)

    # simulating 100 times
    for i in range(SIMULATIONS):
        progressbar.next()

        simulating_state = state_rebalanced_ideal_state.sloppycopy()
        simulating_state.simulation_scenarios = state_rebalanced_ideal_state.simulation_scenarios
        # simulates until the end of the day
        for j in range(
            round(
                SHIFT_DURATION
                / settings.ITERATION_LENGTH_MINUTES
            )
        ):
            system_simulate(simulating_state)

        # recording the available scooters in every cluster after a day
        for cluster in simulating_state.stations:
            simulating_outcomes[cluster.id].append(
                len(cluster.get_available_scooters())
            )

    new_ideal_states = {}

    delta_ideal_state_and_outcomes = {}
    for cluster in state_rebalanced_ideal_state.stations:
        simulating_outcome = simulating_outcomes[cluster.id]
        # setting the new ideal state to trip intensity if min of all outcomes is larger than ideal state
        # -> in all scenarios there is a positive inflow to the cluster
        if min(simulating_outcome) > cluster.get_ideal_state(day, hour):
            new_ideal_states[cluster.id] = math.ceil(
                cluster.get_leave_intensity(day, hour)
            )
        # calculating the difference from outcomes and ideal state
        else:
            delta_ideal_state_and_outcomes[cluster.id] = [
                cluster.get_ideal_state(day, hour) - outcome
                for outcome in simulating_outcomes[cluster.id]
            ]

    # initial parameter for the percentile and delta of the percentile
    percentile = 1.0
    delta = 0.01

    # loop until the sum of new ideal states is less or equal to the number of scooters in the state
    while True:
        if (percentile - delta) <= 0:
            sum_ideal_state = sum(list(new_ideal_states.values()))

            while sum_ideal_state > len(state.get_scooters()):
                cluster_id = state.rng.choice(list(delta_ideal_state_and_outcomes.keys()))
                new_ideal_states[cluster_id] -= 1
                sum_ideal_state = sum(list(new_ideal_states.values()))

        else:
            # setting the new ideal state of the clusters where not all outcomes are greater than ideal state to:
            # ideal state + %-percentile of the difference between ideal state - outcomes
            for cluster_id in delta_ideal_state_and_outcomes.keys():
                quantile_outcomes = round(
                    np.quantile(delta_ideal_state_and_outcomes[cluster_id], percentile)
                )
                # TODO, this code gave typeError can only concatenate list (not "int") to list. 220331-Lasse    
                # new_ideal_states[cluster_id] = (
                #     state_rebalanced_ideal_state.locations[cluster_id].ideal_state
                #     + quantile_outcomes
                #     if quantile_outcomes > 0
                #     else 0
                # )

                # TODO, tried to fix wit this, also crash 220331-Lasse
                # new_ideal_states[cluster_id].append(
                #     state_rebalanced_ideal_state.locations[cluster_id].ideal_state
                #     + quantile_outcomes
                #     if quantile_outcomes > 0
                #     else 0
                # )


        sum_ideal_state = sum(list(new_ideal_states.values()))

        # breaking
        if sum_ideal_state <= len(state.get_scooters()):
            for cluster_id in new_ideal_states.keys():
                state.locations[cluster_id].average_number_of_scooters = state.locations[
                    cluster_id
                ].ideal_state
                state.locations[cluster_id].ideal_state[day][hour] = new_ideal_states[cluster_id]
            break

        percentile -= delta

    progressbar.finish()

            
def compute_and_set_ideal_state(state, sample_scooters: list, entur_data_dir):
    progressbar = Bar(
        "| Computing ideal state", max=len(os.listdir(entur_data_dir))
    )
    number_of_scooters_counter = np.zeros(
        (len(state.locations), len(os.listdir(entur_data_dir)))
    )
    for index, file_path in enumerate(sorted(os.listdir(entur_data_dir))):
        progressbar.next()
        current_snapshot = clustering.methods.read_bounded_csv_file(f"{entur_data_dir}/{file_path}")
        current_snapshot = current_snapshot[
            current_snapshot["id"].isin(sample_scooters)
        ]
        current_snapshot["cluster"] = [
            state.get_cluster_by_lat_lon(row["lat"], row["lon"]).id
            for index, row in current_snapshot.iterrows()
        ]
        for cluster in state.stations:
            number_of_scooters_counter[cluster.id][index] = len(
                current_snapshot[current_snapshot["cluster"] == cluster.id]
            )
    cluster_ideal_states = np.mean(number_of_scooters_counter, axis=1)
    normalized_cluster_ideal_states = normalize_to_integers(
        cluster_ideal_states, sum_to=len(sample_scooters)
    )

    progressbar.finish()

    for cluster in state.stations:
        cluster.ideal_state = [[normalized_cluster_ideal_states[cluster.id]]]

    # setting number of scooters to ideal state
    state_rebalanced_ideal_state = idealize_state(state)

    # adjusting ideal state by average cluster in- and outflow
    simulate_state_outcomes(state_rebalanced_ideal_state, state, 0, 0)

    # copy all day0 hour0 values to all other times
    for cluster in state.stations:
        for day in range(7):
            if day > 0:
                cluster.ideal_state.append([])
            for hour in range(24):
                if (day != 0) or (hour > 0):
                    cluster.ideal_state[day].append([])
                cluster.ideal_state[day][hour] = cluster.ideal_state[0][0]
