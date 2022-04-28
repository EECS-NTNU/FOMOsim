from progress.bar import Bar

import sim
import numpy as np

def generate_scenarios(state: sim.State, number_of_scenarios=10000):
    """
    Generate system simulation scenarios. This is used to speed up the training simulation
    :param state: new state
    :param number_of_scenarios: how many scenarios to generate
    :return: the scenarios list of (cluster id, number of trips, list of end cluster ids)
    """

    progress = Bar(
        "| Generating Scenarios",
        max=7*24,
    )

    scenarios = []
    for day in range(7):
        scenarios.append([])
        for hour in range(24):
            scenarios[day].append([])
            scenarios[day][hour] = []

            cluster_indices = np.arange(len(state.locations))
            for i in range(number_of_scenarios):
                one_scenario = []
                for cluster in state.stations:
                    number_of_trips = round(
                        state.rng.poisson(cluster.get_leave_intensity(day, hour))
                    )
                    end_cluster_indices = state.rng.choice(
                        cluster_indices,
                        p=cluster.get_leave_distribution(state, day, hour),
                        size=number_of_trips,
                    ).tolist()
                    one_scenario.append((cluster.id, number_of_trips, end_cluster_indices))

                scenarios[day][hour].append(one_scenario)
        
            progress.next()
    progress.finish()
    return scenarios
