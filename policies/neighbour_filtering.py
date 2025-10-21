"""
File containing the important neighbour filtering schema used to reduce the action space
From HHS2021, only used in epsilon_greedy
"""


def filtering_neighbours(
    state, day, hour,
    vehicle,
    pick_up,
    delivery,
    number_of_neighbours,
    exclude=None,
):
    has_inventory = len(vehicle.bike_inventory) + pick_up - delivery > 0
    exclude = exclude if exclude else []
    clusters_positive_deviation = sorted(
        [
            cluster
            for cluster in state.locations.values()
            if cluster.location_id != vehicle.location.location_id
            and cluster.location_id not in exclude
            and len(cluster.get_available_bikes()) - cluster.get_target_state(day, hour) > 0
        ],
        key=lambda cluster: len(cluster.get_available_bikes()) - cluster.get_target_state(),
        reverse=True,
    )

    clusters_negative_deviation = sorted(
        [
            cluster
            for cluster in state.locations.values()
            if cluster.location_id != vehicle.location.location_id
            and cluster.location_id not in exclude
            and len(cluster.get_available_bikes()) - cluster.get_target_state(day, hour) < 0
        ],
        key=lambda cluster: len(cluster.get_available_bikes()) - cluster.get_target_state(),
    )

    has_more_capacity = (
        len(vehicle.bike_inventory) + pick_up - delivery
        < vehicle.bike_inventory_capacity
    )

    if has_inventory:
        if has_more_capacity and len(clusters_positive_deviation) > 0:
            returnval = clusters_negative_deviation[: number_of_neighbours - 1] + [
                clusters_positive_deviation[0]
            ]
        else:
            returnval = clusters_negative_deviation[:number_of_neighbours]
    else:
        returnval = clusters_positive_deviation[:number_of_neighbours]

    if len(returnval) > 0:
        return returnval
    else:
        return [ cluster for cluster in state.locations.values() if cluster.location_id != vehicle.location.location_id and cluster.location_id not in exclude ]
