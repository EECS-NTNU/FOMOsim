"""
File containing the important neighbour filtering schema used to reduce the action space
"""


def filtering_neighbours(
    state, day, hour,
    vehicle,
    pick_up,
    delivery,
    number_of_neighbours,
    exclude=None,
):
    has_inventory = len(vehicle.scooter_inventory) + pick_up - delivery > 0
    exclude = exclude if exclude else []
    clusters_positive_deviation = sorted(
        [
            cluster
            for cluster in state.stations
            if cluster.id != vehicle.current_location.id
            and cluster.id not in exclude
            and len(cluster.get_available_scooters()) - cluster.get_ideal_state(day, hour) > 0
        ],
        key=lambda cluster: len(cluster.get_available_scooters()) - cluster.get_ideal_state(day, hour),
        reverse=True,
    )

    clusters_negative_deviation = sorted(
        [
            cluster
            for cluster in state.stations
            if cluster.id != vehicle.current_location.id
            and cluster.id not in exclude
            and len(cluster.get_available_scooters()) - cluster.get_ideal_state(day, hour) < 0
        ],
        key=lambda cluster: len(cluster.get_available_scooters()) - cluster.get_ideal_state(day, hour),
    )

    has_more_capacity = (
        len(vehicle.scooter_inventory) + pick_up - delivery
        < vehicle.scooter_inventory_capacity
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
        return [ cluster for cluster in state.stations if cluster.id != vehicle.current_location.id and cluster.id not in exclude ]

def add_depots_as_neighbours(state, time, vehicle, max_swaps):
    """
    Adds big depot and closest available (able to change all flat batteries) small depot
    """
    if vehicle.is_at_depot() or not (
        time
        and vehicle.battery_inventory - max_swaps
        < vehicle.battery_inventory_capacity * 0.2
    ):
        return []

    big_depot, *small_depots = state.depots
    # Filter out small depots that are not able to change all flat batteries for current vehicles
    available_small_depots = [
        depot
        for depot in small_depots
        if depot.get_available_battery_swaps(time) >= vehicle.flat_batteries()
    ]
    return (
        [big_depot]
        + [
            min(
                [depot for depot in available_small_depots],
                key=lambda depot: state.get_distance(
                    vehicle.current_location.id, depot.id
                ),
            )
        ]
        if available_small_depots
        else [big_depot]
    )


# def get_deviation_ideal_state(state, day, hour):
#     # cluster score based on deviation
#     return [cluster.get_ideal_state(day, hour) - len(cluster.scooters) for cluster in state.stations]


# def get_battery_deficient_in_clusters(state):
#     # cluster score based on how much deficient of battery the cluster have
#     return [
#         len(cluster.scooters) - cluster.get_current_state()
#         for cluster in state.stations
#     ]
