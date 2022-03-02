import itertools
from scipy.interpolate import make_interp_spline
from matplotlib import gridspec
from settings import (
    BLACK,
    RED,
    BLUE,
    GEOSPATIAL_BOUND_NEW,
    COLORS,
    ACTION_OFFSET,
    VEHICLE_COLORS,
)
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sim


def display_graph(
    graph, node_color, node_border, node_size, labels, font_size, ax, with_labels=True
):
    """
    Displaying a networkx graph
    """
    pos = nx.get_node_attributes(graph, "pos")

    # set edge color for solution
    edges = graph.edges()
    edge_colors = [graph[u][v]["color"] for u, v in edges]
    edge_weights = [graph[u][v]["width"] for u, v in edges]

    # draw solution graph
    nx.draw(
        graph,
        pos,
        node_color=node_color,
        edgecolors=node_border,
        edge_color=edge_colors,
        width=edge_weights,
        node_size=node_size,
        linewidths=2,
        alpha=0.7,
        with_labels=with_labels,
        ax=ax,
    )
    if with_labels:
        nx.draw_networkx_labels(
            graph,
            pos,
            labels,
            font_size=font_size,
            font_color="white",
            font_weight="bold",
            ax=ax,
        )


def plot_vehicle_info(current_vehicle, next_vehicle, ax):
    """
    Adds vehicle information to a subplot
    """

    # vehicle info box
    current_vehicle_info = (
        f"Current Vehicle:\nCap - {current_vehicle.scooter_inventory_capacity}\n"
        f"Battery - {current_vehicle.battery_inventory}\n"
        f"Scooters:"
    )

    for scooter in current_vehicle.scooter_inventory:
        current_vehicle_info += f"\n{scooter}"

    next_vehicle_info = (
        f"Next Vehicle:\nCap - {next_vehicle.scooter_inventory_capacity}\n"
        f"Battery - {next_vehicle.battery_inventory}\n"
        f"Scooters:"
    )

    for scooter in next_vehicle.scooter_inventory:
        next_vehicle_info += f"\n{scooter}"

    props = dict(boxstyle="round", facecolor="wheat", pad=0.5, alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(
        0,
        0.98,
        current_vehicle_info,
        transform=ax.transAxes,
        fontsize=10,
        horizontalalignment="left",
        verticalalignment="top",
        bbox=props,
    )

    ax.text(
        0,
        0.89 - ACTION_OFFSET * len(current_vehicle.scooter_inventory),
        next_vehicle_info,
        transform=ax.transAxes,
        fontsize=10,
        horizontalalignment="left",
        verticalalignment="top",
        bbox=props,
    )


def plot_action(action, current_location, world_time, action_time, ax, offset=0):
    """
    Adds action information to a subplot
    """
    props = dict(boxstyle="round", facecolor="wheat", pad=0.5, alpha=0.5)

    action_string = f"Swaps:\n {len(action.battery_swaps)}\n"

    action_string += f"\nPickups:\n {len(action.pick_ups)}\n"

    action_string += f"\nDeliveries:\n {len(action.delivery_scooters)}\n"

    action_string += f"\nCurrent : {current_location}\nNext : {action.next_location}"

    action_string += f"\nCurrent time: {world_time}\nAction time: {action_time}"

    ax.text(
        0,
        0.80 - offset,
        action_string,
        transform=ax.transAxes,
        fontsize=10,
        horizontalalignment="left",
        verticalalignment="top",
        bbox=props,
    )


def plot_trips(trips, ax):
    """
    Adds trips information to a subplot
    """
    props = dict(boxstyle="round", facecolor="wheat", pad=0.5, alpha=0.5)

    trips_string = ""

    for trip in trips:
        start, end, scooter = trip
        trips_string += f"{start} -> {end}: {scooter}\n"

    ax.text(
        0,
        0.98,
        trips_string,
        transform=ax.transAxes,
        fontsize=10,
        horizontalalignment="left",
        verticalalignment="top",
        bbox=props,
    )


def make_graph(
    coordinates: [(float, float)],
    location_ids: [int],
    depot_ids=None,
    current_location=None,
    next_location=None,
):
    """
    Makes a networkx graph of a list of locations and uses cluster id to give the locations color
    Location/coordinates can both be clusters and scooters
    """
    cartesian_clusters = convert_geographic_to_cart(coordinates, GEOSPATIAL_BOUND_NEW)
    depot_ids = depot_ids if depot_ids else []
    # make graph object
    graph = nx.DiGraph()
    graph.add_nodes_from([c for c in np.arange(len(cartesian_clusters))])

    # set node label and position in graph
    labels = {}
    node_color = []
    node_border = []
    for i, cartesian_cluster_coordinates in enumerate(cartesian_clusters):
        labels[i] = location_ids[i]
        border_color = BLUE if current_location == location_ids[i] else BLACK
        node_color.append(
            BLUE if depot_ids.__contains__(location_ids[i]) else COLORS[location_ids[i]]
        )
        node_border.append(RED if next_location == location_ids[i] else border_color)
        graph.nodes[i]["pos"] = cartesian_cluster_coordinates

    return graph, labels, node_border, node_color


def convert_geographic_to_cart(
    coordinates: [(float, float)], bound: (float, float, float, float)
) -> [(int, int)]:
    """
    Converts (lon,lat) -> ([0,1], [0,1]) give the bounds of lat/lon
    :param coordinates: list of coordinates to be converted
    :param bound: lat/lon bounds
    :return: list of cartesian coordinates on the interval [0,1]
    """
    lat_min, lat_max, lon_min, lon_max = bound
    delta_lat = lat_max - lat_min
    delta_lon = lon_max - lon_min
    zero_lat = lat_min / delta_lat
    zero_lon = lon_min / delta_lon

    output = []

    for lat, lon in coordinates:
        y = lat / delta_lat - zero_lat
        x = lon / delta_lon - zero_lon

        output.append((x, y))

    return output


def choose_label_alignment(start: int, end: int, pos: dict):
    """
    Helper method to set the alignment of label above or beyond an edge
    """
    start_pos = pos[start]
    end_pos = pos[end]

    if start_pos[0] <= end_pos[0]:
        return "top"
    else:
        return "bottom"


def edge_label(start: int, end: int, pos: dict, number_of_trip: int):
    """
    Construct edge label depending on the direction of the edge arrow
    """
    start_pos = pos[start]
    end_pos = pos[end]

    if start_pos[0] <= end_pos[0]:
        return f"{start} --> {end}: {number_of_trip}"
    else:
        return f"{number_of_trip} : {end} <-- {start}"


def create_standard_state_plot():
    """
    Creates standard subplot for a state with image of Oslo as background
    """
    fig = plt.figure(figsize=(20, 9.7))

    # creating subplots
    spec = gridspec.GridSpec(figure=fig, ncols=1, nrows=1)
    ax = fig.add_subplot(spec[0])
    ax.axis("off")

    oslo = plt.imread("images/kart_oslo.png")
    ax.imshow(
        oslo,
        zorder=0,
        extent=(0, 1, 0, 1),
        aspect="auto",
        alpha=0.8,
    )

    return fig, ax


def create_three_subplot_fig(titles=["", "", ""], fig_title=""):
    """
    Subplot framework for the simulation visualization
    """

    # generate plot and subplots
    fig = plt.figure(figsize=(20, 9.7))

    fig.suptitle(fig_title, fontsize=16)

    # creating subplots
    spec = gridspec.GridSpec(
        figure=fig, ncols=3, nrows=1, width_ratios=[1, 8, 8], wspace=0, hspace=0
    )

    return (fig, *create_subplots_from_gripspec(fig, spec, titles))


def create_two_subplot_fig(titles=["", ""], fig_title=""):
    """
    Subplot framework for the simulation visualization
    """

    # generate plot and subplots
    fig = plt.figure(figsize=(20, 9.7))

    fig.suptitle(fig_title, fontsize=16)

    # creating subplots
    spec = gridspec.GridSpec(
        figure=fig, ncols=2, nrows=1, width_ratios=[1, 16], wspace=0, hspace=0
    )

    return (fig, *create_subplots_from_gripspec(fig, spec, titles))


def create_subplots_from_gripspec(fig, spec, titles):
    """
    Create subplot with normalized axis
    """
    subplots = []
    oslo = plt.imread("images/kart_oslo.png")
    for i in range(spec._ncols):
        ax = fig.add_subplot(spec[i])
        ax.set_title(titles[i])
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.axis("off")
        if i > 0:
            ax.imshow(
                oslo,
                zorder=0,
                extent=(0, 1, 0, 1),
                aspect="auto",
                alpha=0.8,
            )
        subplots.append(ax)

    return subplots


def create_plot_with_axis_labels(fig, spec, x_label, y_label, plot_title):
    """
    Creates subplot with axis label and plot title
    """
    ax = fig.add_subplot(spec)
    ax.set_xlabel(x_label, labelpad=10, fontsize=12)
    ax.set_ylabel(y_label, labelpad=10, fontsize=12)
    ax.set_title(plot_title, fontsize=14)
    return ax


def add_cluster_info(state, graph, ax):
    """
    Adds cluster info to a subplot (info i set right above the node/scatter)
    """
    pos = nx.get_node_attributes(graph, "pos")
    # add number of scooters and battery label to nodes
    for i, cluster in enumerate(state.stations):
        node_info = f"S = {cluster.number_of_scooters()}\n"
        x, y = pos[i]
        ax.annotate(
            node_info, xy=(x, y + 0.03), horizontalalignment="center", fontsize=12
        )


def add_flow_edges(graph, flows):
    """
    Adds edge flow to the networkx graph
    """
    pos = nx.get_node_attributes(graph, "pos")
    edge_labels = {}
    alignment = []
    # adding edges
    for start, end, number_of_trips in flows:
        if number_of_trips > 0:
            graph.add_edge(start, end, color=BLACK, width=2)
            alignment.append(choose_label_alignment(start, end, pos))
            edge_labels[(start, end)] = edge_label(start, end, pos, number_of_trips)

    return edge_labels, alignment


def add_vehicle_routes(
    graph, node_border, vehicles, current_vehicle_id, next_location=None
):
    pos = nx.get_node_attributes(graph, "pos")
    route_labels = {}
    alignment = []
    for vehicle in vehicles:
        route = vehicle.service_route
        for i in range(len(route) - 1):
            start, end = route[i][0].id, route[i][1].next_location
            if i != len(route) - 1 or vehicle.id == current_vehicle_id:
                graph.add_edge(start, end, color=COLORS[vehicle.id], width=2)
                if not route_labels.keys().__contains__((start, end)):
                    route_labels[(start, end)] = f"{i}"
                else:
                    route_labels[(start, end)] = route_labels[(start, end)] + f", {i}"
                alignment.append(choose_label_alignment(start, end, pos))
            else:
                graph.add_edge(start, end, color=RED, width=2)

        if vehicle.id == current_vehicle_id and len(route):
            node_border[vehicle.current_location.id] = BLUE
            node_border[next_location] = RED
            graph.add_edge(
                vehicle.current_location.id, next_location, color=RED, width=3
            )

    return route_labels, alignment


def add_scooter_id_and_battery(scooters, graph, ax, scooter_battery=False):
    """
    Adds scooter id above the node/scatter
    """
    pos = nx.get_node_attributes(graph, "pos")

    for i, scooter in enumerate(scooters):
        x, y = pos[i]
        ax.text(x, y + 0.015, f"{scooter.id}", horizontalalignment="center", fontsize=8)
        if scooter_battery:
            ax.text(
                x,
                y - 0.02,
                f"B - {round(scooter.battery,1)}",
                horizontalalignment="center",
                fontsize=8,
            )


def alt_draw_networkx_edge_labels(
    G,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=1.0,
    bbox=None,
    ax=None,
    rotate=True,
    **kwds,
):
    """
    New method to plot edge labels with different alignment for every edge
    """
    pos = nx.get_node_attributes(G, "pos")
    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = dict(((u, v), d) for u, v, d in G.edges(data=True))
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )

        if rotate:
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360  # degrees
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(
                boxstyle="round",
                ec=(1.0, 1.0, 1.0),
                fc=(1.0, 1.0, 1.0),
            )
        if str(label) != label:
            label = str(label)  # this will cause "1" and 1 to be labeled the same

        # set optional alignment
        horizontalalignment = kwds.get("horizontalalignment", "center")
        if horizontalalignment is list:
            horizontalalignment = horizontalalignment[
                list(labels.values()).index(label)
            ]

        verticalalignment = kwds.get("verticalalignment", "center")[
            list(labels.values()).index(label)
        ]
        if verticalalignment is list:
            verticalalignment = verticalalignment[list(labels.values()).index(label)]

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=10,
            clip_on=True,
        )
        text_items[(n1, n2)] = t

    return text_items


def setup_cluster_visualize(
    state,
    current_location_id=None,
    next_location_id=None,
    fig=None,
    ax=None,
):
    node_size = 1000
    font_size = 14

    if not fig and not ax:
        # if subplot isn't specified, construct it
        fig, ax = create_standard_state_plot()

    # constructs the networkx graph from cluster location and with cluster id
    graph, labels, node_border, node_color = make_graph(
        [(location.get_location()) for location in state.locations],
        [location.id for location in state.locations],
        depot_ids=[depot.id for depot in state.depots],
        current_location=current_location_id,
        next_location=next_location_id,
    )

    # adds cluster info (#scooters and tot battery) on plot
    add_cluster_info(state, graph, ax)

    # displays plot
    display_graph(graph, node_color, node_border, node_size, labels, font_size, ax)

    return graph, fig, ax, graph, labels, node_border, node_color, node_size, font_size


def make_scooter_visualize(state, ax, scooter_label=False):
    node_size = 50
    font_size = 10
    # make a list of all scooters
    all_scooters = list(
        itertools.chain.from_iterable(
            map(lambda cluster: cluster.scooters, state.stations)
        )
    )

    # list of all scooter ids for the scooter label plot
    all_scooters_id = [scooter.id for scooter in all_scooters]

    # list of all cluster ids associated with scooters (so we can get the right color of the scooter nodes)
    all_cluster_ids = list(
        itertools.chain.from_iterable(
            [cluster.id] * len(cluster.scooters) for cluster in state.stations
        )
    )

    # constructs the networkx graph from cluster location, second input is for color purpose
    graph, labels, node_border, node_color = make_graph(
        [scooter.get_location() for scooter in all_scooters],
        all_cluster_ids,
    )

    if scooter_label:
        # add scooter id as label above each node in plot
        add_scooter_id_and_battery(all_scooters, graph, ax)

    # display graph
    display_graph(
        graph,
        node_color,
        node_border,
        node_size,
        labels,
        font_size,
        ax,
        with_labels=False,
    )

    return (
        graph,
        node_color,
        node_border,
        node_size,
        labels,
        font_size,
        all_scooters,
        all_scooters_id,
    )


def add_location_center(locations, ax):

    cluster_locations = convert_geographic_to_cart(
        [location.get_location() for location in locations], GEOSPATIAL_BOUND_NEW
    )

    for location in locations:
        center_x, center_y = cluster_locations[location.id]
        ax.scatter(
            center_x,
            center_y,
            c=BLUE if isinstance(location, sim.Depot) else COLORS[location.id],
            edgecolor="None",
            alpha=0.8,
            s=200,
            zorder=10,
        )
        ax.annotate(
            location.id,
            (center_x, center_y),
            ha="center",
            va="center",
            weight="bold",
            zorder=11,
        )


def plot_smoothed_curve(x, y, ax, color, label, z_order=1):
    x, y = post_process_curve(x, y)
    x_smooth = np.linspace(np.min(x), np.max(x), 1000)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)
    ax.plot(x_smooth, y_smooth, c=color, label=label, zorder=z_order)


def post_process_curve(x, y):
    start = x[0]
    start_index = 0
    value = 0
    x_new, y_new = [], []
    for i, time in enumerate(x):
        if start == time and i != len(x) - 1:
            value += y[i]
        else:
            x_new.append(start)
            y_new.append(value / (i - start_index))
            start = time
            start_index = i
            value = y[i]

    return x_new, y_new


def get_policy_label(policy):
    if hasattr(policy, "roll_out_policy") and hasattr(
        policy.roll_out_policy, "value_function"
    ):
        return (
            f"Rollout: {policy.roll_out_policy}\n"
            f"w/{policy.roll_out_policy.value_function} -t{policy.roll_out_policy.value_function.shifts_trained}"
        )
    elif hasattr(policy, "value_function"):
        return (
            f"Rollout: {policy}\n"
            f"w/{policy.value_function} -t{policy.value_function.shifts_trained}"
        )
    else:
        return f"{policy}"
