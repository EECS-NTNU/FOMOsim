"""
Methods for visualizing different aspects of the system.
"""


from sim import Action, Scooter, State, Vehicle
from visualization.helpers import *
import matplotlib.pyplot as plt
import copy
from itertools import cycle


def visualize_clustering(clusters):
    """
    Associates a color to every cluster and gives it an id. Then plot all scooter with map in background
    :param clusters: cluster objects
    """
    fig, ax = plt.subplots(figsize=[10, 6])

    # Add image to background
    oslo = plt.imread("images/kart_oslo.png")
    lat_min, lat_max, lon_min, lon_max = GEOSPATIAL_BOUND_NEW
    ax.imshow(
        oslo,
        zorder=0,
        extent=(lon_min, lon_max, lat_min, lat_max),
        aspect="auto",
        alpha=0.6,
    )
    colors = cycle("bgrcmyk")
    # Add clusters to figure
    for cluster in clusters:
        scooter_locations = [
            (scooter.get_lat(), scooter.get_lon()) for scooter in cluster.scooters
        ]
        cluster_color = next(colors)
        df_scatter = ax.scatter(
            [lon for lat, lon in scooter_locations],
            [lat for lat, lon in scooter_locations],
            c=cluster_color,
            alpha=0.6,
            s=3,
        )
        center_lat, center_lon = cluster.get_location()
        rs_scatter = ax.scatter(
            center_lon,
            center_lat,
            c=cluster_color,
            edgecolor="None",
            alpha=0.8,
            s=200,
        )
        ax.annotate(
            cluster.id,
            (center_lon, center_lat),
            ha="center",
            va="center",
            weight="bold",
        )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if len(clusters) > 0:
        # Legend will use the last cluster color. Check for clusters to avoid None object
        ax.legend(
            [df_scatter, rs_scatter],
            ["Full dataset", "Cluster centers"],
            loc="upper right",
        )
    plt.show()


def visualize_state(state):
    """
    Visualize the clusters of a state with battery and number of scooters in the clusters
    :param state: State object to be visualized
    """
    setup_cluster_visualize(state)
    # shows the plots in IDE
    plt.tight_layout(pad=1.0)
    plt.show()


def visualize_cluster_flow(state: State, flows: [(int, int, int)]):
    """
    Visualize the flow in a state from a simulation
    :param state: State to display
    :param flows: flow of scooter from one cluster to another
    :return:
    """
    (
        graph,
        fig,
        ax,
        graph,
        labels,
        node_border,
        node_color,
        node_size,
        font_size,
    ) = setup_cluster_visualize(state)

    if flows:
        # adds edges of flow between the clusters
        edge_labels, alignment = add_flow_edges(graph, flows)

        # displays edges on plot
        alt_draw_networkx_edge_labels(
            graph,
            edge_labels=edge_labels,
            verticalalignment=alignment,
            bbox=dict(alpha=0),
            ax=ax,
        )

    # displays plot
    display_graph(graph, node_color, node_border, node_size, labels, font_size, ax)

    # shows the plots in IDE
    plt.tight_layout(pad=1.0)
    plt.show()


def visualize_vehicle_routes(
    state,
    current_vehicle_id=None,
    current_location_id=None,
    next_location_id=None,
    policy="",
):
    """
    Visualize the vehicle route in a state from a simulation
    :param policy: name of current policy
    :param current_location_id: vehicles current location id
    :param state: State to display
    :param current_vehicle_id: current vehicle at a cluster
    :param next_location_id: id of next state
    :return:
    """
    fig, ax1, ax2 = create_two_subplot_fig(
        titles=[
            "Tabu list",
            f"Vehicle {current_vehicle_id} arriving at location {current_location_id} and heading to location {next_location_id}",
        ],
        fig_title=policy.__str__(),
    )

    (
        graph,
        fig,
        ax,
        graph,
        labels,
        node_border,
        node_color,
        node_size,
        font_size,
    ) = setup_cluster_visualize(
        state, current_location_id, next_location_id, fig=fig, ax=ax2
    )

    if current_vehicle_id or current_vehicle_id == 0:
        route_labels, alignment = add_vehicle_routes(
            graph, node_border, state.vehicles, current_vehicle_id, next_location_id
        )

        alt_draw_networkx_edge_labels(
            graph,
            edge_labels=route_labels,
            verticalalignment=alignment,
            bbox=dict(alpha=0),
            ax=ax2,
        )

    # displays plot
    display_graph(graph, node_color, node_border, node_size, labels, font_size, ax2)

    func = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [func("_", COLORS[i]) for i in range(len(state.vehicles))]
    legends = [f"Vehicle {vehicle.id}" for vehicle in state.vehicles]
    ax2.legend(handles, legends, framealpha=1)

    # shows the plots in IDE
    plt.tight_layout(pad=1.0)
    plt.show()


def visualize_action(
    state_before_action: State,
    vehicle_before_action: Vehicle,
    current_state: State,
    current_vehicle: Vehicle,
    action: Action,
    world_time: float,
    action_time: float,
    scooter_label=True,
    policy="",
):
    # creating the subplots for the visualization
    fig, ax1, ax2, ax3 = create_three_subplot_fig(
        titles=["Action", "State before action", "State after action"],
        fig_title=policy.__str__(),
    )

    # plots the vehicle info and the action in the first plot
    plot_vehicle_info(vehicle_before_action, current_vehicle, ax1)
    plot_action(
        action,
        vehicle_before_action.current_location.id,
        world_time,
        action_time,
        ax1,
        offset=(
            len(vehicle_before_action.scooter_inventory)
            + len(current_vehicle.scooter_inventory)
        )
        * ACTION_OFFSET,
    )

    make_scooter_visualize(state_before_action, ax2, scooter_label=scooter_label)
    add_location_center(state_before_action.locations, ax2)

    make_scooter_visualize(current_state, ax3, scooter_label=scooter_label)
    add_location_center(state_before_action.locations, ax3)

    plt.tight_layout(pad=1.0)
    plt.show()


def visualize_scooters_on_trip(current_state: State, trips: [(int, int, Scooter)]):
    fig, ax1, ax2 = create_two_subplot_fig(["Current trips", "State"])

    plot_trips(trips, ax1)

    make_scooter_visualize(current_state, ax2, scooter_battery=True)

    add_location_center(current_state.locations, ax2)

    plt.tight_layout(pad=1.0)
    plt.show()


def visualize_scooter_simulation(
    current_state: State,
    trips,
):
    """
    Visualize scooter trips of one system simulation
    :param current_state: Initial state for the simulation
    :param trips: trips completed during a system simulation
    """

    # creating the subplots for the visualization
    fig, ax1, ax2, ax3 = create_three_subplot_fig(
        ["Trips", "Current state", "Next State"]
    )

    plot_trips(trips, ax1)

    (
        graph,
        node_color,
        node_border,
        node_size,
        labels,
        font_size,
        all_current_scooters,
        all_current_scooters_id,
    ) = make_scooter_visualize(current_state, ax2, scooter_label=True)

    # have to copy the networkx graph since the plot isn't shown in the IDE yet
    next_graph = copy.deepcopy(graph)

    # convert location of the scooter that has moved during the simulation
    cartesian_coordinates = convert_geographic_to_cart(
        [scooter.get_location() for star, end, scooter in trips], GEOSPATIAL_BOUND_NEW
    )

    number_of_current_scooters = len(all_current_scooters)

    # adds labels to the new subplot of the scooters from the state before simulation
    add_scooter_id_and_battery(
        all_current_scooters, next_graph, ax3, scooter_battery=True
    )

    # loop to add nodes/scooters that have moved during a simulation
    for i, trip in enumerate(trips):
        start, end, scooter = trip
        x, y = cartesian_coordinates[i]
        previous_label = all_current_scooters_id.index(scooter.id)

        # add new node
        next_graph.add_node(number_of_current_scooters + i)
        # adds location in graph for the new node
        next_graph.nodes[number_of_current_scooters + i]["pos"] = (x, y)

        # adds label and color of new node
        labels[number_of_current_scooters + i] = previous_label
        node_color.append(COLORS[end.id])
        node_border.append(BLACK)

        # set the previous position of the scooter to a white node
        node_color[previous_label] = "white"

        # add edge from previous location of scooter to current
        next_graph.add_edge(
            previous_label, number_of_current_scooters + i, color=BLACK, width=1
        )

        # display label on subplot
        ax3.text(
            x, y + 0.015, f"{scooter.id}", horizontalalignment="center", fontsize=8
        )

        ax3.text(
            x,
            y - 0.02,
            f"B - {round(scooter.battery, 1)}",
            horizontalalignment="center",
            fontsize=8,
        )

    display_graph(
        next_graph,
        node_color,
        node_border,
        node_size,
        labels,
        font_size,
        ax3,
        with_labels=False,
    )

    plt.tight_layout(pad=1.0)
    plt.show()


def visualize_analysis(instances, title=None):
    """
    :param instances: world instances to analyse
    :param title: plot title
    :return: plot for the analysis
    """
    # generate plot and subplots
    #     fig = plt.figure(figsize=(20, 9.7))
    fig = plt.figure(figsize=(8, 6))

    # creating subplots
    spec = gridspec.GridSpec(figure=fig, ncols=2, nrows=2)

    subplots_labels = [
        ("Time", "Number of lost trips", " Lost demand"),
        ("Time", "Number of available scooters", "Total available scooters"),
        ("Time", "Average battery (%)", "Average battery"),
    ]
    # figure
    subplots = []
    for i, (x_label, y_label, plot_title) in enumerate(subplots_labels):
        ax = create_plot_with_axis_labels(
            fig,
            spec[i],
            x_label=x_label,
            y_label=y_label,
            plot_title=plot_title,
        )
        subplots.append(ax)

    ax1, ax2, ax4 = subplots
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")

    for i, instance in enumerate(instances):
        metrics = instance.metrics.get_all_metrics()

        if len(metrics.get("lost_demand", [])) > 0:
            x = [item[0] for item in metrics["lost_demand"]]
            x.append(instance.metrics.max_time)
            y = []
            last = 0
            for item in metrics["lost_demand"]:
                y.append(last + item[1])
                last += item[1]
            y.append(last)
            
            ax1.plot(x, y, c=COLORS[i], label=instance.label)
        if len(metrics["total_available_scooters"]) > 0:
            x = [item[0] for item in metrics["total_available_scooters"]]
            y = [item[1] for item in metrics["total_available_scooters"]]
            ax2.plot(x, y, c=COLORS[i], label=instance.label)
        if len(metrics["average_battery"]) > 0:
            x = [item[0] for item in metrics["average_battery"]]
            y = [item[1] for item in metrics["average_battery"]]
            ax4.plot(x, y, c=COLORS[i], label=instance.label)

    for subplot in subplots:
        subplot.legend()
        subplot.set_ylim(ymin=0)
    if title is not None:
        fig.suptitle(
            title,
            fontsize=16,
        )

    fig.tight_layout()

    plt.show()

    return fig


def visualize_loss(losses: [[float]]):
    # generate plot and subplots
    fig = plt.figure(figsize=(20, 9.7))

    # creating subplots
    spec = gridspec.GridSpec(
        figure=fig, ncols=1, nrows=1, width_ratios=[1], wspace=0.2, hspace=0
    )

    ax = create_plot_with_axis_labels(
        fig,
        spec[0],
        x_label="Epoch",
        y_label="Loss",
        plot_title="",
    )

    x = np.arange(len(losses))
    ax.plot(x, losses, color=COLORS[0])

    fig.suptitle(
        f"Loss development",
        fontsize=16,
    )

    ax.set_xlim(xmin=0)

    plt.show()

    return fig


def heatmap():
    import folium
    import sim
    import init_state.entur.scripts
    from folium.plugins import HeatMap

    map_hooray = folium.Map(location=[59.925586, 10.730721], zoom_start=13)

    world_to_analyse = sim.World(
        960,
        None,
        clustering.scripts.get_initial_state(
            2500,
            50,
            number_of_vans=2,
        ),
        verbose=False,
        visualize=False,
        MODELS_TO_BE_SAVED=5,
        TRAINING_SHIFTS_BEFORE_SAVE=50,
        ANN_LEARNING_RATE=0.0001,
        ANN_NETWORK_STRUCTURE=[1000, 2000, 1000, 200],
        REPLAY_BUFFER_SIZE=100,
        test_parameter_name="dr_ltr",
    )
    percentage = 1500 / 2500
    all_coordinates = []
    for cluster in world_to_analyse.state.stations:
        cluster.scooters = cluster.scooters[: round(len(cluster.scooters) * percentage)]
        for scooter in cluster.scooters:
            all_coordinates.append(scooter.get_location())

    HeatMap(all_coordinates).add_to(map_hooray)

    # Display the map
    map_hooray.save("heatmap.HTML")
