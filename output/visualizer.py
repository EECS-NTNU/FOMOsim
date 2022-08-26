"""
Methods for visualizing different aspects of the system.
"""

import datetime
from sim import Action, EBike, State, Vehicle, Metric
import matplotlib.pyplot as plt
import copy
import matplotlib.dates as mdates
from matplotlib import gridspec

COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
    "#000000",
    "#207188",
    "#86aa6d",
    "#db494c",
    "#34393c",
    "#eba432",
    "#c2c1bc",
    "#A4847E",
    "#056672",
    "#D8FB27",
    "#BBE5D1",
    "#FEEB81",
    "#126027",
    "#7666E7",
    "#530788",
    "#A281ED",
    "#954701",
    "#B42760",
    "#F0E466",
    "#A32315",
    "#4886E8",
    "#117427",
    "#A3A66A",
    "#F124AC",
    "#4572BD",
    "#93EB5F",
    "#ECDCCD",
    "#48317F",
    "#DF8547",
    "#1DE961",
    "#5BD669",
    "#4FAA9B",
    "#937016",
    "#840FF6",
    "#3EAEFD",
    "#F6F34D",
    "#015133",
    "#59025B",
    "#F03B29",
    "#53A912",
    "#34058C",
    "#FA928D",
    "#3C70C3",
    "#AB9869",
    "#B6BD37",
    "#693C24",
    "#2588F7",
    "#54B006",
    "#6604CE",
    "#4A4329",
    "#0175B1",
    "#177982",
    "#544FAD",
    "#DD5409",
    "#583ED1",
    "#CD9D69",
    "#6B0BCE",
    "#D14B12",
    "#96725D",
    "#BB137F",
    "#7B53B5",
    "#BFFB24",
    "#F9D08F",
    "#CF03B8",
    "#A6F591",
    "#D7CFDB",
    "#2D4AD6",
    "#BC5286",
    "#6245C8",
    "#E40EB7",
    "#E2DA97",
    "#EE5089",
    "#CAF026",
    "#668981",
    "#8E424B",
    "#49633D",
    "#8A4CE4",
    "#827C33",
    "#35EFF2",
    "#325041",
    "#2BC23F",
    "#44857A",
    "#DA0043",
    "#87A43F",
    "#D4FCEC",
    "#9FD87C",
    "#0D36DF",
    "#241B73",
    "#524526",
    "#163F53",
    "#4C9B58",
    "#00F4DB",
    "#20054B",
    "#82026F",
    "#CA561D",
    "#F94B06",
    "#5CCBDB",
    "#8B6882",
    "#9C28B0",
    "#15357B",
    "#BB00F4",
    "#451918",
    "#B94AE1",
    "#698290",
    "#415697",
    "#61B95D",
    "#957BD8",
    "#01A1C5",
    "#69E54F",
    "#D40C21",
    "#08A810",
    "#05ECC3",
    "#8FA2B5",
    "#D45A2C",
    "#1689EA",
    "#7DD21F",
    "#A615B6",
    "#430E4C",
    "#557F16",
    "#68E3A4",
    "#E19180",
    "#8B0197",
    "#7314C4",
    "#A397DA",
    "#175ACE",
    "#6185AD",
    "#D981A8",
    "#984ED3",
    "#37FFF0",
    "#90BB50",
    "#A818B0",
    "#28F263",
    "#700EA8",
    "#5C0D3A",
    "#CAF06F",
    "#815F36",
    "#CCF509",
    "#21C91D",
    "#D09B45",
    "#282AF6",
    "#053525",
    "#0FAE75",
    "#213E02",
    "#1572AA",
    "#9D9A3A",
    "#1C1DA9",
    "#C6A728",
    "#0BE59B",
    "#272CAF",
    "#75BA93",
    "#E29981",
    "#45F101",
    "#D8BA19",
    "#BF7545",
    "#0F85B1",
    "#E6DC7B",
    "#6B6548",
    "#78B075",
    "#AFDF4D",
    "#D0BD94",
    "#C6F81B",
    "#27C209",
    "#3C6574",
    "#2CE0B3",
    "#9C6E06",
    "#53CECD",
    "#A5EC06",
    "#AA83D6",
    "#7705D2",
    "#806015",
    "#881E9E",
    "#617730",
    "#1F9ACF",
    "#8AE30F",
    "#D1E1B4",
    "#D924F6",
    "#5FE267",
    "#6BDDF2",
    "#5E40A5",
    "#9B1580",
    "#B6E49C",
    "#619C46",
    "#504BDE",
]

def totime(ts, startdate):
    return datetime.datetime.fromtimestamp(startdate.timestamp() + ts * 60)

def create_plot_with_axis_labels(fig, spec, x_label, y_label, plot_title):
    """
    Creates subplot with axis label and plot title
    """
    ax = fig.add_subplot(spec)
    ax.set_xlabel(x_label, labelpad=10, fontsize=12)
    ax.set_ylabel(y_label, labelpad=10, fontsize=12)
    ax.set_title(plot_title, fontsize=14)
    return ax

def visualize_trips(instances, title=None, week=1):
    """
    :param instances: world instances to analyse
    :param title: plot title
    :return: plot for the analysis
    """
    # generate plot and subplots
    #     fig = plt.figure(figsize=(20, 9.7))
    fig = plt.figure(figsize=(8, 6))

    # creating subplots
    spec = gridspec.GridSpec(figure=fig, ncols=1, nrows=1)

    subplots_labels = [
        ("Time", "Number of trips", "Trips"),
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

    ax1 = subplots[0]

    weektext = "2022 " + str(week) + " 1 00:00"
    startdate=datetime.datetime.strptime(weektext, "%Y %W %w %H:%M")

    for i, insts in enumerate(instances):
        if type(insts) is list:
            metric = Metric.merge_metrics([instance.metrics for instance in insts])
            label = insts[0].label
        else:
            metric = insts.metrics
            label = insts.label

        x = [totime(metric.min_time, startdate)]
        y = [0]
        if "trips" in metric.metrics:
            x.extend([totime(item[0], startdate) for item in metric.metrics["trips"]])
            y.extend([item[1] for item in metric.metrics["trips"]])
        x.append(totime(metric.max_time, startdate))
        y.append(y[-1])
        ax1.plot(x, y, c=COLORS[i], label=label)

    for subplot in subplots:
        subplot.legend()
        subplot.set_ylim(ymin=0)
        subplot.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
        subplot.xaxis.set_minor_formatter(mdates.DateFormatter("%a %H:%M"))
    if title is not None:
        fig.suptitle(
            title,
            fontsize=16,
        )

    plt.show()

    return fig


def visualize_starvation(instances, title=None, week=1):
    """
    :param instances: world instances to analyse
    :param title: plot title
    :return: plot for the analysis
    """
    # generate plot and subplots
    #     fig = plt.figure(figsize=(20, 9.7))
    fig = plt.figure(figsize=(8, 6))

    # creating subplots
    spec = gridspec.GridSpec(figure=fig, ncols=1, nrows=1)

    subplots_labels = [
        ("Time", "Number of lost trips", "Starvation"),
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

    ax1 = subplots[0]

    weektext = "2022 " + str(week) + " 1 00:00"
    startdate=datetime.datetime.strptime(weektext, "%Y %W %w %H:%M")

    for i, insts in enumerate(instances):
        if type(insts) is list:
            metric = Metric.merge_metrics([instance.metrics for instance in insts])
            label = insts[0].label
        else:
            metric = insts.metrics
            label = insts.label

        # lost demand
        x = [totime(metric.min_time, startdate)]
        y = [0]
        if "starvation" in metric.metrics:
            x.extend([totime(item[0], startdate) for item in metric.metrics["starvation"]])
            y.extend([item[1] for item in metric.metrics["starvation"]])
        x.append(totime(metric.max_time, startdate))
        y.append(y[-1])
        ax1.plot(x, y, c=COLORS[i], label=label)

    for subplot in subplots:
        subplot.legend()
        subplot.set_ylim(ymin=0)
        subplot.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
        subplot.xaxis.set_minor_formatter(mdates.DateFormatter("%a %H:%M"))
    if title is not None:
        fig.suptitle(
            title,
            fontsize=16,
        )

    plt.show()

    return fig


def visualize_congestion(instances, title=None, week=1):
    """
    :param instances: world instances to analyse
    :param title: plot title
    :return: plot for the analysis
    """
    # generate plot and subplots
    #     fig = plt.figure(figsize=(20, 9.7))
    fig = plt.figure(figsize=(8, 6))

    # creating subplots
    spec = gridspec.GridSpec(figure=fig, ncols=1, nrows=1)

    subplots_labels = [
        ("Time", "Number of trips to full stations", "Congestion"),
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

    ax1 = subplots[0]

    weektext = "2022 " + str(week) + " 1 00:00"
    startdate=datetime.datetime.strptime(weektext, "%Y %W %w %H:%M")

    for i, insts in enumerate(instances):
        if type(insts) is list:
            metric = Metric.merge_metrics([instance.metrics for instance in insts])
            label = insts[0].label
        else:
            metric = insts.metrics
            label = insts.label

        x = [totime(metric.min_time, startdate)]
        y = [0]
        if "congestion" in metric.metrics:
            x.extend([totime(item[0], startdate) for item in metric.metrics["congestion"]])
            y.extend([item[1] for item in metric.metrics["congestion"]])
        x.append(totime(metric.max_time, startdate))
        y.append(y[-1])
        ax1.plot(x, y, c=COLORS[i], label=label)

    for subplot in subplots:
        subplot.legend()
        subplot.set_ylim(ymin=0)
        subplot.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
        subplot.xaxis.set_minor_formatter(mdates.DateFormatter("%a %H:%M"))
    if title is not None:
        fig.suptitle(
            title,
            fontsize=16,
        )

    plt.show()

    return fig


def save_starvation_csv(instances, title=None, week=1):
    """
    :param instances: world instances to analyse
    :param title: plot title
    :return: plot for the analysis
    """
    weektext = "2022 " + str(week) + " 1 00:00"
    startdate=datetime.datetime.strptime(weektext, "%Y %W %w %H:%M")

    for i, insts in enumerate(instances):
        if type(insts) is list:
            metric = Metric.merge_metrics([instance.metrics for instance in insts])
        else:
            metric = insts.metrics

        # lost demand
        x = [totime(metric.min_time, startdate)]
        y = [0]
        if "starvation" in metric.metrics:
            x.extend([totime(item[0], startdate) for item in metric.metrics["starvation"]])
            y.extend([item[1] for item in metric.metrics["starvation"]])
        x.append(totime(metric.max_time, startdate))
        y.append(y[-1])

        print(f"{insts[0].label}_x", end="")
        for el in x:
            print(f";{el}", end="")
        print("")

        print(f"{insts[0].label}_y", end="")
        for el in y:
            print(f";{el}", end="")
        print("")


def visualize_end(instances, xvalues, title=None, week=1):
    """
    :param instances: world instances to analyse
    :param title: plot title
    :return: plot for the analysis
    """
    # generate plot and subplots
    fig = plt.figure(figsize=(8, 6))

    # creating subplots
    spec = gridspec.GridSpec(figure=fig, ncols=1, nrows=1)

    subplots_labels = [
        ("Num vehicles", "Lost trips", "Starvation"),
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

    ax1 = subplots[0]

    weektext = "2022 " + str(week) + " 1 00:00"
    startdate=datetime.datetime.strptime(weektext, "%Y %W %w %H:%M")

    yvalues_ld = []
    yvalues_c = []

    for i, insts in enumerate(instances):
        if type(insts) is list:
            metric = Metric.merge_metrics([instance.metrics for instance in insts])
        else:
            metric = insts.metrics

        if "starvation" in metric.metrics:
            yvalues_ld.append(metric.metrics["starvation"][-1][1])
        else:
            yvalues_ld.append(0)

        if "congestion" in metric.metrics:
            yvalues_c.append(metric.metrics["congestion"][-1][1])
        else:
            yvalues_c.append(0)

    ax1.plot(xvalues, yvalues_ld, c=COLORS[i], label=insts[0].label)

    print("X;Y")
    for i in range(len(xvalues)):
        print(f"{xvalues[i]};{yvalues_ld[i]}")

    for subplot in subplots:
        subplot.legend()
        subplot.set_ylim(ymin=0)
    if title is not None:
        fig.suptitle(
            title,
            fontsize=16,
        )

#    fig.tight_layout()

    plt.show()

    return fig
