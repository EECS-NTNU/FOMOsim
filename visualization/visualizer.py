"""
Methods for visualizing different aspects of the system.
"""

import datetime
from sim import Action, Scooter, State, Vehicle, Metric
import matplotlib.pyplot as plt
import copy
from itertools import cycle
import matplotlib.dates as mdates
from matplotlib import gridspec

from settings import COLORS

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

def visualize_lost_demand(instances, title=None, week=1):
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
        ("Time", "Number of lost trips", " Lost demand"),
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
        if "lost_demand" in metric.metrics:
            x.extend([totime(item[0], startdate) for item in metric.metrics["lost_demand"]])
            y.extend([item[1] for item in metric.metrics["lost_demand"]])
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


def save_lost_demand_csv(instances, title=None, week=1):
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
        if "lost_demand" in metric.metrics:
            x.extend([totime(item[0], startdate) for item in metric.metrics["lost_demand"]])
            y.extend([item[1] for item in metric.metrics["lost_demand"]])
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
        ("Num vans", "Lost trips", "Starvation"),
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

        if "lost_demand" in metric.metrics:
            yvalues_ld.append(metric.metrics["lost_demand"][-1][1])
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
