"""
Methods for visualizing different aspects of the system.
"""

import datetime
from sim import Action, EBike, State, Vehicle, Metric
import matplotlib.pyplot as plt
import copy
import matplotlib.dates as mdates
from matplotlib import gridspec

def totime(ts):
    weektext = "2022 " + str(1) + " 1 00:00"
    startdate=datetime.datetime.strptime(weektext, "%Y %W %w %H:%M")

    return datetime.datetime.fromtimestamp(startdate.timestamp() + ts * 60)

def visualize(metrics, metric="trips"):
    fig, ax = plt.subplots()
    ax.set_xlabel("Time", labelpad=10, fontsize=12)
    ax.set_ylabel("Number", labelpad=10, fontsize=12)
    ax.set_title(metric, fontsize=14)

    if type(metrics) is list:
        metricObj = Metric.merge_metrics(metrics)
    else:
        metricObj = metrics

    xx = []
    yy = []

    # add start point
    if metricObj.isAggregate(metric):
        xx.append(totime(metricObj.min_time))
        yy.append(0)

    # add main points
    if metric in metricObj.metrics:
        xx.extend([totime(item[0]) for item in metricObj.metrics[metric]])
        yy.extend([item[1] for item in metricObj.metrics[metric]])

    # add end point
    if metricObj.isAggregate(metric):
        xx.append(totime(metricObj.max_time))
        yy.append(yy[-1])

    ax.plot(xx, yy)

    ax.set_ylim(ymin=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%a %H:%M"))

    plt.show()

    return fig
