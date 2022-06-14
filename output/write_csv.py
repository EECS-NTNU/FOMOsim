"""
Methods for visualizing different aspects of the system.
"""

import datetime
from progress.bar import Bar

import sim

def totime(ts, startdate):
    return datetime.datetime.fromtimestamp(startdate.timestamp() + ts * 60)

def write_csv(instances, filename, week, hourly=False):
    f = open(filename, "w")

    if type(instances) is list:
        metric = sim.Metric.merge_metrics([instance.metrics for instance in instances])
    else:
        metric = instances.metrics
    
    weektext = "2022 " + str(week) + " 1 00:00"
    startdate=datetime.datetime.strptime(weektext, "%Y %W %w %H:%M")

    keys = metric.metrics.keys()

    f.write("Time;")
    for key in keys:
        f.write(key + ";" + key + "_norm" + ";")
    f.write("\n")

    progress = Bar(
        "Saving results",
        max = len(metric.timeline()),
    )

    idx = {}
    for m in keys:
        idx[m] = 0
    tripIdx = 0

    last_hour = 0

    for time in metric.timeline():
        if not hourly or ((time // 60) > last_hour):
            last_hour = time // 60
            f.write(str(totime(time, startdate)) + ";")
            trips, tripIdx = metric.getValue("trips", tripIdx, time)
            for m in keys:
                value, idx[m] = metric.getValue(m, idx[m], time)
                if value is None:
                    f.write(";;")
                else:
                    if trips != 0 and trips is not None:
                        f.write(str(value) + ";" + str(value / trips) + ";")
                    else:
                        f.write(str(value) + ";;")
            f.write("\n")

        progress.next()

    progress.finish()

    f.close()
        
