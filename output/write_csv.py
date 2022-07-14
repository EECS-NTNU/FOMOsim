import datetime
from progress.bar import Bar

import sim

def totime(ts, startdate):
    return datetime.datetime.fromtimestamp(startdate.timestamp() + ts * 60)

# TODO: needs cleanup
def write_csv(instances, filename, week=None, hourly=False, mode="w", parameters=None):
    f = open(filename, mode)

    if type(instances) is list:
        metric = sim.Metric.merge_metrics([instance.metrics for instance in instances])
    else:
        metric = instances.metrics
    
    if week is not None:
        weektext = "2022 " + str(week) + " 1 00:00"
        startdate=datetime.datetime.strptime(weektext, "%Y %W %w %H:%M")

    keys = sorted(metric.metrics.keys())

    if parameters is not None:
        for key, _ in parameters:
            f.write(key + ";")

    f.write("Time;")
    for key in keys:
        f.write(key + ";")
    f.write("\n")

    progress = Bar(
        "Saving results",
        max = len(metric.timeline()),
    )

    idx = {}
    for m in keys:
        idx[m] = 0

    last_hour = 0

    for time in metric.timeline():
        if not hourly or ((time // 60) > last_hour):
            last_hour = time // 60

            if parameters is not None:
                for _, value in parameters:
                    f.write(str(value) + ";")

            if week is not None:
                f.write(str(totime(time, startdate)) + ";")
            else:
                if hourly:
                    f.write(str(last_hour) + ";")
                else:
                    f.write(str(time) + ";")

            for m in keys:
                value, idx[m] = metric.getValue(m, idx[m], time)
                if value is None:
                    f.write(";")
                else:
                    f.write(str(value) + ";")
            f.write("\n")

        progress.next()

    f.write("\n")
    f.close()
        
    progress.finish()
