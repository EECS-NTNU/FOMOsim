import sys
import numpy as np

class Metric:
    """
    Class for storing and aggregate the metric data of the instance
    """

    def __init__(self):
        self.metrics = {}

        self.min_time = 0
        self.max_time = 0

    def add_metric(self, sim, metric, value):
        if metric not in self.metrics:
            self.metrics[metric] = []

        self.metrics[metric].append((sim.time, value))
        if((self.min_time == 0) or (self.min_time > sim.time)):
            self.min_time = sim.time
        if(sim.time > self.max_time):
            self.max_time = sim.time

    def add_metric_time(self, time, metric, value):
        if metric not in self.metrics:
            self.metrics[metric] = []

        self.metrics[metric].append((time, value))
        if((self.min_time == 0) or (self.min_time > time)):
            self.min_time = time
        if(time > self.max_time):
            self.max_time = time

    def add_aggregate_metric(self, sim, metric, value):
        if metric not in self.metrics:
            self.metrics[metric] = []

        series = self.metrics[metric]
        last_time = -1
        last_value = 0
        if len(series) > 0:
            last_time = series[-1][0]
            last_value = series[-1][1]
        if last_time == sim.time:
            series[-1] = (last_time, last_value + value)
        else:
            series.append((sim.time, last_value + value))

        if((self.min_time == 0) or (self.min_time > sim.time)):
            self.min_time = sim.time
        if(sim.time > self.max_time):
            self.max_time = sim.time

    # analysis metrics added every timestep
    def add_analysis_metrics(self, sim):
        #self.add_metric(sim, "average_battery", sum([bike.battery for bike in sim.state.get_all_bikes() if bike.hasBattery()]) / len(sim.state.get_all_bikes()))
        #self.add_metric(sim, "total_available_bikes", sim.state.get_num_available_bikes())
        pass

    def timeline(self):
        times = []
        for metric in self.metrics.keys():
            times.extend([item[0] for item in self.metrics[metric]])
        return sorted(list(dict.fromkeys(times)))

    def getValue(self, metric, from_idx, time):
        for i in range(from_idx, len(self.metrics[metric])):
            key = self.metrics[metric][i][0]
            if key == time:
                return (self.metrics[metric][i][1], i)
            if key > time:
                if i > 0:
                    return (self.metrics[metric][i-1][1], i)
                else:
                    return (None, i)
        return (self.metrics[metric][i][1], i)

    def values(self, metric):
        return [item[1] for item in metrics[metric]]

    def get_all_metrics(self):
        """
        Returns all metrics recorded for analysis
        """
        return self.metrics

    ###############################################################################

    def get_aggregate_value(self, key):
        if key in self.metrics:
            return self.metrics[key][-1][1]
        else:
            return 0

    def get_n_time(self, time, key):
        pointer = 0
        while True:
            if key not in self.metrics:
                return sys.maxsize
            if len(self.metrics[key]) <= pointer:
                return sys.maxsize
            if self.metrics[key][pointer][0] > time:
                return self.metrics[key][pointer][0]
            pointer += 1

    @staticmethod
    def get_next_time(metrics, time, key):
        next_time = sys.maxsize
        for m in metrics:
            n = m.get_n_time(time, key)
            if n < next_time:
                next_time = n
        return next_time

    @staticmethod
    def get_average(metrics, time, key):
        sum = 0
        for m in metrics:
            sum += m.get_value(time, key)
        return sum / len(metrics)

    @staticmethod
    def merge_metrics(metrics):
        metric = Metric()

        keys = []
        for m in metrics:
            for key in m.metrics.keys():
                if key not in keys:
                    keys.append(key)

        max_time = 0
        for m in metrics:
            if m.max_time > max_time:
                max_time = m.max_time

        for key in keys:
            pointers = [0] * len(metrics)
            
            time = metrics[0].min_time

            while time <= max_time:
                # get value
                value = 0
                num = 0
                for i, m in enumerate(metrics):
                    if (key in m.metrics) and (len(m.metrics[key]) > pointers[i]):
                        v = m.metrics[key][pointers[i]][1]
                        t = m.metrics[key][pointers[i]][0]
                        if(t <= time):
                            value += v
                            num += 1
                if num > 0:
                    value = value / num

                # add metric
                metric.add_metric_time(time, key, value)

                # get next time
                mintime = sys.maxsize
                chosen = 0
                for i, m in enumerate(metrics):
                    if (key in m.metrics) and (len(m.metrics[key]) > (pointers[i]+1)):
                        t = m.metrics[key][pointers[i]+1][0]
                        if(t <= mintime):
                            mintime = t
                            chosen = i
                pointers[chosen] += 1
                time = mintime

        return metric
