import numpy as np

class Metric:
    """
    Class for storing and aggregate the metric data of the instance
    """

    def __init__(self):
        self.metrics = {}

        self.min_time = 0
        self.max_time = 0
        self.metrics["average_battery"] = []
        self.metrics["total_available_scooters"] = []

    def add_metric(self, sim, metric, value):
        if metric not in self.metrics:
            self.metrics[metric] = []
        self.metrics.get(metric, []).append((sim.time, value))
        if((self.min_time == 0) or (self.min_time > sim.time)):
            self.min_time = sim.time
        if(sim.time > self.max_time):
            self.max_time = sim.time

    def add_analysis_metrics(self, sim):
        self.metrics["average_battery"].append((sim.time, 
                                                  sum([scooter.battery for scooter in sim.state.get_all_scooters() if scooter.hasBattery()]) / len(sim.state.get_all_scooters()))
        )
        self.metrics["total_available_scooters"].append((sim.time, 
            sum(
                [
                    len(station.get_available_scooters())
                    for station in sim.state.stations
                ]
            )
        ))
        if((self.min_time == 0) or (self.min_time > sim.time)):
            self.min_time = sim.time
        if(sim.time > self.max_time):
            self.max_time = sim.time

    def timeline(self, metric="average_battery"):
        return [item[0] for item in metrics[metric]]

    def values(self, metric):
        return [item[1] for item in metrics[metric]]

    def get_all_metrics(self):
        """
        Returns all metrics recorded for analysis
        """
        return self.metrics
