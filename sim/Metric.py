import numpy as np

class Metric:
    """
    Class for storing and aggregate the metric data of the instance
    """

    def __init__(self):
        self.metrics = {}

        self.metrics["average_battery"] = []
        self.metrics["total_available_scooters"] = []

    def add_metric(self, sim, metric, value):
        if metric not in self.metrics:
            self.metrics[metric] = []
        self.metrics.get(metric, []).append((sim.time, value))

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

    def timeline(self, metric="average_battery"):
        return [item[0] for item in metrics[metric]]

    def values(self, metric):
        return [item[1] for item in metrics[metric]]

    def get_all_metrics(self):
        """
        Returns all metrics recorded for analysis
        """
        return self.metrics
