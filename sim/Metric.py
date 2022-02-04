import numpy as np

class Metric:
    """
    Class for storing and aggregate the metric data of the instance
    """

    def __init__(self):
        self.metrics = {}

        self.metrics["lost_demand"] = []
        self.metrics["deficient_battery"] = []
        self.metrics["total_available_scooters"] = []

        self.timeline = []

    def add_analysis_metrics(self, sim):
        """
        Add data to analysis
        :param sim: sim object to record state from
        """
        self.metrics["lost_demand"].append(
            sum(
                [
                    1
                    for reward, location in sim.rewards
                    if reward == sim.LOST_TRIP_REWARD
                ]
            )
            if len(sim.rewards) > 0
            else 0
        )
        self.metrics["deficient_battery"].append(
            sum([scooter.battery for scooter in sim.state.get_scooters() if scooter.hasBattery()]) / len(sim.state.get_scooters())
        )
        self.metrics["total_available_scooters"].append(
            sum(
                [
                    len(cluster.get_available_scooters())
                    for cluster in sim.state.stations
                ]
            )
        )
        self.timeline.append(sim.time)

    def get_all_metrics(self):
        """
        Returns all metrics recorded for analysis
        """
        return self.metrics
