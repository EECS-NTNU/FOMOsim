import numpy as np

class Metric:
    """
    Class for storing and aggregate the metric data of the instance
    """

    def __init__(self, test_parameter_name="", test_parameter_value=0.0):
        self.lost_demand = []
        self.deficient_battery = []
        self.timeline = []
        self.total_available_scooters = []
        self.testing_parameter_name = test_parameter_name
        self.testing_parameter_value = test_parameter_value

    @classmethod
    def aggregate_metrics(cls, metrics):
        def lists_average(lists):
            return np.mean(np.stack(lists, axis=0), axis=1).tolist()

        new_sim_metric = cls()
        if all([len(metric.timeline) == 0 for metric in metrics]):
            return new_sim_metric
        number_of_metrics = len(metrics)

        # Fields to take the average of
        average_fields = [
            "lost_demand",
            "deficient_battery",
            "total_available_scooters",
        ]
        # Create dict with list for every field, start all values on zero
        fields = {field: [[0] * number_of_metrics] for field in average_fields}
        # Find the time for the latest event
        max_time = np.max(np.concatenate([metric.timeline for metric in metrics]))
        new_sim_metric.timeline = list(range(int(max_time) + 1))
        # populate fields with average at every time step
        for time in new_sim_metric.timeline[1:]:
            # If there is a new value in the timeline, update the timeline
            if any([time in metric.timeline for metric in metrics]):
                for field in fields.keys():
                    # Add new value if there is a new one, otherwise add previous value
                    fields[field].append(
                        [
                            getattr(metric, field)[
                                metric.timeline.index(time)
                            ]  # Takes the first recording in current time
                            if time in metric.timeline
                            else fields[field][time - 1][i]
                            for i, metric in enumerate(metrics)
                        ]
                    )
            # Otherwise, add previous values
            else:
                for field in fields.keys():
                    fields[field].append(fields[field][-1])
        # Take the average of all the runs
        new_sim_metric.__dict__.update(
            {
                field: lists_average(metric_list)
                for field, metric_list in fields.items()
            }
        )

        new_sim_metric.testing_parameter_name = metrics[0].testing_parameter_name
        new_sim_metric.testing_parameter_value = metrics[
            0
        ].testing_parameter_value
        return new_sim_metric

    def add_analysis_metrics(self, sim):
        """
        Add data to analysis
        :param sim: sim object to record state from
        """
        self.lost_demand.append(
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
        self.deficient_battery.append(
            sum([scooter.battery for scooter in sim.state.get_scooters() if scooter.hasBattery()]) / len(sim.state.get_scooters())
            # sum(
            #     [
            #         cluster.ideal_state * 100
            #         - (
            #             sum(
            #                 [
            #                     scooter.battery
            #                     for scooter in cluster.get_available_scooters()
            #                 ]
            #             )
            #         )
            #         for cluster in sim.state.stations
            #         if len(cluster.scooters) < cluster.ideal_state
            #     ]
            # )
            # / len(sim.state.get_scooters())
        )
        self.total_available_scooters.append(
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
        return (
            self.lost_demand,
            self.deficient_battery,
            self.total_available_scooters,
        )

