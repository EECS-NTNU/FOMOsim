class Event:
    """
    Base class for all events in the Event Based Simulation Engine
    """

    def __init__(self, time: int):
        self.time = time

    def perform(self, simul) -> None:
        if simul.state.time <= self.time:
            simul.state.time = self.time
        else:
            raise ValueError(
                f"{self.__class__.__name__} object tries to move the simul backwards in time. Event time: {self.time}"
                f", Simul time: {simul.state.time}"
            )

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time}>"
