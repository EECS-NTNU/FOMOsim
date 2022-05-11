class Event:
    """
    Base class for all events in the Event Based Simulation Engine
    """

    def __init__(self, time: int):
        self.time = time

    def perform(self, world) -> None:
        if world.time <= self.time:
            world.time = self.time
        else:
            raise ValueError(
                f"{self.__class__.__name__} object tries to move the world backwards in time. Event time: {self.time}"
                f", World time: {world.time}"
            )

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time}>"
