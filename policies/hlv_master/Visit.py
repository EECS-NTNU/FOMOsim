import settings
import sim
import time

class Visit():
    def __init__(self, 
                 station, 
                 loading_quantity, 
                 unloading_quantity, 
                 swap_quantity, 
                 arrival_time, 
                 vehicle,
                 operating_radius = settings.OPERATOR_RADIUS):
        self.station = station
        self.loading_quantity = loading_quantity
        self.unloading_quantity = unloading_quantity
        self.swap_quantity = swap_quantity
        self.arrival_time = arrival_time
        self.vehicle = vehicle
        self.operating_radius = operating_radius
    
    def get_depature_time(self):
        walking_time = [0.8,1,2,3]
        return self.arrival_time + (self.loading_quantity + self.unloading_quantity + self.swap_quantity) * settings.MINUTES_PER_ACTION * (walking_time[self.operating_radius])
    
    def __repr__(self) -> str:
        return f'Station: {self.station.location_id}, arrrival_time: {self.arrival_time}, {self.loading_quantity} + {self.unloading_quantity} + {self.swap_quantity}'