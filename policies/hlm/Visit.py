import settings
import sim
import time

class Visit():
    def __init__(self, station, loading_quantity, unloading_quantity, swap_quantity, arrival_time, vehicle):
        self.station = station
        self.loading_quantity = loading_quantity
        self.unloading_quantity = unloading_quantity
        self.swap_quantity = swap_quantity
        self.arrival_time = arrival_time
        self.vehicle = vehicle
    
    def get_depature_time(self):
        return self.arrival_time + (self.loading_quantity + self.unloading_quantity)*settings.MINUTES_PER_ACTION

