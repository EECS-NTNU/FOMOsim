import sim
from sim.LoadSave import LoadSave
import numpy as np
from settings import *
import copy
import json
import gzip
import os
# from policies.inngjerdingen_moeller.parameters_MILP import MILP_data 


class State(LoadSave):
    """
    Container class for the whole state of all stations. Data concerning the interplay between stations are stored here
    """

    def __init__(
        self,
        stations = [],
        vehicles = [],
        bikes_in_use = {}, # bikes not parked at any station
        mapdata=None,
        traveltime_matrix=None,
        traveltime_matrix_stddev=None,
        traveltime_vehicle_matrix=None,
        traveltime_vehicle_matrix_stddev=None, 
        rng = None,
        rng2 = None,
        seed = None
    ):
        if rng is None:
            self.rng = np.random.default_rng(None)
        else:
            self.rng = rng

        if rng2 is None:
            self.rng2 = np.random.default_rng(None)
        else:
            self.rng2 = rng2

        self.vehicles = vehicles
        self.bikes_in_use = bikes_in_use
        self.seed = seed

        self.set_stations(stations)

        self.traveltime_matrix = traveltime_matrix
        self.traveltime_matrix_stddev = traveltime_matrix_stddev
        self.traveltime_vehicle_matrix = traveltime_vehicle_matrix
        self.traveltime_vehicle_matrix_stddev = traveltime_vehicle_matrix_stddev

        if traveltime_matrix is None:
            self.traveltime_matrix = self.calculate_traveltime(BIKE_SPEED)

        if traveltime_vehicle_matrix is None:
            self.traveltime_vehicle_matrix = self.calculate_traveltime(VEHICLE_SPEED)

        self.mapdata = mapdata

    def sloppycopy(self, *args):
        locationscopy = []
        for s in self.locations:
            locationscopy.append(s.sloppycopy())

        new_state = State(
            locationscopy,
            copy.deepcopy(self.vehicles),
            copy.deepcopy(self.bikes_in_use),
            traveltime_matrix = self.traveltime_matrix,
            traveltime_matrix_stddev = self.traveltime_matrix_stddev,
            traveltime_vehicle_matrix = self.traveltime_vehicle_matrix,
            traveltime_vehicle_matrix_stddev = self.traveltime_vehicle_matrix_stddev,
            rng = self.rng,
        )

        for vehicle in new_state.vehicles:
            vehicle.location = new_state.get_location_by_id(
                vehicle.location.id
            )

        return new_state


    @staticmethod
    def get_initial_state(statedata):
        # create stations

        stations = []
        # initial_battery_levels = [83, 39, 45, 34, 84, 63, 73, 69, 92, 55, 77, 27, 84, 95, 57, 98, 23, 44, 58, 73, 55, 69, 23, 38, 44, 94, 69, 95, 50, 89, 61, 93, 98, 33, 94, 65, 74, 92, 20, 63, 29, 77, 52, 60, 78, 26, 30, 37, 26, 42, 75, 45, 98, 68, 74, 87, 78, 32, 42, 20, 50, 28, 21, 31, 48, 98, 84, 65, 70, 39, 74, 43, 50, 78, 71, 84, 97, 22, 60, 76, 71, 49, 25, 93, 77, 55, 87, 79, 48, 88, 41, 89, 41, 34, 98, 76, 95, 89, 81, 73, 89, 35, 34, 41, 96, 67, 86, 38, 52, 29, 72, 69, 92, 78, 57, 26, 74, 31, 84, 95, 67, 80, 94, 45, 91, 59, 57, 75, 71, 73, 61, 76, 97, 38, 29, 60, 38, 22, 63, 83, 21, 84, 50, 82, 87, 33, 26, 41, 85, 54, 23, 23, 87, 39, 57, 23, 52, 48, 25, 63, 90, 91, 46, 58, 90, 99, 60, 50, 52, 78, 81, 58, 87, 73, 90, 45, 37, 90, 40, 62, 23, 45, 56, 20, 61, 43, 89, 52, 38, 52, 98, 97, 37, 72, 94, 41, 90, 81, 75, 97, 27, 90, 58, 22, 84, 93, 93, 51, 82, 40, 71, 67, 42, 90, 34, 28, 29, 65, 31, 81, 23, 64, 51, 89, 99, 88, 41, 39, 71, 36, 79, 79, 54, 49, 93, 59, 64, 73, 94, 96, 96, 96, 35, 44, 84, 95, 77, 26, 35, 73, 55, 21, 22, 69, 25, 69, 56, 54, 29, 34, 36, 46, 45, 83, 90, 72, 27, 62, 21, 23, 25, 46, 39, 68, 52, 23, 80, 34, 27, 29, 36, 84, 35, 90, 31, 57, 94, 99, 65, 72, 92, 29, 82, 78, 87, 41, 99, 95, 48, 84, 27, 55, 59, 85, 77, 21, 67, 65, 57, 72, 87, 22, 73, 72, 40, 52, 46, 41, 58, 63, 30, 36, 93, 48, 91, 51, 71, 43, 60, 54, 99, 80, 30, 47, 47, 75, 46, 29, 84, 51, 51, 93, 49, 33, 84, 43, 71, 95, 34, 81, 64, 53, 86, 46, 74, 25, 55, 47, 81, 59, 81, 64, 84, 30, 20, 36, 84, 34, 43, 56, 31, 39, 86, 36, 31, 29, 90, 60, 59, 47, 89, 54, 41, 96, 32, 96, 54, 38, 35, 96, 78, 59, 28, 91, 60, 25, 23, 50, 90, 23, 90, 35, 59, 51, 95, 90, 68, 97, 65, 29, 23, 78, 60, 75, 99, 22, 50, 97, 83, 96, 68, 97, 64, 52, 68, 90, 75, 72, 52, 73, 48, 32, 88, 20, 88, 51, 44, 84, 83, 82, 23, 20, 49, 39, 87, 54, 86, 86, 25, 67, 34, 31, 90, 47, 65, 70, 45, 35, 80, 25, 59, 73, 57, 30, 90, 25, 87, 95, 20, 92, 93, 71, 71, 86, 60, 71, 86, 88, 74, 69, 21, 84, 92, 60, 83, 34, 37, 42, 46, 78, 32, 50, 60, 20, 25, 30, 32, 55, 81, 94, 20, 29, 70, 98, 54, 25, 31, 66, 83, 26, 68, 71, 88, 95, 91, 59, 96, 71, 21, 36, 20, 94, 60, 21, 83, 72, 62, 23, 95, 98, 40, 51, 27, 83, 54, 89, 32, 94, 57, 48, 71, 57, 73, 57, 29, 89, 95, 26, 54, 44, 49, 34, 71, 85, 44, 29, 95, 92, 95, 86, 31, 33, 81, 95, 75, 77, 66, 30, 24, 67, 95, 45, 88, 54, 59, 24, 77, 89, 29, 87, 90, 30, 38, 20, 53, 80, 60, 93, 51, 71, 27, 71, 51, 33, 25, 82, 83, 62, 76, 74, 75, 62, 79, 30, 84, 83, 56, 26, 60, 95, 49, 35, 87, 86, 96, 29, 84, 80, 32, 69, 62, 45, 55, 53, 98, 78, 70, 63, 57, 82, 75, 70, 61, 89, 24, 58, 78, 69, 72, 23, 53, 83, 69, 41, 97, 46, 92, 78, 83, 20, 79, 32, 78, 59, 72, 59, 86, 88, 75, 20, 54, 99, 40, 72, 96, 21, 98, 40, 74, 28, 93, 72, 33, 70, 88, 74, 74, 64, 50, 58, 30, 89, 60, 56, 72, 99, 92, 39, 51, 30, 78, 90, 38, 57, 86, 60, 34, 30, 40, 45, 53, 51, 68, 39, 38, 99, 49, 79, 23, 23, 52, 88, 49, 53, 93, 33, 33, 70, 71, 80, 31, 57, 64, 21, 22, 72, 32, 46, 49, 80, 60, 91, 72, 68, 72, 71, 21, 63, 95, 36, 47, 47, 71, 63, 59, 46, 85, 33, 56, 96, 63, 23, 80, 88, 77, 99, 53, 25, 98, 46, 53, 23, 92, 89, 80, 42, 66, 37, 92, 63, 88, 49, 75, 49, 60, 45, 39, 29, 51, 89, 74, 83, 55, 86, 33, 99, 78, 69, 94, 68, 77, 48, 55, 83, 33, 47, 47, 47, 24, 54, 26, 54, 73, 23, 24, 44, 33, 27, 34, 41, 79, 69, 52, 36, 40, 94, 39, 32, 45, 94, 99, 45, 58, 57, 79, 24, 62, 48, 51, 21, 34, 92, 41, 49, 97, 72, 24, 80, 93, 71, 41, 84, 85, 51, 44, 64, 55, 84, 79, 45, 43, 22, 64, 47, 24, 34, 88, 57, 74, 95, 39, 79, 56, 45, 36, 24, 45, 81, 24, 59, 29, 45, 88, 23, 22, 75, 60, 35, 87, 71, 59, 23, 82, 31, 28, 58, 69, 29, 64, 56, 63, 36, 98, 75, 87, 63, 77, 86, 92, 71, 93, 71, 95, 89, 76, 44, 46, 38, 99, 39, 61, 50, 83, 91, 70, 30, 84, 42, 69, 32, 83, 97, 64, 91, 24, 21, 25, 27, 97, 42, 26, 40, 40, 97, 40, 78, 60, 95, 61, 31, 97, 36, 67, 35, 55, 27, 48, 99, 62, 64, 21, 39, 71, 52, 92, 30, 60, 41, 65, 52, 86, 62, 99, 22, 35, 86, 76, 96, 73, 46, 20, 83, 68, 42, 90, 99, 72, 96, 52, 75, 86, 60, 42, 51, 41, 41, 71, 23, 95, 60, 63, 62, 37, 63, 35, 53, 54, 47, 20, 67, 23, 60, 74, 58, 31, 32, 92, 64, 63, 43, 50, 24, 74, 32, 72, 99, 61, 80, 30, 98, 36, 37, 54, 40, 26, 88, 65, 47, 38, 66, 90, 47, 57, 51, 67, 41, 87, 86, 32, 78, 83, 53, 62, 58, 57, 96, 38, 59, 43, 63, 56, 82, 48, 54, 96, 61, 61, 40, 36, 95, 51, 91, 62, 39, 69, 59, 34, 33, 27, 66, 73, 70, 23, 31, 99, 99, 71, 60, 47, 84, 39, 82, 45, 64, 81, 46, 23, 80, 68, 54, 25, 45, 31, 80, 27, 77, 48, 35, 79, 34, 89, 39, 90, 47, 46, 59, 42, 48, 47, 57, 45, 66, 41, 31, 51, 51, 83, 96, 66, 82, 44, 32, 99, 55, 34, 40, 63, 80, 53, 55, 88, 68, 99, 54, 37, 66, 40, 86, 22, 84, 23, 98, 63, 36, 60, 68, 70, 40, 85, 73, 30, 61, 41, 45, 69, 25, 87, 99, 27, 34, 58, 81, 64, 75, 25, 65, 90, 84, 85, 69, 24, 37, 60, 64, 70, 57, 70, 72, 26, 54, 67, 22, 42, 29, 63, 75, 75, 95, 52, 88, 64, 69, 76, 90, 63, 92, 59, 41, 95, 64, 67, 28, 89, 22, 74, 88, 74, 64, 24, 43, 35, 49, 21, 55, 64, 68, 97, 62, 20, 32, 36, 21, 62, 61, 94, 31, 66, 31, 77, 21, 69, 80, 45, 74, 65, 20, 59, 27, 72, 92, 21, 54, 21, 33, 52, 73, 67, 38, 81, 32, 56, 60, 70, 25, 55, 22, 89, 55, 44, 42, 81, 92, 36, 81, 67, 40, 71, 20, 61, 32, 28, 39, 82, 65, 39, 39, 39, 58, 91, 53, 41, 73, 76, 93, 38, 46, 38, 35, 88, 87, 30, 47, 71, 22, 96, 69, 72, 84, 87, 75, 76, 25, 55, 74, 47, 84, 87, 65, 43, 72, 62, 50, 86, 41, 60, 76, 51, 56, 31, 33, 64, 88, 60, 49, 53, 83, 77, 34, 55, 66, 38, 38, 94, 22, 50, 73, 63, 90, 59, 88, 49, 26, 30, 36, 40, 22, 64, 62, 38, 25, 40, 39, 90, 60, 25, 53, 54, 44, 89, 27, 32, 57, 73, 90, 39, 61, 89, 51, 96, 51, 88, 95, 43, 77, 87, 42, 27, 81, 71, 99, 44, 77, 71, 20, 38, 99, 59, 53, 49, 80, 43, 54, 78, 60, 76, 90, 60, 32, 96, 56, 33, 22, 56, 24, 26, 36, 24, 46, 23, 96, 30, 96, 35, 46, 84, 99, 88, 68, 86, 89, 84, 25, 53, 55, 41, 21, 56, 77, 54, 89, 87, 59, 37, 78, 53, 95, 48, 29, 88, 93, 41, 89, 34, 62, 38, 37, 33, 93, 94, 45, 75, 87, 38, 46, 85, 47, 36, 55, 97, 62, 87, 96, 54, 50, 54, 78, 29, 52, 66, 52, 52, 62, 33, 34, 77, 43, 27, 91, 84, 38, 30, 65, 37, 71, 91, 76, 54, 43, 91, 80, 92, 20, 55, 65, 64, 86, 67, 49, 56, 90, 28, 44, 68, 82, 95, 26, 47, 27, 92, 25, 40, 91, 23, 76, 40, 85, 39, 64, 54, 71, 87, 28, 77, 91, 79, 59, 74, 44, 48, 85, 65, 69, 99, 29, 44, 52, 21, 94, 68, 42, 52, 27, 70, 39, 53, 95, 56, 58, 70, 43, 63, 53, 59, 54, 48, 76, 50, 47, 41, 47, 68, 50, 51, 25, 87, 66, 48, 34, 99, 55, 44, 22, 95, 36, 40, 32, 37, 83, 49, 48, 69, 31, 94, 77, 75, 42, 38, 42, 55, 53, 24, 83, 98, 62, 58, 30, 43, 50, 42, 62, 33, 83, 58, 32, 72, 75, 68, 94, 87, 91, 36, 39, 58, 96, 74, 63, 24, 74, 56, 90, 25, 62, 71, 54, 74, 30, 89, 95, 99, 43, 45, 33, 83, 96, 73, 41, 62, 80, 49, 35, 53, 89, 64, 27, 24, 27, 35, 68, 53, 47, 91, 94, 52, 56, 96, 38, 55, 98, 50, 33, 82, 34, 33, 81, 86, 36, 54, 38, 37, 96, 37, 77, 20, 87, 29, 30, 98, 56, 62, 90, 56, 74, 41, 59, 22, 67, 21, 71, 41, 48, 93, 59, 25, 40, 91, 63, 20, 56, 48, 82, 40, 31, 22, 21, 32, 42, 46, 71, 66, 51, 86, 52, 85, 87, 26, 91, 21, 60, 79, 51, 57, 58, 22, 71, 23, 48, 93, 44, 99, 76, 92, 79, 23, 32, 57, 50, 93, 91, 77, 52, 56, 99, 20, 64, 34, 47, 28, 22, 36, 23, 90, 88, 56, 36, 34, 67, 57, 45, 48, 97, 82, 22, 21, 71, 20, 26, 29, 68, 87, 66, 63, 44, 64, 55, 99, 88, 80, 30, 99, 82, 86, 68, 87, 65, 65, 45, 26, 71, 82, 64, 75, 26, 78, 98, 63, 22, 23, 37, 46, 32, 31, 65, 41, 58, 98, 29, 32, 90, 33, 53, 96, 38, 38, 74, 59, 71, 77, 98, 65, 58, 40, 33, 71, 91, 40, 25, 70, 42, 88, 69, 89, 66, 59, 73, 89, 50, 85, 33, 28, 67, 27, 81, 75, 89, 57, 73, 68, 42, 79, 27, 38, 62, 98, 54, 50, 63, 44, 71, 65, 99, 28, 43, 48, 78, 30, 84, 30, 46, 24, 30, 23, 86, 42, 33, 79, 43, 72, 45, 49, 61, 87, 83, 90, 51, 45, 73, 77, 73, 91, 95, 81, 94, 46, 38, 50, 58, 83, 36, 62, 25, 81, 99, 69, 77, 82, 56, 50, 31, 74, 81, 29, 48, 53, 79, 62, 24, 66, 68, 75, 40, 26, 84, 45, 41, 60, 84, 77, 30, 74, 82, 33, 95, 96, 47, 97, 23, 77, 90, 26, 92, 60, 99, 25, 67, 90, 45, 76, 68, 52, 53, 47, 72, 65, 46, 33, 37, 51, 96, 91, 26, 80, 70, 22, 82, 60, 80, 40, 36, 60, 46, 69, 72, 45, 69, 69, 27, 35, 30, 91, 55, 21, 41, 84, 32, 70, 95, 59, 37, 51, 42, 95, 22, 84, 54, 69, 73, 30, 87, 70, 63, 68, 72, 86, 73, 98, 52, 39, 69, 61, 95, 82, 62, 63, 88, 35, 30, 26, 31, 23, 46, 91, 63, 45, 36, 22, 46, 91, 84, 53, 23, 79, 31, 80, 73, 97, 24, 87, 96, 49, 61, 31, 47, 84, 49, 70, 63, 94, 38, 60, 32, 32, 95, 94, 96, 60, 26, 35, 42, 97, 71, 67, 45, 43, 69, 47, 48, 80, 70, 63, 96, 84, 68, 22, 81, 74, 67, 38, 91, 47, 23, 20, 62, 76, 45, 43, 80, 95, 76, 77, 79, 78, 58, 29, 26, 21, 51, 64, 82, 30, 29, 61, 24, 63, 22, 63, 99, 66, 41, 68, 65, 81, 47, 82, 89, 57, 97, 50, 94, 28, 41, 55, 84, 63, 56, 76, 66, 85, 22, 96, 84, 66, 37, 99, 58, 71, 33, 54, 70, 43, 65, 63, 81, 81, 60, 33, 79, 39, 91, 95, 79, 58, 26, 81, 95, 77, 32, 72, 89, 41, 66, 93, 70, 95, 93, 20, 68, 32, 77, 78, 61, 49, 89, 71, 26, 78, 57, 25, 95, 99, 42, 90, 94, 38, 39, 39, 21, 36, 75, 66, 73, 45, 49, 88, 63, 90, 62, 81, 98, 21, 73, 56, 36, 50, 46, 99, 46, 39, 68, 93, 70, 60, 32, 88, 56, 23, 23, 25, 45, 88, 54, 66, 42, 72, 35, 84, 65, 52, 49, 41, 74, 94, 52, 93, 74, 35, 34, 98, 65, 34, 32, 62, 22, 22, 65, 70, 65, 30, 26, 52, 30, 93, 40, 28, 24, 84, 32, 53, 39, 71, 24, 42, 57, 46, 79, 79, 74, 27, 43, 69, 51, 52, 81, 57, 26, 32, 56, 26, 90, 66, 38, 26, 77, 41, 56, 74, 99, 64, 21, 65, 58, 28, 26, 93, 84, 32, 62, 53, 32, 62, 30, 92, 36, 85, 83, 24, 24, 47, 83, 48, 31, 34, 66, 89, 85, 57, 65, 51, 30, 87, 85, 40, 57, 67, 45, 95, 57, 69, 60, 41, 97, 79, 61, 43, 56, 65, 37, 80, 71, 29, 54, 52, 30, 46, 42, 22, 66, 76, 98, 89, 84, 88, 62, 83, 84, 67, 89, 52, 37, 42, 23, 35, 63, 74, 25, 98, 35, 69, 32, 90, 52, 50, 74, 56, 64, 43, 64, 48, 94, 24, 96, 57, 60, 23, 67, 99, 96, 80, 74, 94, 76, 25, 21, 79, 92, 80, 51, 83, 30, 26, 86, 35, 58, 54, 21, 70, 91, 97, 60, 54, 33, 44, 76, 40, 22, 41, 67, 38, 21, 80, 96, 48, 47, 58, 62, 38, 53, 99, 89, 53, 20, 71, 91, 84, 79, 73, 38, 21, 50, 23, 32, 96, 64, 33, 28, 90, 65, 23, 82, 81, 81, 89, 79, 57, 20, 33, 57, 51, 72, 90, 28, 87, 29, 31, 40, 90, 94, 83, 65, 44, 81, 39, 54, 64, 45, 35, 94, 59, 92, 45, 99, 70, 89, 73, 52, 44, 45, 29, 27, 50, 70, 55, 76, 81, 94, 71, 34, 62, 27, 29, 58, 45, 57, 20, 68, 75, 70, 66, 30, 35, 75, 70, 64, 89, 68, 85, 92, 26, 76, 91, 47, 50, 32, 84, 46, 89, 95, 68, 85, 44, 75, 83, 56, 91, 98, 40, 63, 77, 41, 73, 69, 76, 79, 73, 46, 68, 93, 41, 60, 59, 41, 97, 94, 97, 97, 38, 23, 82, 69, 21, 72, 77, 37, 75, 62, 45, 93, 80, 23, 30, 54, 25, 78, 97, 30, 32, 81, 23, 78, 87, 32, 56, 78, 91, 99, 93, 62, 79, 88, 76, 44, 70, 61, 46, 28, 24, 66, 84, 83, 86, 54, 20, 60, 79, 73, 88, 55, 86, 94, 52, 51, 31, 29, 68, 70, 89, 20, 22, 41, 27, 49, 97, 63, 55, 45, 82, 26, 47, 74, 55, 93, 73, 67, 40, 46, 40, 48, 60, 87, 38, 91, 62, 20, 54, 23, 24, 78, 29, 45, 35, 60, 64, 89, 90, 47, 22, 52, 35, 22, 91, 28, 65, 86, 26, 54, 94, 26, 82, 43, 72, 33, 35, 47, 66, 59, 52, 21, 56, 47, 24, 36, 62, 75, 50, 72, 81, 59, 71, 52, 41, 44, 25, 31, 76, 83, 67, 78, 61, 42, 62, 63, 82, 34, 78, 69, 44, 80, 81, 80, 93, 96, 79, 89, 24, 47, 96, 77, 37, 72, 47, 26, 22, 53, 75, 30, 23, 73, 59, 84, 55, 38, 64, 66, 34, 39, 64, 35, 98, 39, 53, 77, 81, 98, 64, 92, 97, 57, 70, 66, 46, 74, 53, 21, 35, 79, 88, 27, 81, 28, 61, 70, 49, 91, 49, 41, 94, 29, 49, 41, 82, 57, 91, 73, 81, 39, 26, 96, 37, 41, 33, 86, 60, 45, 34, 38, 28, 77, 75, 38, 29, 47, 81, 54, 86, 57, 64, 35, 90, 76, 52, 82, 79, 74, 72, 67, 95, 99, 43, 85, 74, 36, 35, 67, 83, 76, 42, 51, 86, 62, 51, 21, 38, 83, 70, 35, 90, 76, 44, 69, 39, 48, 77, 52, 95, 23, 35, 74, 53, 38, 59, 62, 44, 68, 60, 34, 59, 76, 58, 22, 39, 91, 74, 43, 83, 75, 37, 82, 86, 34, 55, 55, 98, 50, 95, 85, 84, 40, 26, 36, 87, 64, 36, 64, 83, 47, 47, 98, 49, 78, 88, 21, 25, 37, 31, 74, 96]

        id_counter = 0
        for station_id, station in enumerate(statedata["stations"]):
            capacity = DEFAULT_STATION_CAPACITY
            if "capacity" in station:
                capacity = station["capacity"]

            original_id = None
            if "original_id" in station:
                original_id = station["original_id"]

            position = None
            if "location" in station:
                position = station["location"]

            charging_station = False
            if "charging_station" in station:
                charging_station = station["charging_station"]

            if ("is_depot" in station) and station["is_depot"]:
                depot_capacity = DEFAULT_DEPOT_CAPACITY
                if "depot_capacity" in station:
                    depot_capacity = station["depot_capacity"]
                stationObj = sim.Depot(station_id, depot_capacity=depot_capacity, capacity=capacity, original_id=original_id, center_location=position, charging_station=charging_station, leave_intensities = station["leave_intensities"],
                                         leave_intensities_stdev = station["leave_intensities_stdev"],
                                         arrive_intensities = station["arrive_intensities"],
                                         arrive_intensities_stdev = station["arrive_intensities_stdev"],
                                         move_probabilities = station["move_probabilities"],)

            else:
                stationObj = sim.Station(station_id,
                                         capacity=capacity,
                                         original_id=original_id,
                                         center_location=position,
                                         charging_station=charging_station,
                                         leave_intensities = station["leave_intensities"],
                                         leave_intensities_stdev = station["leave_intensities_stdev"],
                                         arrive_intensities = station["arrive_intensities"],
                                         arrive_intensities_stdev = station["arrive_intensities_stdev"],
                                         move_probabilities = station["move_probabilities"],
                                         )

            # create bikes
            bikes = []
            for _ in range(station["num_bikes"]):
                if statedata["bike_class"] == "EBike":
                    bikes.append(sim.EBike(bike_id=id_counter, battery=100)) # initial_battery_levels[id_counter]))
                else:
                    bikes.append(sim.Bike(bike_id=id_counter))
                id_counter += 1

            stationObj.set_bikes(bikes)

            stations.append(stationObj)

        # create state

        mapdata = None
        if "map" in statedata:
            mapdata = (statedata["map"], statedata["map_boundingbox"])

        state = State(stations,
                      mapdata = mapdata,
                      traveltime_matrix=statedata["traveltime"],
                      traveltime_matrix_stddev=statedata["traveltime_stdev"],
                      traveltime_vehicle_matrix=statedata["traveltime_vehicle"],
                      traveltime_vehicle_matrix_stddev=statedata["traveltime_vehicle_stdev"])
        
        neighbor_dict = state.read_neighboring_stations_from_file()
        for station in stations:
            station.set_neighboring_stations(neighbor_dict, state.stations)

        return state

    def calculate_traveltime(self, speed):
        traveltime_matrix = []
        for location in self.locations:
            neighbour_traveltime = []
            for neighbour in self.locations:
                neighbour_traveltime.append((location.distance_to(*neighbour.get_location()) / speed)*60) #multiplied by 60 to get minutes
            traveltime_matrix.append(neighbour_traveltime)
        
        return traveltime_matrix

    def set_stations(self, locations):
        self.locations = locations
        self.stations = { station.id : station for station in locations }
        self.depots = { station.id : station for station in locations if isinstance(station, sim.Depot) }

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def set_vehicles(self, policies):
        self.vehicles = []
        for vehicle_id, policy in enumerate(policies):
            self.vehicles.append(sim.Vehicle(vehicle_id, self.locations[0], policy, 
                                             VEHICLE_BATTERY_INVENTORY, VEHICLE_BIKE_INVENTORY))

    def set_move_probabilities(self, move_probabilities):
        for st in self.locations:
            st.move_probabilities = move_probabilities[st.id]

    def set_target_state(self, target_state):
        for st in self.locations:
            st.target_state = target_state[st.id]

    def get_station_by_lat_lon(self, lat: float, lon: float):
        """
        :param lat: lat location of bike
        :param lon:
        :return:
        """
        return min(list(self.stations.values()), key=lambda station: station.distance_to(lat, lon))

    def bike_in_use(self, bike):
        self.bikes_in_use[bike.id] = bike

    def remove_used_bike(self, bike):
        del self.bikes_in_use[bike.id]

    def get_used_bike(self):
        if len(self.bikes_in_use) > 0:
            bike = next(iter(self.bikes_in_use))
            self.remove_used_bike(bike)
            return bike

    # parked bikes
    def get_bikes(self):
        all_bikes = []
        for station in self.stations.values():
            all_bikes.extend(station.get_bikes())
        return all_bikes

    # parked and in-use bikes
    def get_all_bikes(self):
        all_bikes = []
        for station in self.locations:
            all_bikes.extend(station.get_bikes())
        all_bikes.extend(self.bikes_in_use.values())
        for vehicle in self.vehicles:
            all_bikes.extend(vehicle.get_bike_inventory())
            
        return all_bikes

    # parked bikes with usable battery
    def get_num_available_bikes(self):
        num = 0
        for station in self.locations:
            num += len(station.get_available_bikes())
        return num

    def get_travel_time(self, start_location_id: int, end_location_id: int):
        if self.traveltime_matrix_stddev is not None:
            return self.rng2.lognormal(self.traveltime_matrix[start_location_id][end_location_id],
                                      self.traveltime_matrix_stddev[start_location_id][end_location_id])
        else:
            return self.traveltime_matrix[start_location_id][end_location_id]

    def get_vehicle_travel_time(self, start_location_id: int, end_location_id: int):
        if self.traveltime_vehicle_matrix_stddev is not None:
            return self.rng2.lognormal(self.traveltime_vehicle_matrix[start_location_id][end_location_id],
                                self.traveltime_vehicle_matrix_stddev[start_location_id][end_location_id])
        else:
            return self.traveltime_vehicle_matrix[start_location_id][end_location_id]

    def do_action(self, action, vehicle, time):
        """
        Performs an action on the state -> changing the state
        :param time: at what time the action is performed
        :param vehicle: Vehicle to perform this action
        :param action: Action - action to be performed on the state
        """
        refill_time = 0
        if vehicle.is_at_depot():
            batteries_to_swap = min(
                vehicle.get_num_flat_batteries(),
                vehicle.location.get_available_battery_swaps(time),
            )

            refill_time += vehicle.location.swap_battery_inventory(
                time, batteries_to_swap
            )
            vehicle.add_battery_inventory(batteries_to_swap)

            #### TODO - legg til tid for dette
            for e_scooter in vehicle.get_bike_inventory():
                e_scooter.swap_battery()
                
        else:
            for pick_up_bike_id in action.pick_ups:
                pick_up_bike = vehicle.location.get_bike_from_id(
                    pick_up_bike_id
                )
                
                # Picking up bike and adding to vehicle inventory and swapping battery
                vehicle.pick_up(pick_up_bike)

                # Remove bike from current station
                vehicle.location.remove_bike(pick_up_bike)
                
            # Perform all battery swaps
            for battery_swap_bike_id in action.battery_swaps:
                battery_swap_bike = vehicle.location.get_bike_from_id(
                    battery_swap_bike_id
                )
                # Decreasing vehicle battery inventory
                vehicle.change_battery(battery_swap_bike)

            # Dropping of bikes
            for delivery_bike_id in action.delivery_bikes:
                # Removing bike from vehicle inventory
                delivery_bike = vehicle.drop_off(delivery_bike_id)

                # Adding bike to current station and changing coordinates of bike
                vehicle.location.add_bike(delivery_bike)

        # Moving the state/vehicle from this to next station
        vehicle.location = self.get_location_by_id(action.next_location)

        return refill_time

    def __repr__(self):
        string = f"<State: {len(self.get_bikes())} bikes in {len(self.stations)} stations with {len(self.vehicles)} vehicles>\n"
        for station in self.locations:
            string += f"{str(station)}\n"
        for vehicle in self.vehicles:
            string += f"{str(vehicle)}\n"
        string += f"In use: {len(self.bikes_in_use.values())}"
        return string

    def get_neighbours(
        self,
        location: sim.Location,
        number_of_neighbours=None,
        is_sorted=True,
        exclude=None,
        not_full=False,
        not_empty=False
    ):
        """
        Get sorted list of stations closest to input station
        :param is_sorted: Boolean if the neighbours list should be sorted in a ascending order based on distance
        :param location: location to find neighbours for
        :param number_of_neighbours: number of neighbours to return
        :param exclude: neighbor ids to exclude
        :return:
        """
        neighbours = [
            state_location
            for state_location in self.locations
            if state_location.id != location.id
            and state_location.id not in (exclude if exclude else [])
        ]
        if is_sorted:
            if not_full:
                neighbours = sorted(
                    [
                        state_location
                        for state_location in self.locations
                        if state_location.id != location.id
                        and state_location.id not in (exclude if exclude else [])
                        and state_location.spare_capacity() >= 1
                    ],
                    key=lambda state_location: self.traveltime_matrix[location.id][
                        state_location.id
                    ],
                )
            elif not_empty:
                neighbours = sorted(
                    [
                        state_location
                        for state_location in self.locations
                        if state_location.id != location.id
                        and state_location.id not in (exclude if exclude else [])
                        and len(state_location.get_available_bikes()) >= 1
                    ],
                    key=lambda state_location: self.traveltime_matrix[location.id][
                        state_location.id
                    ],
                )
            else:
                neighbours = sorted(
                    [
                        state_location
                        for state_location in self.locations
                        if state_location.id != location.id
                        and state_location.id not in (exclude if exclude else [])
                    ],
                    key=lambda state_location: self.traveltime_matrix[location.id][
                        state_location.id
                    ],
                )

        test = neighbours[:number_of_neighbours] if number_of_neighbours else neighbours
        if len(test) == 0:
            print('number of neighmors', neighbours[:number_of_neighbours], number_of_neighbours)
            print('neighbors', neighbours)
                
        return neighbours[:number_of_neighbours] if number_of_neighbours else neighbours

    def get_location_by_id(self, location_id: int):
        return self.locations[location_id]

    def sample(self, sample_size: int):
        # Filter out bikes not in sample
        sampled_bike_ids = self.rng2.choice(
            [bike.id for bike in self.get_bikes()], sample_size, replace=False,
        )
        for station in self.stations.values():
            station.set_bikes([
                bike
                for bike in station.get_bikes()
                if bike.id in sampled_bike_ids
            ])

    def get_vehicle_by_id(self, vehicle_id):
        """
        Returns the vehicle object in the state corresponding to the vehicle id input
        :param vehicle_id: the id of the vehicle to fetch
        :return: vehicle object
        """
        return self.vehicles[vehicle_id]
    
    def read_neighboring_stations_from_file(self):
        neighboring_stations = dict()      #{station_ID: [list of station_IDs]}
        filename = 'policies/inngjerdingen_moeller/saved_time_data/' + (self.mapdata[0]).split('.')[0].split('/')[-1] +'_static_data.json'
        if not os.path.exists(filename):
            print("JSON-file", filename, "does not exist")

        with open(filename,'r') as infile:
            json_dict = json.load(infile)
        for key, value in json_dict["neighboring_stations"].items():
            neighboring_stations[int(key)] = value


        return neighboring_stations
