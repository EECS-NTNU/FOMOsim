from shapely.geometry import MultiPoint
import numpy as np
from sim.Location import Location
import sim
from settings import *
import copy

class Station(Location):
    """
    Station class representing a collection of bikes. Contains all customer behaviour data.
    """

    def __init__(
        self,
        station_id,
        bikes = {},
        leave_intensities=None,
        leave_intensities_stdev=None,
        arrive_intensities=None,
        arrive_intensities_stdev=None,
        center_location=None,
        move_probabilities=None,
        average_number_of_bikes=None,
        target_state=None,
        capacity = DEFAULT_STATION_CAPACITY,
        original_id = None,
        charging_station = None, 
        area = None,
        is_station_based = True
    ):
        super().__init__(
            *(center_location if center_location else self.__compute_center(bikes)), station_id
        )

        self.set_bikes(bikes)

        self.area = area
        self.leave_intensities = leave_intensities if leave_intensities else [[0 for _ in range(24)] for _ in range(7)] 
        self.leave_intensities_stdev = leave_intensities_stdev if leave_intensities_stdev else [[0 for _ in range(24)] for _ in range(7)] 
        self.arrive_intensities = arrive_intensities if arrive_intensities else [[0 for _ in range(24)] for _ in range(7)] 
        self.arrive_intensities_stdev = arrive_intensities_stdev if arrive_intensities_stdev else [[0 for _ in range(24)] for _ in range(7)] 

        self.move_probabilities = move_probabilities
        self.is_station_based = is_station_based

        self.average_number_of_bikes = average_number_of_bikes
        self.capacity = int(capacity) if capacity != 'inf' else float(capacity) # handles if capacity isn't infinite
        self.original_id = original_id
        self.charging_station = charging_station
        self.neighbours = []

        if target_state is not None:
            self.target_state = target_state
        else:
            self.target_state = [[0 for hour in range(24)] for day in range(7)]
            
        self.metrics = sim.Metric()

        if len(self.bikes) > self.capacity:
            self.capacity = len(self.bikes)

    def sloppycopy(self, *args):
        return Station(
            #self.location_id,
            self.id,
            list(copy.deepcopy(self.bikes).values()),

            leave_intensities=self.leave_intensities,
            leave_intensities_stdev=self.leave_intensities_stdev,
            arrive_intensities=self.arrive_intensities,
            arrive_intensities_stdev=self.arrive_intensities_stdev,

            move_probabilities=self.move_probabilities,

            center_location=self.get_location(),
            average_number_of_bikes=self.average_number_of_bikes,
            target_state=self.target_state,
            capacity=self.capacity,
            original_id=self.original_id,
            charging_station=self.charging_station,
        )

    def is_depot(self):
        return False

    def set_bikes(self, bikes):
        self.bikes = {bike.bike_id : bike for bike in bikes}
        for bike in bikes:
            #bike.set_location(self.lat, self.lon, self.location_id)
            bike.set_location(self.lat, self.lon)

    def spare_capacity(self):
        return self.capacity - len(self.bikes)
    
    def get_neighbours(self):
        return self.neighbours
    
    def get_target_state(self, day, hour):
        #return self.target_state[day % 7][hour % 24]
        ts = self.target_state
        #print("TARGET STATE:", ts)
        # Support multiple possible formats for target_state that can appear in instances:
        # - 7x24 list: ts[day][hour]
        # - 7-length list of ints: ts[day]
        # - 24-length list of ints: ts[hour]
        # - scalar int/float: uniform target
        if isinstance(ts, (int, float)):
            #print("YOOOOOOOOOO")
            return int(ts)
        try:
            #print("HHHEHCGHROCGRGCR#CGR#HCH#RLHCI#RHCH#RIC")
            # Preferred: 7x24
            return ts[day % 7][hour % 24]
        except Exception:
            #print("EHHEHEHE")
            try:
                # 7-length (per day)
                if len(ts) == 7:
                    val = ts[day % 7]
                    return int(val) if isinstance(val, (int, float)) else val
                # 24-length (per hour)
                if len(ts) == 24:
                    val = ts[hour % 24]
                    return int(val) if isinstance(val, (int, float)) else val
            except Exception:
                pass
        # Fallback
        return 0

    def get_move_probabilities(self, state, day, hour):
        """
        Returns a dictionary. Key = location_id, Value = probability to go there
        """
        if self.move_probabilities is None:
            num_stations = len(state.stations)
            mp = {station_id: 1/num_stations for station_id in state.get_station_ids()}
            return mp
        return self.move_probabilities[day % 7][hour % 24]

    def get_arrive_intensity(self, day, hour):
        return self.arrive_intensities[day % 7][hour % 24]

    def get_leave_intensity(self, day, hour):
        return self.leave_intensities[day % 7][hour % 24]

    def get_arrive_intensity_stdev(self, day, hour):
        return self.arrive_intensities_stdev[day % 7][hour % 24]
    
    def get_leave_intensity_stdev(self, day, hour):
        return self.leave_intensities_stdev[day % 7][hour % 24]

    def number_of_bikes(self):
        return len(self.bikes)

    def __compute_center(self, bikes):
        if len(bikes) > 0:
            station_centroid = MultiPoint(
                list(map(lambda bike: (bike.get_location()), bikes))
            ).centroid
            return station_centroid.x, station_centroid.y
        else:
            return 0, 0

    def add_bike(self, bike):
        if len(self.bikes) >= self.capacity:
            return False
        # Adding bike to bike list
        self.bikes[bike.bike_id] = bike
        #bike.set_location(self.get_lat(), self.get_lon(), self.location_id)
        bike.set_location(self.get_lat(), self.get_lon())
        return True

    def remove_bike(self, bike):
        del self.bikes[bike.bike_id]
        # bike.set_location(None, None, None)

    def get_bikes(self):
        return list(self.bikes.values())

    def get_available_bikes(self):
        return [
            bike for bike in self.bikes.values() if bike.usable()
        ]
    
    def get_unusable_bikes(self):
        return [
            bike for bike in self.bikes.values() if not bike.usable()
        ]

    def get_swappable_bikes(self, battery_limit=BATTERY_LIMIT_TO_SWAP):
        """
        Filter out bikes with 100% battery and sort them by battery percentage
        """
        bikes = [
            bike for bike in self.bikes.values() if bike.hasBattery() and bike.battery < battery_limit
        ]
        return sorted(bikes, key=lambda bike: bike.battery, reverse=False)

    def get_bike_from_id(self, bike_id):
        return self.bikes[bike_id]
    
    def get_average_maintenance_criticality(self):
        """
        Calculate the average maintenance criticality score of all bikes at this station.
        
        Returns:
            float: Average maintenance criticality (0.0 if no bikes present)
        """
        if len(self.bikes) == 0:
            return 0.0
        
        total_criticality = sum(bike.maintenance_criticality for bike in self.bikes.values())
        return total_criticality / len(self.bikes)
    
    def get_maintenance_criticality_stats(self):
        """
        Get detailed maintenance criticality statistics for bikes at this station.
        
        Returns:
            dict: {
                'average': float,
                'max': float,
                'min': float,
                'count': int,
                'high_criticality_count': int (bikes with criticality > 0.5)
            }
        """
        if len(self.bikes) == 0:
            return {
                'average': 0.0,
                'max': 0.0,
                'min': 0.0,
                'count': 0,
                'high_criticality_count': 0
            }
        
        criticalities = [bike.maintenance_criticality for bike in self.bikes.values()]
        
        return {
            'average': sum(criticalities) / len(criticalities),
            'max': max(criticalities),
            'min': min(criticalities),
            'count': len(criticalities),
            'high_criticality_count': sum(1 for c in criticalities if c > 0.5)
        }
    
    def get_bikes_in_need_of_maintenance(self, threshold=MAINTENANCE_LIMIT_TO_CHECK):
        """
        Get a list of bikes that need maintenance based on a criticality threshold.
        
        Args:
            threshold (float): Maintenance criticality threshold (default: MAINTENANCE_LIMIT_TO_CHECK)
        
        Returns:
            list: List of Bike objects needing maintenance
        """
        bikes = [
            bike for bike in self.bikes.values() if bike.maintenance_criticality >= threshold
        ]
         
        return sorted(bikes, key=lambda bike: bike.maintenance_criticality, reverse=True)
    
    def set_neighboring_stations(self, neighboring_stations_dict, location_list):
        """
        Defines the list neighboring_stations consisting of Station-objects
        """
        self_index = location_list.index(self)
        neighboring_stations_list = neighboring_stations_dict[self_index]
        for index in neighboring_stations_list:
            self.neighbours.append(location_list[index])
    
    def set_move_probabilities(self, station_list):
        move_probabilities = [[{} for _ in range(24)] for _ in range(7)]
        for day in range(7):
            for hour in range(24):
                for ind in range(len(self.move_probabilities[day][hour])):
                    station_id = station_list[ind].id
                    move_probabilities[day][hour][station_id] = self.move_probabilities[day][hour][ind]
        
        self.move_probabilities = move_probabilities

    def __repr__(self):
        return (
            f"<Station {self.id}: {len(self.bikes)} bikes>"
        )

    def __str__(self):
        return f"Station {self.id}: Arrive {self.get_arrive_intensity(0, 8):4.2f} Leave {self.get_leave_intensity(0, 8):4.2f} Ideal {self.get_target_state(0, 8)} Bikes {len(self.bikes):3d}"
