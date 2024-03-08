WEEKS = 3

import random
import h3
from random import randint, sample
import requests
from shapely.geometry import Point, Polygon
from datetime import datetime, timezone
import numpy as np

class Hexagon:
    "Lager et område, brukes til å telle arrivals/departures og move probabilities"
    def __init__(self, hex_id, vertices, center):
        self.hex_id = hex_id
        self.vertices = vertices
        self.center = center
        self.e_scooters = []  # list of int
        self.arrival_intensities = [[0 for i in range(24)] for i in range(7)]
        self.depature_intensities = [[0 for i in range(24)] for i in range(7)]
        self.average_arrival_intensities =  [[0 for i in range(24)] for i in range(7)]
        self.average_depature_intensities =  [[0 for i in range(24)] for i in range(7)]
        self.move_probabilities = [] # [[ [{area_id: probability} x alle_steder] x 24] x 7]
        self.move_probabilities_normalized = []  # [[ [{area_id: probability} x alle_steder] x 24] x 7]

    def add_e_scooters_random(self, num_scooters):
        for _ in range(num_scooters):
            self.e_scooters.append(random.randint(0, 100))

    def is_point_inside_boundingbox(self, bbox):
        lat, lon = self.center
        min_lon, max_lon, min_lat, max_lat = bbox
        return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat

    def is_point_inside(self, location):
        polygon = Polygon(self.vertices)
        return polygon.contains(Point(location))

    def get_elevation(self, api_key):
        # Definerer URLen til Google Maps Elevation API
        latitude = self.center[0]
        longitude = self.center[1]

        url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={latitude},{longitude}&key={api_key}"

        try:
            # Sender en GET-forespørsel til APIet for å hente høydedata
            response = requests.get(url)
            
            # Sjekker om forespørselen var vellykket (status 200)
            if response.status_code == 200:
                # Henter JSON-data fra responsen
                data = response.json()
                
                # Sjekker om det finnes høydedata i responsen
                if 'results' in data and data['results']:
                    # Henter ut høydedata fra responsen
                    elevation = data['results'][0]['elevation']
                    # Returnerer høydedataen
                    return elevation
        except Exception as e:
            return None
    
    
    def update_depature_intensities(self, day, hour, end_hex):
        self.depature_intensities[day][hour] += 1

        # Hvis den er innenfor kartet
        if end_hex is not None:
            self.update_move_probabilities(day, hour, end_hex)

    def update_arrival_intensities(self, day, hour):
        self.depature_intensities[day][hour] += 1
    
    def calculate_average_arrivals(self):
        np_array = np.array(self.arrival_intensities)
        average_array = np_array / WEEKS
        self.average_arrival_intensities = average_array.tolist()

    def calculate_average_depatures(self):
        np_array = np.array(self.depature_intensities)
        average_array = np_array / WEEKS
        self.average_depature_intensities = average_array.tolist()
 
    def update_move_probabilities(self, day, hour, end_hex):
        self.move_probabilities[day][hour][end_hex] += 1

    def normalize_move_probabilities(self):
        for day in range(7):
            for hour in range(23):
                total_trips = sum(self.move_probabilities[day][hour].values())
                if total_trips != 0:
                    for key, value in self.move_probabilities[day][hour].items():
                        self.move_probabilities_normalized[day][hour][key] = value / total_trips
         

class HexWeb:
    def __init__(self, lat, lng, resolution, ring_radius, total_scooters, hexagons = []):
        self.lat = lat
        self.lng = lng
        self.resolution = resolution
        self.ring_radius = ring_radius
        self.hexagons = hexagons
        self.total_scooters = total_scooters 

    def generate_hex_web(self):
        # Generate H3 index for the given coordinates at the specified resolution
        index = h3.geo_to_h3(self.lat, self.lng, self.resolution)
        # Determine the hexagons within the specified ring radius from the center
        hexagons = h3.k_ring(index, self.ring_radius)
    
        id = 0
        for hex_id in hexagons:

            # Get the vertices of the hexagon
            vertices = h3.h3_to_geo_boundary(hex_id, geo_json=False)
            # Get the center of the hexagon
            center = h3.h3_to_geo(hex_id)
            # Create a Hexagon object
            hexagon = Hexagon("A" + str(id), vertices, center)
            self.hexagons.append(hexagon)
            id += 1
    
    @staticmethod
    def generate_hex_web_from_json(data):
        hexagons = []
        for hex_data in data['areas']:
            hex_id = hex_data['hex_id']
            vertices = hex_data['edges']  # Bruker 'edges' som vertices
            center = hex_data['location']  # Bruker 'location' som center
            hexagon = Hexagon(hex_id, vertices, center)
            hexagons.append(hexagon)
        return hexagons

    def distribute_scooters_random(self):
        scooters_remaining = self.total_scooters
        while scooters_remaining > 0:
            chosen_hex = random.choice(self.hexagons)
            chosen_hex.add_e_scooters_random(1)
            scooters_remaining -= 1

    def filter_elevation_hex_centers(self, api_key):
        relevant_hexagons = []
        for hexagon in self.hexagons:
            # Get the map data for the hexagon center
            elevation = hexagon.get_elevation(api_key)
            # Check if the elevation is 0 or negative
            if elevation > 0:
                # Print the hexagon center and its elevation
                #print("evaluation > 0")
                #print(f"Hexagon Center: {latitude}, {longitude}")
                relevant_hexagons.append(hexagon)
        return relevant_hexagons

    # Funksjon for å generere hexagoner og filtrere basert på bounding box
    def filter_hexagons_outside_boundingbox(self, bbox):
        inside_hexagons = []
        for hex in self.hexagons:
             # Finner senteret av hexagonet
            if hex.is_point_inside_boundingbox(bbox):
                inside_hexagons.append(hex)

        return inside_hexagons
    
    def add_trip_to_stats(self, start_coords, end_coords, start_time, end_time):
        # Konverter tidsstempel til datetime
        start_dt = datetime.fromtimestamp(start_time / 1000, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end_time / 1000, tz=timezone.utc)

        start_hex = None
        end_hex = None

        #start_coords kommer i feil rekkefølge, så vi må bytte om på dem
        start_coords = (start_coords[1], start_coords[0])
        end_coords = (end_coords[1], end_coords[0])
        
        # Finn hex ID for start og slutt koordinater
        for hex in self.hexagons:
            if hex.is_point_inside(start_coords):
                start_hex = hex
            if hex.is_point_inside(end_coords):
                end_hex = hex
            if start_hex is not None and end_hex is not None:
                break

        if end_hex != None: 
            end_hex.update_arrival_intensities(end_dt.weekday(), end_dt.hour)
        
        if start_hex is not None and end_hex != None:
            start_hex.update_depature_intensities(start_dt.weekday(), start_dt.hour, end_hex.hex_id)

    
    def calc_average_arrival_depature(self):
        for hex in self.hexagons: 
            hex.calculate_average_depatures()
            hex.calculate_average_arrivals()
    
    def calc_move_probabilities(self):
        for hex in self.hexagons:
            hex.normalize_move_probabilities()
    
    def init_move_probabilities(self):
        hex_dict = {hex.hex_id: 0 for hex in self.hexagons}
        for hex in self.hexagons:
            hex.move_probabilities = [[hex_dict for i in range(24)] for i in range(7)]
            hex.move_probabilities_normalized = [[hex_dict for i in range(24)] for i in range(7)]
    
    def find_arrival_depature_intensities(self, data):
        # Siden 'data' nå forstås å være et dictionary med en nøkkel 'data' som peker til en liste av turer
        for trip in data['data']:  # Rettet tilgangen til listen av turer
            start_coords = trip["route"]["features"][0]["geometry"]["coordinates"]
            end_coords = trip["route"]["features"][-1]["geometry"]["coordinates"]
            start_time = trip["start_time"]
            end_time = trip["end_time"]
            
            self.add_trip_to_stats(start_coords, end_coords, start_time, end_time)
