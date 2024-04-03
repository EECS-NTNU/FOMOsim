WEEKS = 3

import random
import h3
from random import randint, sample
import requests
from shapely.geometry import Point, Polygon
from datetime import datetime, timezone
import numpy as np
import geopy.distance

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
        self.target_states = [[0 for i in range(24)] for i in range(7)]
        self.initial_target_states = 0

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
        if end_hex not in self.move_probabilities[day][hour].keys():
            self.move_probabilities[day][hour][end_hex] = 1
        else:
            self.move_probabilities[day][hour][end_hex] += 1

    def normalize_move_probabilities(self):
        for day in range(7):
            for hour in range(23):
                total_trips = sum(self.move_probabilities[day][hour].values())
                if total_trips != 0:
                    for key, value in self.move_probabilities[day][hour].items():
                        self.move_probabilities_normalized[day][hour][key] = value / total_trips
    
    def get_arrival_depature_intensity(self, day, hour):
        return self.arrival_intensities[day][hour], self.depature_intensities[day][hour]
    

        #scenarioer target state:
                #scenario 1: 
                # hvis mellom kl 00 og 05, target state er forhåndsbestemt og alltid lik
                
                #scenario 2:
                # hvis bare positiv i feks 3 timer på rad 
                #- target state lik depature rate for de 3 timene

                #scenario 3:
                # hvis bare negativ i feks 3 timer på rad 
                #- absolutt sum av net_demand for de 3 timene

                #scenario 4:
                # hvis først positiv og så negativ 
                #- target state lik abs sum av negativ net demand (evt - positiv, dersom abs sum > positiv) helt til skrifter fortegn

                #scenario 5: 
                # hvis først negativ og så positiv 
                #-sum net demand for første negative

                #scenario 6:
                # hvis negativ så positv så negativ 
                #- første periode + net demand neste periode 

                #scenario 7: 
                # hvis positiv så negativ så positiv
    
                #scenario 8:
                # siste dagen og siste timen. Target state lik net_demand

class HexagonTargetState():
    def __init__(self, hex_id, arrival_intensities, depature_intensities):
        self.hex_id = hex_id
        self.average_arrival_intensities = arrival_intensities
        self.average_depature_intensities = depature_intensities
        self.target_states = [[0 for i in range(24)] for i in range(7)]
        self.initial_target_states = 0


    def calc_target_state(self):
        #target_states = [[0 for _ in range(24)] for _ in range(7)]  # Forbered en liste for å holde target states

        for day in range(7):  # 0 = Mandag, 6 = Søndag
            for hour in range(24):
                 # Scenario 10 - siste timen. Target state lik net_demand for den timen
                if hour == 23: #siste time
                            self.target_states[day][hour] = abs(min(round(self.average_arrival_intensities[day][hour] - self.average_depature_intensities[day][hour]), 0)) # Setter target state til net_demand
                # Scenario 1 - mellom kl 00 og 05. Target state er forhåndsbestemt og alltid lik            
                elif  0 <= hour <= 5: 
                    self.target_states[day][hour] = self.initial_target_states  # Forhåndsbestemt verdi
                    
                else:

                    all_next_demands = [] # Liste for å holde net_demand for fremtidige perioder
                    all_next_depature_rates = [] # Liste for å holde depature rate for fremtidige perioder    
                    next_demands = [] # Liste for å holde net_demand for fremtidige perioder
                    next_depature_rates = [] # Liste for å holde depature rate for fremtidige perioder
                    p = 0   # Startpunkt for perioder
                    periods = 3 # Antall perioder som skal inkluderes
                    current_day = day
                    next_hour = hour 
                    current_net_demand = self.average_arrival_intensities[day][hour] - self.average_depature_intensities[day][hour]

                    #MÅTE 1: finne net demand og depature rates for de neste periodene (stopper ikke når vi går forbi kl 23, kun hvis det er dag utenfor uken)
                    #while p < periods:
                    #    next_day, next_hour = (current_day, current_hour + 1) if current_hour < 23 else (current_day + 1, 0)
                    #    if next_day == 7: break #dersom neste periode er utenfor uken
                    #    next_net_demand = self.average_arrival_intensities[next_day][next_hour] - self.average_depature_intensities[next_day][next_hour] #legg inn riktig
                    #    next_depature_rate = self.average_depature_intensities[next_day][next_hour]
                    #    next_demands.append(next_net_demand)
                    #    next_depature_rates.append(next_depature_rate)
                    #    p += 1
                   
                   #MÅTE 2: finne alle net demand og alle depature rates for de neste periodene frem til kl 23 uansett (og tre perioder frem, for å bestemme scenario)
                    while next_hour < 23:
                        next_hour = next_hour + 1 
                        next_net_demand = self.average_arrival_intensities[current_day][next_hour] - self.average_depature_intensities[current_day][next_hour] #legg inn riktig
                        next_depature_rate = self.average_depature_intensities[current_day][next_hour]
                        all_next_demands.append(next_net_demand)
                        all_next_depature_rates.append(next_depature_rate)
                        if p < periods: 
                            next_demands.append(next_net_demand)
                            next_depature_rates.append(next_depature_rate)
                            p += 1
                       
                    while len(next_demands) < 3:
                        next_demands.append(0)
                        next_depature_rates.append(0)

                    # Scenario 2 (kun positive)
                    if all(next_demand >= 0 for next_demand in next_demands): #lik sum alle fremtidige depature rates før trend skifter
                        #TODO dobbeltsjekk
                        print("ppp", next_demands)
                        # amount_positive_periods = 0
                        # for next_demand in all_next_demands: 
                        #     if next_demand >= 0:
                        #         amount_positive_periods += 1
                        #     else: 
                        #         break
                        # Sum the departure rates for the positive demand periods
                        # sum_departure_rates = sum(all_next_depature_rates[i] for i in range(amount_positive_periods))
    
                        # Update the target state for the given day and hour with the absolute sum of departure rates
                        # self.target_states[day][hour] = round(abs(sum_departure_rates))
                        self.target_states[day][hour] = next_depature_rates[0]

                    # Scenario 3 (kun negative)
                    elif all(next_demand < 0 for next_demand in next_demands): #absolutt sum av net_demand for de 3 timene
                        print("nnn", next_demands)
                        sum_negative_periods = 0
                        for next_demand in all_next_demands:
                            if next_demand < 0:
                                sum_negative_periods += next_demand
                            else: 
                                break
                        self.target_states[day][hour] = abs(round(sum_negative_periods))
                        
                    # Scenario 4 (først positiv så negativ, negativ)
                    elif next_demands[1] < 0 and next_demands[2] < 0 and next_demands[0] >= 0:  #target state lik abs sum negati net demand - positiv net demand i starten helt til skrifter fortegn
                        print("pnn", next_demands)
                        sum_negative_periods = 0
                        for next_demand in all_next_demands[1:]:
                            if next_demand < 0:
                                sum_negative_periods += next_demand
                            else: 
                                break
                        self.target_states[day][hour] = abs(sum_negative_periods) - next_demands[0]
                        if self.target_states[day][hour] < 0:
                            self.target_states[day][hour] = 0

                    # Scenario 5 (først positiv, så positiv, negativ)
                    elif next_demands[2] < 0 and next_demands[1] >= 0 and next_demands[0] >= 0:
                        # print("ppn", next_demands)
                        sum_negative_periods = 0
                        for next_demand in all_next_demands[2:]:
                            if next_demand < 0:
                                sum_negative_periods += next_demand
                            else: 
                                break
                        self.target_states[day][hour] = abs(sum_negative_periods) - next_demands[0] - next_demands[1]
                        if self.target_states[day][hour] < 0:
                            self.target_states[day][hour] = 0

                    # Scenario 6 (først negativ så positiv, positiv)
                    elif next_demands[1] >=0 and next_demands[2] >= 0 and next_demands[0] < 0: 
                        print("npp", next_demands)
                        self.target_states[day][hour] = abs(next_demands[0])

                    # Scenario 7 (først negativ så negativ, positiv)
                    elif next_demands[2] >=0 and next_demands[1] < 0 and next_demands[0] < 0: 
                        print("nnp", next_demands)
                        self.target_states[day][hour] = abs(next_demands[0] + next_demands[1])
                
                    # Scenario 8 (negativ -> positiv -> negativ)
                    elif next_demands[0] < 0 and next_demands[1] >= 0 and next_demands[2] < 0:
                        print("npn", next_demands)
                        self.target_states[day][hour] = abs(next_demands[0])
                        sum_pos_neg = next_demands[1] + next_demands[2]
                        if sum_pos_neg < 0:
                            self.target_states[day][hour] += abs(sum_pos_neg)

                
                        
                    # Scenario 9 (positiv -> negativ -> positiv) #usikker på denne
                    elif next_demands[0] >= 0 and next_demands[1] < 0 and next_demands[2] >= 0:
                        print("pnp", next_demands)
                        sum_neg_pos = next_demands[0] + next_demands[1]
                        if sum_neg_pos < 0:
                            self.target_states[day][hour] = abs(sum_neg_pos)

                    

        return self.target_states


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
    
    @staticmethod
    def generate_hex_web_target_state_from_json(data):
        hexagons = []
        for hex_data in data['areas']:
            hex_id = hex_data['hex_id']
            arrival_intensity = hex_data['arrival_intensities']  # Bruker 'edges' som vertices
            leave_intensity = hex_data['depature_intensities']  # Bruker 'location' som center
            hexagon = HexagonTargetState(hex_id, arrival_intensity, leave_intensity)
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
    
    def calculate_traveltime(self, speed):
        traveltime_matrix = {}

        for i, hex1 in enumerate(self.hexagons):
            for hex2 in self.hexagons[i+1:]:  # Start from the next location to avoid duplicates
                distance = self._distance_to(hex1, hex2)
                travel_time = (distance / speed) * 60  # Calculate travel time in minutes
                # Set travel time for both directions due to symmetry
                traveltime_matrix[str(hex1.hex_id) + "_" + str(hex2.hex_id)] = travel_time
                traveltime_matrix[str(hex2.hex_id) + "_" + str(hex1.hex_id)] = travel_time
        return traveltime_matrix
    
    def _distance_to(self, hex1, hex2):
        return geopy.distance.distance(hex1.center, hex2.center).km

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
        # hex_dict = {hex.hex_id: 0 for hex in self.hexagons}
        for hex in self.hexagons:
            hex.move_probabilities = [[{} for i in range(24)] for i in range(7)]
            hex.move_probabilities_normalized = [[{} for i in range(24)] for i in range(7)]
    
    def find_arrival_depature_intensities(self, data):
        for trip in data['data']:
            start_coords = trip["route"]["features"][0]["geometry"]["coordinates"]
            end_coords = trip["route"]["features"][-1]["geometry"]["coordinates"]
            start_time = trip["start_time"]
            end_time = trip["end_time"]
            
            self.add_trip_to_stats(start_coords, end_coords, start_time, end_time)
