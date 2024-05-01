
WEEKS = 3

import random
import h3
from random import randint, sample
import requests
from shapely.geometry import Point, Polygon
from datetime import datetime, timezone
import numpy as np
import geopy.distance
import folium
import json
from json_settings import *
import gzip

class HexWeb:
    def __init__(self, lat, lng):
        self.lat = lat
        self.lng = lng
        self.center = lat, lng
        self.hexagons = None

    
    def generate_hex_web_from_json(self, data):
        hexagons = []
        hex_id = 0
        for hex_data in data['areas']:
            #hex_id = hex_data['hex_id']
            hex_id += 1
            vertices = hex_data['edges']  # Bruker 'edges' som vertices
            center = hex_data['location']  # Bruker 'location' som center
            hexagon = Hexagon(hex_id, vertices, center)
            hexagons.append(hexagon)
        self.hexagons = hexagons
        return hexagons

    def map(self): 
        # Create a map centered around the given coordinates

        mymap = folium.Map(location=[self.lat, self.lng], zoom_start=13)  # Adjusted zoom level


        lat, lng = 63.446827, 10.421906
        resolution = 11 # Adjusted resolution level

        # Generate H3 index for the given coordinates at the specified resolution
        index = h3.geo_to_h3(lat, lng, resolution)

        # Determine the hexagons within 1 ring from the center
        hexagons = h3.k_ring(index, 100)

        # Plot the hexagons on the map
        for hex in hexagons:
            # Get the vertices of the hexagon
            vertices = h3.h3_to_geo_boundary(hex, geo_json=True)
            # Folium requires reversing coordinates due to GeoJSON standard (lng, lat)
            vertices = [(v[1], v[0]) for v in vertices]
            # Create a hexagon object
            # Create a polygon and add to the map
            folium.Polygon(locations=vertices, fill=True, color="#ff7800", weight=2).add_to(mymap)

        return mymap
    # Display the map

    def plot_and_save_depature_hexs(self, mymap):
        # Create a map centered around the given coordinates
        # Read JSON data from file
        with open('/Users/elinehareide/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Eline - Høst 2023/Prosjektoppgave/fomo/instances/Ryde/trip_data_h6_h8_4_24_sept.json', 'r') as file:
            data = json.load(file)

        # Plot trip coordinates as markers on the map
        for trip in data['data']['trips']:
            feature = trip['route']['features'][0]
            lng, lat = feature['geometry']['coordinates']
            #folium.Marker(location=[lat, lng], popup=trip['trip_id'], icon=folium.Icon(color='red')).add_to(mymap)
            folium.CircleMarker(location=[lat, lng], radius=2, color='green', fill=True, fill_color='green').add_to(mymap)
            

            start_hex = None
            start_coords = trip["route"]["features"][0]["geometry"]["coordinates"]
            start_coords = (start_coords[1], start_coords[0])

            for hex in self.hexagons:
                if hex.is_point_inside(start_coords):
                    start_hex = hex
                
            if start_hex is not None:
                start_hex.count_depatures()

    
                #end_coords = trip["route"]["features"][-1]["geometry"]["coordinates"]
                #start_time = trip["start_time"]
                #end_time = trip["end_time"]

            #feature = trip['route']['features'][1]
            #lng, lat = feature['geometry']['coordinates']
            #folium.Marker(location=[lat, lng], popup=trip['trip_id'], icon=folium.Icon(color='blue')).add_to(mymap)
            #folium.CircleMarker(location=[lat, lng], radius=2, color='red', fill=True, fill_color='red').add_to(mymap)
        
        mymap.show()
    

class Hexagon:
    "Lager et område, brukes til å telle arrivals/departures og move probabilities"
    def __init__(self, hex_id, vertices, center):
        self.hex_id = hex_id
        self.vertices = vertices
        self.center = center
        self.depatures = 0
    
    def count_depatures(self):
        self.depatures += 1


    def is_point_inside_boundingbox(self, bbox):
        lat, lon = self.center
        min_lon, max_lon, min_lat, max_lat = bbox
        return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat

    def is_point_inside(self, location):
        polygon = Polygon(self.vertices)
        return polygon.contains(Point(location))
    

    

hex_data = None
FINISHED_DATA_FILE = "/Users/elinehareide/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Eline - Høst 2023/Prosjektoppgave/fomo/instances/Ryde/TD_W19_test_W3.json.gz"
with gzip.open(FINISHED_DATA_FILE, 'rt', encoding='utf-8') as f:
    hex_data = json.load(f)


hexweb = HexWeb(63.431711, 10.403537)
hexagons = hexweb.generate_hex_web_from_json(hex_data)
mymap = hexweb.map()
mymap
#hexweb.plot_and_save_depature_hexs(mymap)

