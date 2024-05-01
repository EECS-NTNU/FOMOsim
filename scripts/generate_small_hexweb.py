#MAIN
from hexweb import Hexagon, HexWeb
import json
import gzip
from json_settings import *


bbox = [10.350317, 10.489613, 63.396206, 63.457983] #Trondheim



FINISHED_DATA_FILE = "/Users/elinehareide/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Eline - Høst 2023/Prosjektoppgave/fomo/instances/Ryde/TD_W19_test_W3.json.gz"
with gzip.open(FINISHED_DATA_FILE, 'rt', encoding='utf-8') as f:
    hex_data = json.load(f)

hexagons = Hexweb.generate_hex_web_from_json_small(hex_data,bbox)