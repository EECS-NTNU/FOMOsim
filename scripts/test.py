import random
from random import randint, sample
from datetime import datetime, timezone
import json
from json_settings import *
import gzip
import math

with gzip.open('C:/Users/itlam/fomo/instances/Ryde/finished_target_state_1066.json.gz', 'r') as file:
    data = json.load(file)

counter = 0 
final_target_states = [[0 for j in range(24)] for i in range(7)]
hex_data_dict = {}
targert_states = [[0 for i in range(24)] for j in range(7)]
     
for hex_data in data['areas']:  
    
    hex_final_target_states = hex_data['target_states']
    for i in range(7):
        for j in range(24):
            final_target_states[i][j] += hex_final_target_states[i][j]
print("alle hex ferdig")
#print(final_target_states)
for hex_data in data['areas']:
    hex_id = hex_data['id']
    hex_data_dict[hex_id] = hex_data['target_states']
    
        # Plot trip coordinates as markers on the map
#print(final_target_states)


for i in range(7):
    print("day:", i)
    for j in range(24):
        print("hour:", j)
        while final_target_states[i][j] != 1066:
            if final_target_states[i][j] > 1066:
                for hex_target_state in hex_data_dict.keys():
                    if hex_data_dict[hex_target_state][i][j] > 0:
                        hex_data_dict[hex_target_state][i][j] -= 1
                        final_target_states[i][j] -= 1
                        break
            elif final_target_states[i][j] < 1066:
                for hex_target_state in hex_data_dict.keys():
                    if hex_data_dict[hex_target_state][i][j] > 0:
                        hex_data_dict[hex_target_state][i][j] += 1
                        final_target_states[i][j] += 1
                        break

print(final_target_states)

json_data = {
        'areas': [
        {
            'id': hex_target_state,  # A0
            # 'location': hexagon.center,  # [lat, lon]
            # 'initial_target_states': hexagon.average_initial_target_state,
            # 'round_initial_target_states': hexagon.round_average_initial_target_state,  # [[{id: value, ...} * 24] * 7]
            'target_states': hex_data_dict[hex_target_state],
        }
        for hex_target_state in hex_data_dict.keys()
    ],
      
        'total_average_initial_state': final_target_states
}
print("skriver til fil")
        
with open('C:/Users/itlam/fomo/instances/Ryde/finished_target_state_1066_all.json', 'w') as file:
    json.dump(json_data, file, indent=4)
print("skrevet til fil")
