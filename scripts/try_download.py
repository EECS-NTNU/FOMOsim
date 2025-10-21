import gzip
import json
import os

dir_path = '../instances/Ryde/travel'
master_path = '../instances/Ryde/travel/master.json.gz'

# dir_path = '/Users/isabellam/NTNU/H2023/Prosjektoppgave/fomo/instances/Ryde/travel'
# master_path = '/Users/isabellam/NTNU/H2023/Prosjektoppgave/fomo/instances/Ryde/travel/master.json.gz'


master_data = {'travel_matrix': {}, 'travel_vehicle_matrix': {}}

for i, filename in enumerate(os.listdir(dir_path)):
    if filename == "master.json.gz":
         continue
    file_path = dir_path + '/' + filename
    with gzip.open(file_path, 'r') as file:
        data = json.load(file)
    
    master_data['travel_matrix'].update(data['travel_matrix'])
    master_data['travel_vehicle_matrix'].update(data['travel_vehicle_matrix'])

    if i % 5 == 0:
        with gzip.open(master_path, 'r') as file:
            data2 = json.load(file)
        master_data['travel_matrix'].update(data2['travel_matrix'])
        master_data['travel_vehicle_matrix'].update(data2['travel_vehicle_matrix'])

        with gzip.open(master_path, 'wt', encoding='utf-8') as file:
            json.dump(master_data, file, ensure_ascii=False, indent=4)
        master_data = {'travel_matrix': {}, 'travel_vehicle_matrix': {}}
    
    print(len(master_data['travel_matrix']), filename)


# import gzip
# import json
# import os

# json_path = '../instances/Ryde/TD_W19_test_W3.json.gz'
# master_path = '../instances/Ryde/travel/master.json.gz'
# master_path = '/Users/isabellam/NTNU/H2023/Prosjektoppgave/fomo/instances/Ryde/travel/matrix_0.json.gz'


# with gzip.open(master_path, 'wt', encoding='utf-8') as file:
#         json.dump({'travel_matrix': {}, 'travel_vehicle_matrix': {}}, file, ensure_ascii=False, indent=4)

# with gzip.open(master_path, 'r') as file:
#     data = json.load(file)

# with gzip.open(json_path, 'r') as f:
#     json_data = json.load(f)

# json_data['traveltime_matrix'] = data['travel_matrix']
# json_data['traveltime_vehicle_matrix'] = data['travel_vehicle_matrix']

# final_path = '../instances/Ryde/TD_W19_final_W3.json.gz'
# with gzip.open(final_path, 'wt', encoding='utf-8') as file:
    # json.dump(json_data, file, ensure_ascii=False, indent=4)