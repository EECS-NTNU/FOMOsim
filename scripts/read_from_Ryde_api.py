import requests
import json
import datetime
from json_settings import *

def read_write_ryde_data():
    # url = "https://prod-api.ryde.vip/mds/bcd3af0f-d606-443c-b862-ed47f236fdc8/status_changes"
    url = "https://prod-api.ryde.vip/mds/bcd3af0f-d606-443c-b862-ed47f236fdc8/trips"


    all_data = []
    headers = {
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJOVE5VIiwicGFzc3dvcmQiOiJSeWRlMjAyMzExMjMwOTI3NDMyMDIzMTEyMzA5Mjc0MzIwMjMxMTIzMDkyNzQzIiwiaWQiOiI0MCIsImlhdCI6MTcwMDcyODA2MywianRpIjoiNmIzODRhOWEtOGRhNC00NmQ5LWFlODktNDE0NDk4MDhkYmQyIiwidXNlcm5hbWUiOiJOVE5VIn0.zCXzrdG86AHcMecL1r8xpzvCj7W9FMU6CwTf7Z9f58A',
        'x-api-key': 'pcRuYOEmc84Ycz7zjilqN4T3zZ9BBfSr1vIudz5X',
        'Cookie': 'AWSALB=hDVKywDuG3fU5bvULzqrQypxacbm2c4mXgITrg05uLwpvv4n1wmMQIvg90kJoc+EI1Nz3YkwBn7/whE2nq0gAIONAC0OoZ7a3eUq0mqo5a+/iwqdtK4emFUuFRUE; AWSALBCORS=hDVKywDuG3fU5bvULzqrQypxacbm2c4mXgITrg05uLwpvv4n1wmMQIvg90kJoc+EI1Nz3YkwBn7/whE2nq0gAIONAC0OoZ7a3eUq0mqo5a+/iwqdtK4emFUuFRUE; JSESSIONID=5D07F6F1BBE2448400397D073B98C43E'
        }

    # Loop until the year changes
    while day < 8 + WEEKS * 7: # Endre denne!!!
        formatted_time = f'{year}-{month:02d}-{day:02d}T{hour:02d}'

        # payload = {'event_time': formatted_time}
        payload = {'end_time': formatted_time}

        response = requests.request("POST", url, headers=headers, data=payload)

        if response.ok:
            data = response.json()

            if "data" in data and "trips" in data["data"]:
                all_data.extend(data["data"]["trips"])

            # if "data" in data and "status_changes" in data["data"]:
            #     # Create a new entry with end_time and trips
            #     new_entry = {
            #        "event_time": formatted_time,
            #        "trips": data["data"]["status_changes"]
            #     }
            #     all_data.append(new_entry)

        new_time = datetime.datetime(year, month, day, hour) + datetime.timedelta(hours=1)
        year, month, day, hour = new_time.year, new_time.month, new_time.day, new_time.hour

    with open(RYDE_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump({"data": all_data}, f, indent=4)