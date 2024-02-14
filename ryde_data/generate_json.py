#     "end_time": f'{year}-{month if month < 10 else "0"+str(month)}-{day if day < 10 else "0"+str(day)}T{hour if hour < 10 else "0"+str(hour)}'
import requests
import json
import datetime

url = "https://prod-api.ryde.vip/mds/bcd3af0f-d606-443c-b862-ed47f236fdc8/status_changes"

year = 2023
month = 10
day = 1
hour = 0

all_data = []
headers = {
    'Authorization': 'Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJOVE5VIiwicGFzc3dvcmQiOiJSeWRlMjAyMzExMjMwOTI3NDMyMDIzMTEyMzA5Mjc0MzIwMjMxMTIzMDkyNzQzIiwiaWQiOiI0MCIsImlhdCI6MTcwMDcyODA2MywianRpIjoiNmIzODRhOWEtOGRhNC00NmQ5LWFlODktNDE0NDk4MDhkYmQyIiwidXNlcm5hbWUiOiJOVE5VIn0.zCXzrdG86AHcMecL1r8xpzvCj7W9FMU6CwTf7Z9f58A',
    'x-api-key': 'pcRuYOEmc84Ycz7zjilqN4T3zZ9BBfSr1vIudz5X',
    'Cookie': 'AWSALB=hDVKywDuG3fU5bvULzqrQypxacbm2c4mXgITrg05uLwpvv4n1wmMQIvg90kJoc+EI1Nz3YkwBn7/whE2nq0gAIONAC0OoZ7a3eUq0mqo5a+/iwqdtK4emFUuFRUE; AWSALBCORS=hDVKywDuG3fU5bvULzqrQypxacbm2c4mXgITrg05uLwpvv4n1wmMQIvg90kJoc+EI1Nz3YkwBn7/whE2nq0gAIONAC0OoZ7a3eUq0mqo5a+/iwqdtK4emFUuFRUE; JSESSIONID=5D07F6F1BBE2448400397D073B98C43E'
    }

# Loop until the year changes
while year == 2023:
    formatted_time = f'{year}-{month:02d}-{day:02d}T{hour:02d}'
    payload = {'event_time': formatted_time}

    # Make the request
    response = requests.request("POST", url, headers=headers, data=payload)

    # Process the response
    if response.ok:
        data = response.json()

        # Check for the 'trips' key
        if "data" in data and "status_changes" in data["data"]:
            # Create a new entry with end_time and trips
            new_entry = {
                "event_time": formatted_time,
                "trips": data["data"]["status_changes"]
            }
            all_data.append(new_entry)
        else:
            print(f"No 'status_changes' in response data for payload: {payload}")
    else:
        print(f"Failed to retrieve data: {response.status_code}. {payload}")

    # Increment the time by one hour
    new_time = datetime.datetime(year, month, day, hour) + datetime.timedelta(hours=1)
    year, month, day, hour = new_time.year, new_time.month, new_time.day, new_time.hour

# After finishing the loop, write the structured data to a file
with open('ryde_data/status_change_2023_octdec.json', 'w') as f:
    json.dump({"data": all_data}, f, indent=4)

print("The structured data has been saved to response.json")