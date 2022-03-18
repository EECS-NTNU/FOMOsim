# parse.py
import json
# from os import stat
# import requests
import sys
#import numpy as np

# TODO inn som classe og objekt ??? @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def timeInHoursAndMinutes(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return str(h) + ":" + str(m)


def summary(filename):
    JSON_file = open(filename, "r")
    bikeData = json.loads(JSON_file.read())
    print("Summary of bike trips read from " + filename)
    print("* ", len(bikeData), "trips")
    stations = []
    stationNames = {} # id to name mapping
    tripDurations = [] # not sure if this is needed
    shortestTrip = sys.maxsize
    longestTrip = 0
    for i in range(len(bikeData)):
        startStation_id = int(bikeData[i]["start_station_id"])
        stations.append(startStation_id)
        startStation_name = bikeData[i]["start_station_name"]
        stationNames[startStation_id] = startStation_name
        endStation_id = int(bikeData[i]["end_station_id"])
        stations.append(endStation_id)
        endStation_name = bikeData[i]["end_station_name"]
        stationNames[endStation_id] = endStation_name
        duration = bikeData[i]["duration"] # in seconds
        if duration > longestTrip:
            longestTrip = duration
        elif duration < shortestTrip:
            shortestTrip = duration
        tripDurations.append(duration)

    print("* ", len(set(stations)), " different stations used")
    dictionary_items = stationNames.items()
    sorted_items = sorted(dictionary_items)
    stationsMap = open("report/stations.txt", "w")
    count = 0
    for item in sorted_items:
        str = f'{count:>5}'+ f'{item[0]:>6}' + "  " + item[1] + "\n"
        # print(str, end ='')
        stationsMap.write(str)
        count = count + 1
    stationsMap.close()

    print("  - shortest trip was ", timeInHoursAndMinutes(shortestTrip), "(hours:minutes)" )
    print("  - longest trip was ", timeInHoursAndMinutes(longestTrip), "(hours:minutes)" )
    print("  - id ==> name mapping stored in report/stations.txt")

# inspired by https://realpython.com/python-json/
def sandboxTest():

    # data = requests.get("https://data.urbansharing.com/oslobysykkel.no/trips/v1/2022/01.json") # 15378 trips
    # bikeData = json.loads(data.text)
    # print(len(bikeData), "trips")
    # print("bikeData read")
    # trips = []
    # for i in range(len(bikeData)):
    #     dateString = bikeData[i]["started_at"][11:19]
    #     #print(dateString)
    #     #print("hours", dateString[0:2])
    #     #print("mins", dateString[3:5])
    #     minTime = int(dateString[0:2])*60 + int(dateString[3:5])
    #     #print(minTime)
    #     trips.append(minTime)
    # print(trips)

    # data = requests.get("https://data.urbansharing.com/oslobysykkel.no/trips/v1/2021/06.json") # 217255 trips
    # bikeData = json.loads(data.text)
    # print(len(bikeData), " trips")
    # print("bikeData read")
    pass

#sandboxTest()
