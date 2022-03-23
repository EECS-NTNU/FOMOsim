# parse.py
from importlib.metadata import distribution
import json
from math import dist
import sys
import os.path
import geopy.distance
from datetime import datetime

# TODO inn som classe og objekt ??? @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def timeInHoursAndMinutes(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return str(h) + ":" + str(m)

def printTime():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Time =", current_time)

def summary(filepath):
    JSON_file = open(filepath, "r")
    bikeData = json.loads(JSON_file.read())
    print("Summary of bike trips read from " + filepath)
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

def checkAll(folder):
#    print("checkAll pressed")
    try:
        file_list = os.listdir(folder)
        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".json"))
        ]
        print("checkAll preliminary solution is summary of all JSON files in folder")
        for name in fnames:
            print("checks ", name)
            summary(os.path.join(folder, name))    
    except:
        print("*** USER ERROR *** CheckAll called on empty folder")

class Station:
    def __init__(self, stationId, longitude, latitude, stationName):
        self.stationId = stationId
        self.longitude = longitude
        self.latitude = latitude
        self.stationName = stationName

class BikeTrip:
    def __init__(self, startTime, endTime, duration, start, end):
        self.start = startTime
        self.end = endTime
        self.duration = duration
        self.start = start
        self.end = end
        
    

def reportStations(stations):
    stationsMap = open("tripStats/report/stations.txt", "w")
    count = 0
    for s in stations:
        stationId = f'{count:>5}'+ f'{s.stationId:>6}' + " " + s.longitude + " " + s.latitude + " " + s.stationName + "\n"
        # print(stationId, end ='')
        stationsMap.write(stationId)
        count = count + 1
    stationsMap.close()

def checkTest():  # used for code-developmen and testing
    # storedData = "tripStats/data/Oslo/Oslo-2019-12.json" 
    storedData = "tripStats/data/Utopia/Utopia-Test-1.json" 
    print("Parse data from file: ", storedData, " : ", end= '')
    jsonFile = open(storedData, "r")
    bikeData = json.loads(jsonFile.read())
    print(len(bikeData), "trips, ", end = '')

    stationNo = 0 # station numbers found, counts 0,1,2,...
    stations = []
    stationMap = {} # maps from stationId to station number 

    for i in range(len(bikeData)):
        startId = int(bikeData[i]["start_station_id"])
        if not startId in stationMap:
            stationMap[startId] = stationNo
            stationNo = stationNo + 1
            startLong = str(bikeData[i]["start_station_longitude"])
            startLat = str(bikeData[i]["start_station_latitude"])
            stations.append(Station(bikeData[i]["start_station_id"], startLong, startLat, bikeData[i]["start_station_name"]))
        
        endId = int(bikeData[i]["end_station_id"])
        if not endId in stationMap:
            stationMap[endId] = stationNo
            stationNo = stationNo + 1
            endLong = str(bikeData[i]["end_station_longitude"])
            endLat = str(bikeData[i]["end_station_latitude"])
            stations.append(Station(bikeData[i]["end_station_id"], endLong, endLat, bikeData[i]["end_station_name"]))

    print(len(set(stations)), " stations used") # debug
    reportStations(stations)
 
    # calculate distance matrix
    dist_matrix_km = [] # km in kilometers, integers to comply with main.py
    for rowNo in range(len(stationMap)):
        col = 0
        row = []
        for col in range(len(stationMap)):
            # indices = str(rowNo) + " - " + str(col) # was for debug
            startLong = stations[rowNo].longitude
            startLat = stations[rowNo].latitude
            endLong = stations[col].longitude
            endLat = stations[col].latitude
            distance_float = geopy.distance.distance((startLat, startLong), (endLat, endLong)).km
            # print(distance_float) #debug
            km = round(distance_float)
            row.append(km)
            col = col + 1
        dist_matrix_km.append(row)
        rowNo = rowNo + 1
#   
def calcIntensity(city, mode, periodLength):
    if (city == "Oslo") and (mode == "all"):
        print("leave and arrive intensity calculation for all trips downloaded for Oslo ...")
        printTime()
        stationMap = {} # maps from stationId to station number 
        stationsFile = open("tripStats/report/stations.txt", "r")
        for line in stationsFile.readlines():
            words = line.split()
            stationMap[words[1]] = words[0]
    
        leaveCount = [] # list of leave counts indexed by station number    
        arriveCount = [] # list of arrive counts indexed by station number
        for no in range(len(stationMap)):
            leaveCount.append(0)
            arriveCount.append(0) # TODO last one reached, + 1 ?

        trips = 0
        # refactor code below used several places to loop through all OsloFiles, make function, retrn list of filenames    
        for i in range(1, 35 + 1): # TODO assumes at least 35 trip data files for Oslo # TODO, ugly, magic numbers
            monthNo = 1 + ((i + 2) % 12) # 1 is April
            yearNo = (i + 2)//12 + 2019
            if monthNo < 10: # refactor into function zeroPad
                monthStr = "0" + str(monthNo)
            else:
                monthStr = str(monthNo)    
            jsonFile = open("tripStats/data/Oslo/Oslo-" + str(yearNo) + "-" + monthStr  + ".json", "r")
            bikeData = json.loads(jsonFile.read())
            for i in range(len(bikeData)):
                startNo = int(stationMap[bikeData[i]["start_station_id"]])
                leaveCount[startNo] = leaveCount[startNo] + 1
                endNo = int(stationMap[bikeData[i]["end_station_id"]])
                arriveCount[endNo] = arriveCount[endNo] + 1
                trips = trips + 1
            print(".", end='') # TODO replace with progress bar
  
        # adjust pr. hour or pr. 20 min, assumes 30 days in all months, and 35 months
        periods = 35 * 30 * 24 * (60 / periodLength)
        leaveIntensity = []
        arriveIntensity = []
        for i in range(len(stationMap)):
            leaveIntensity.append(leaveCount[i]/periods)
            arriveIntensity.append(arriveCount[i]/periods)
        printTime()
        print("A total of ", trips, " trips processed")
        leaveIntenseFileName = "Oslo-li-1-35.txt"
        leaveFile = open("tripStats/data/Oslo/" + leaveIntenseFileName, "w")
        arriveIntenseFileName = "Oslo-ai-1-35.txt"
        arriveFile = open("tripStats/data/Oslo/" + arriveIntenseFileName, "w")
        for i in range(len(stationMap)):
            leaveFile.write(str(leaveIntensity[i]) + " ") 
            arriveFile.write(str(arriveIntensity[i])+ " ") 
        print("calcIntensity ends, leave and arrive intensity stored in " + leaveIntenseFileName + " and " + leaveIntenseFileName + "respectively")
        return leaveIntensity, arriveIntensity         

    else:
        print("*** ERROR: calcIntensity --- illegal parameters")

def calcDistances(city, mode):
    if (city == "Oslo") and (mode == "all"):
        print("distance matrix calculation for stations used in all trips downloaded for Oslo ...")
        printTime() 
        stationNo = 0 # station numbers found, counts 0,1,2,...
        stations = []
        stationMap = {} # maps from stationId to station number 
        for i in range(1, 35 + 1): # TODO assumes at least 35 trip data files for Oslo # TODO, ugly, magic numbers
            monthNo = 1 + ((i + 2) % 12) # 1 is April
            yearNo = (i + 2)//12 + 2019
            if monthNo < 10: # refactor into function zeroPad
                monthStr = "0" + str(monthNo)
            else:
                monthStr = str(monthNo)    
            jsonFile = open("tripStats/data/Oslo/Oslo-" + str(yearNo) + "-" + monthStr  + ".json", "r")
            bikeData = json.loads(jsonFile.read())
            for i in range(len(bikeData)):
                startId = int(bikeData[i]["start_station_id"])
                if not startId in stationMap:
                    stationMap[startId] = stationNo
                    stationNo = stationNo + 1
                    startLong = str(bikeData[i]["start_station_longitude"])
                    startLat = str(bikeData[i]["start_station_latitude"])
                    stations.append(Station(bikeData[i]["start_station_id"], startLong, startLat, bikeData[i]["start_station_name"]))
            
                endId = int(bikeData[i]["end_station_id"])
                if not endId in stationMap:
                    stationMap[endId] = stationNo
                    stationNo = stationNo + 1
                    endLong = str(bikeData[i]["end_station_longitude"])
                    endLat = str(bikeData[i]["end_station_latitude"])
                    stations.append(Station(bikeData[i]["end_station_id"], endLong, endLat, bikeData[i]["end_station_name"]))
            print(".", end='') # TODO replace with progress bar
        print("A total of ", len(set(stations)), " stations used, reported on stations.txt")
        reportStations(stations)
        printTime()
        # calculate distance matrix
        dist_matrix_km = [] # km in kilometers, integers to comply with main.py
        dm_fileName = "Oslo-dm-1-35.txt"
        dm_file = open("tripStats/data/Oslo/"+ dm_fileName, "w")
        for rowNo in range(len(stationMap)):
            col = 0
            row = []
            for col in range(len(stationMap)):
                startLong = stations[rowNo].longitude
                startLat = stations[rowNo].latitude
                endLong = stations[col].longitude
                endLat = stations[col].latitude
                distance_float = geopy.distance.distance((startLat, startLong), (endLat, endLong)).km
                # print(distance_float) #debug
                km = round(distance_float)
                row.append(km)
                dm_file.write(str(km))
                dm_file.write(" ")
                col = col + 1
            dist_matrix_km.append(row)
            dm_file.write("\n")
            rowNo = rowNo + 1    
        printTime()
        print("calcDistances ends, stored in " + dm_fileName)
        return dist_matrix_km
    else:
        print("*** ERROR: calcDistances --- illegal parameters")

##################################
# # inspired by https://realpython.com/python-json/
# def sandboxTest():

#     data = requests.get("https://data.urbansharing.com/oslobysykkel.no/trips/v1/2022/01.json") # 15378 trips
#     dataOut = open("out.json", "w")
#     dataOut.write(data.text)
#     dataOut.close()
#     bikeData = json.loads(data.text)
#     print(len(bikeData), "trips")
#     print("bikeData read")
#     trips = []
#     for i in range(len(bikeData)):
#         dateString = bikeData[i]["started_at"][11:19]
#         #print(dateString)
#         #print("hours", dateString[0:2])
#         #print("mins", dateString[3:5])
#         minTime = int(dateString[0:2])*60 + int(dateString[3:5])
#         #print(minTime)
#         trips.append(minTime)
#     print(trips)

#     data = requests.get("https://data.urbansharing.com/oslobysykkel.no/trips/v1/2021/06.json") # 217255 trips
#     bikeData = json.loads(data.text)
#     print(len(bikeData), " trips")
#     print("bikeData read")
#     pass

# sandboxTest()
