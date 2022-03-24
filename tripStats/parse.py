# parse.py
from importlib.metadata import distribution
import json
from math import dist
import sys
import os.path
import geopy.distance
from helpers import *

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
        
def reportStations(stations, city):
    fileName = "tripStats/data/" + city + "/stations.txt"
    stationsMap = open(fileName, "w")
    count = 0
    for s in stations:
        stationId = f'{count:>5}'+ f'{s.stationId:>6}' + " " + s.longitude + " " + s.latitude + " " + s.stationName + "\n"
        # print(stationId, end ='')
        stationsMap.write(stationId)
        count = count + 1
    stationsMap.close()

def checkTest():  # TODO prelim used for code-development and testing
    return
    # storedData = "tripStats/data/Oslo/Oslo-2019-12.json" 
    storedData = "tripStats/data/Utopia/Utopia-Test-1.json" 
    print("Parse data from file: ", storedData, " : ", end= '')
    jsonFile = open(storedData, "r")
    bikeData = json.loads(jsonFile.read())
    # print(len(bikeData), "trips, ", end = '')

def calcDistances(city):
    print("calDistance() called for city: " + city) 
    printTime() 

    stationNo = 0 # station numbers found, counts 0,1,2,...
    stations = []
    stationMap = {} # maps from stationId to station number 

    tripDataPath = "tripStats/data/" + city + "/tripData"
    fileList = os.listdir(tripDataPath)
    for file in fileList:
        jsonFile = open(os.path.join(tripDataPath, file), "r")
        bikeData = json.loads(jsonFile.read())
        for i in range(len(bikeData)):
            startId = int(bikeData[i]["start_station_id"])
            if not startId in stationMap:
                stationMap[startId] = stationNo
                stationNo = stationNo + 1
                startLong = str(bikeData[i]["start_station_longitude"])
                startLat = str(bikeData[i]["start_station_latitude"])
                stations.append(Station(bikeData[i]["start_station_id"], startLong, startLat, bikeData[i]["start_station_name"]))
        
            endId = int(bikeData[i]["end_station_id"]) # TODO refactor?, code similar for start... and end...
            if not endId in stationMap:
                stationMap[endId] = stationNo
                stationNo = stationNo + 1
                endLong = str(bikeData[i]["end_station_longitude"])
                endLat = str(bikeData[i]["end_station_latitude"])
                stations.append(Station(bikeData[i]["end_station_id"], endLong, endLat, bikeData[i]["end_station_name"]))
        print(".", end='') # TODO nice, replace with progress bar
    print("A total of ", len(set(stations)), " stations used, reported on stations.txt")
    reportStations(stations, city)
    printTime()
    # calculate distance matrix
    dist_matrix_km = [] # km in kilometers, integers to comply with main.py
    
    dm_file = open("tripStats/data/" + city + "/Distances.txt", "w")
    for rowNo in range(len(stationMap)):
        col = 0 
        row = []
        for col in range(len(stationMap)):
            dm_file.write(str(geopy.distance.distance(
                (stations[rowNo].latitude, stations[rowNo].longitude), 
                (stations[col].latitude, stations[col].longitude)).km))
            dm_file.write(" ")
            col = col + 1
        dist_matrix_km.append(row)
        dm_file.write("\n")
        rowNo = rowNo + 1    
    printTime()
    print("Distances calculated, stored in Distances.txt and returned thru call")
    return dist_matrix_km

def readStationMap(city):
    stationMap = {} # maps from stationId to station number 
    stationsFile = open("tripStats/data/" + city + "/stations.txt", "r")
    for line in stationsFile.readlines():
        words = line.split()
        stationMap[words[1]] = int(words[0])
    return stationMap

def analyzeTraffic(city, week):
    if (week < 1) or (week >53):
        print("*** Error: week no must be in range 1..53")
    print("Starts analyzing traffic for city: " + city) 
    printTime() 
    years = [] # Must count no of "year-instances" of the given week that are analyzed

    stationMap = readStationMap(city)

    arriveCount = []
    leaveCount = []
    moveCount = []
    durations = []
    for station in range(len(stationMap)):
        arriveCount.append([])
        leaveCount.append([])
        moveCount.append([])
        durations.append([])
        for day in range(7):
            arriveCount[station].append([])
            leaveCount[station].append([])
            moveCount[station].append([])
            for hour in range(24):
                arriveCount[station][day].append(0)
                leaveCount[station][day].append(0)
                stationList = []
                for i in range(len(stationMap)): # TODO, can probably be done more compactly
                    stationList.append(0)
                    durations[station].append([])
                moveCount[station][day].append(stationList)

    trips = 0
    arrivingBikes = 0
    leavingBikes = 0
    tripDataPath = "tripStats/data/" + city + "/tripData"
    fileList = os.listdir(tripDataPath)
    for file in fileList:
        jsonFile = open(os.path.join(tripDataPath, file), "r")
        bikeData = json.loads(jsonFile.read())
        for i in range(len(bikeData)):

            year, weekNo, weekDay = yearWeekNoAndDay(bikeData[i]["ended_at"][0:10])
            years.append(year)
            hour = int(bikeData[i]["ended_at"][11:13])
            endStationNo = 0 #
            if weekNo == week:
                endStationNo = stationMap[bikeData[i]["end_station_id"]]
                arriveCount[endStationNo][weekDay][hour] += 1
                arrivingBikes += 1

            # startString = bikeData[i]["started_at"] debug
            year, weekNo, weekDay = yearWeekNoAndDay(bikeData[i]["started_at"][0:10])
            years.append(year)
            hour = int(bikeData[i]["started_at"][11:13])
            if weekNo == week:
                startStationNo = stationMap[bikeData[i]["start_station_id"]]
                leaveCount[startStationNo][weekDay][hour] += 1
                moveCount[startStationNo][weekDay][hour][endStationNo] += 1    
                leavingBikes += 1
                durations[startStationNo][endStationNo].append(bikeData[i]["duration"])
            trips = trips + 1
        print(".", end='') # TODO replace with progress bar
    avgDuration = []
    for start in range(len(stationMap)):
        avgDuration.append([])
        for end in range(len(stationMap)):
            avgDuration[start].append([])
            sumDuration = 0
            for trip in range(len(durations[start][end])):
                sumDuration += durations[start][end][trip]
            avgDuration[start][end] = sumDuration/len(durations)
    distances = calcDistances(city)
   
    speed_matrix = []
    for start in range(len(stationMap)):
        speed_matrix.append([])
        for end in range(len(stationMap)):
            speed_matrix[start].append(distances[start][end]/(avgDuration[start][end]/3600))
 
    noOfYears = len(set(years))
    arrive_intensities = []
    leave_intensities = []
    move_probabilities= []
    for station in range(len(stationMap)):
        arrive_intensities.append([])
        leave_intensities.append([])
        move_probabilities.append([])
        for day in range(7):
            arrive_intensities[station].append([])
            leave_intensities[station].append([])
            move_probabilities[station].append([])
            for hour in range(24):
                arrive_intensities[station][day].append(arriveCount[station][day][hour]/noOfYears)
                leave_intensities[station][day].append(leaveCount[station][day][hour]/noOfYears)
                move_probabilities[station][day].append([])
                for endStation in range(len(stationMap)):
                    movedBikes = moveCount[station][day][hour][endStation]
                    movedBikesTotal = leaveCount[station][day][hour]
                    if movedBikesTotal > 0:
                        move_probabilities[station][day][hour].append(movedBikes/movedBikesTotal)
                    else:
                        move_probabilities[station][day][hour].append(0.0)    

    print(trips, " trips analyzed. A total of ", leavingBikes, " bikes left and ", end='')
    print(arrivingBikes, " bikes arrived during week ", week, " for ", noOfYears, " years")
    printTime()
    return alt sammen ... 



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
            jsonFile = open("tripStats/data/Oslo/tripData/Oslo-" + str(yearNo) + "-" + monthStr  + ".json", "r")
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
        print("calcIntensity ends, leave and arrive intensity stored in " + leaveIntenseFileName + " and " + arriveIntenseFileName + " respectively")
    else:
        print("*** ERROR: calcIntensity --- illegal parameters")

def calcMoveProbab(city, mode):
    if (city == "Oslo") and (mode == "all"):
        print("move probability calculation for all trips downloaded for Oslo ...") # TODO prelim
        printTime()
        stationMap = {} # maps from stationId to station number
        stationFileName = "tripStats/" + city + "/stations.txt" 
        stationsFile = open(stationFileName, "r")
        for line in stationsFile.readlines(): # TODO, used several places, refactor
            words = line.split()
            stationMap[words[1]] = words[0]
        traffic = []
        for rowNo in range(len(stationMap)):
            row = []
            for col in range(len(stationMap)):
                row.append(0)
            traffic.append(row)
        
        # refactor code below used several places to loop through all OsloFiles, make function, retrn list of filenames    
        trips = 0
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
                endNo = int(stationMap[bikeData[i]["end_station_id"]])
                traffic[startNo][endNo] = traffic[startNo][endNo] + 1 
                trips = trips + 1  
            print(".", end='') # TODO replace with progress bar          
        move_probabilities = []
        for row in range(len(stationMap)):
            sumRow = 0
            probabilitites = []
            for col in range(len(stationMap)):
                sumRow = sumRow + traffic[row][col]
            for col in range(len(stationMap)):
                probabilitites.append(traffic[row][col]/sumRow)   
            move_probabilities.append(probabilitites)    
    else:
        print("*** ERROR: calcIntensity --- illegal parameters")

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
