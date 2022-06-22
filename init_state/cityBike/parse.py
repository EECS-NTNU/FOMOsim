# parse.py

# import operator
# from os import stat
# from sre_parse import Verbose
import sim
import json
import os.path
import geopy.distance

import settings
from GUI import loggFile
from init_state.cityBike.helpers import yearWeekNoAndDay, write, dateAndTimeStr 

class Station:
    def __init__(self, stationId, longitude, latitude, stationName):
        self.stationId = stationId
        self.longitude = longitude
        self.latitude = latitude
        self.stationName = stationName
    moved = False
    renamed = False    
    def __lt__(self, other):
        return self.stationId < other.stationId
    def toString(self):
        return f'{self.stationId:>6}' + " " + self.longitude + " " + self.latitude + " " + self.stationName

# class BikeTrip:
#     def __init__(self, startTime, endTime, duration, start, end):
#         self.start = startTime
#         self.end = endTime
#         self.duration = duration
#         self.start = start
#         self.end = end

#### LEST OVER HIT    ***********************************************************************************'

def sortStations(stationsDict):
    stationsList = []
    for s in sorted(stationsDict, key = stationsDict.get):
        stationsList.append(stationsDict[s])
    return stationsList

def washAndReportStations(stationsList, city):
    # will produce two files, stations.txt that is "washed" by removing stations with capacity = 0 as reported in snapshot from April 26. 2022. This file
    # is used in the simulations. The other file stationsAll.txt with all stations found in the tripData is included to ease debugging. 
    # If Verbose = True, the washing is reported in terminal
    Verbose = False

    fileName = "init_state/cityBike/data/" + city + "/stationsAll.txt"
    stationsDescr = open(fileName, "w")
    count = 0
    stationIsActive = [] # used for washing, see comment above
    for s in range(len(stationsList)):
        stationIdText = f'{count:>5}'+ stationsList[s].toString() + "\n"
        stationsDescr.write(stationIdText)
        count = count + 1
        stationIsActive.append(False) # initialization, used further down
    stationsDescr.close()
    
    stationStatusFile = open("init_state/cityBike/data/Oslo/stationStatus-26-Apr-1140.json", "r") # TODO, is possible to update more dynamically, it is
                                                                                                  # read from https://gbfs.urbansharing.com/oslobysykkel.no/station_information.json
    allStatusData = json.loads(stationStatusFile.read())
    stationData = allStatusData["data"]

    id2no = readStationMap(city) # map is needed to find active stations, readStationMap reads from stationsAll.txt
    for i in range(len(stationData["stations"])): # loop thru all active stations
        stationId = stationData["stations"][i]["station_id"]
        if stationId in id2no:
            stationNo = id2no[stationId] # stationNo is number in list of stations
            stationIsActive[stationNo] = True

    fileName = "init_state/cityBike/data/" + city + "/stations.txt" # produce the WASHED list
    stationsDescr = open(fileName, "w")
    count = 0
    activeStations = []
    for s in range(len(stationsList)):
        stationIdText = f'{count:>5}'+ stationsList[s].toString() + "\n"
        if stationIsActive[s]:
            stationsDescr.write(stationIdText)
            activeStations.append(s)
            count = count + 1
        elif Verbose:
            print("Station removed by washing: ", stationIdText, end="")
    stationsDescr.close()
    return activeStations # list of activeStations, as indices into list of all

def calcDistances(city):
    """ All stations data available are processed. Stations that are moved while keeping their identy can be reported using
        the flag settings.REPORT_CHANGES. The same goes for stations that have a name change.
    """
    stationNo = 0 # station numbers found, counts 0,1,2,...
    stationMap = {} # maps from stationId to station number, ALL stations found in the folder <city>/data/tripData are included 
    stationsData = {} # ALL stations found in tripData are included 
    tripDataPath = "init_state/cityBike/data/" + city + "/tripData"
    fileList = os.listdir(tripDataPath)
    fileList.sort() # needed to get linux and windows behave similarly
    if len(fileList) == 0:
        print("*** Error: you must download city data" )
    else:    
        for file in fileList:
            # NOTE we read all stored trip-data to find all "possible" stations. If there are stations that
            # are taken out of operation, we should remove them TODO How to remove? manually? describe?
            jsonFile = open(os.path.join(tripDataPath, file), "r")
            bikeData = json.loads(jsonFile.read())

            for i in range(len(bikeData)):
                startId = int(bikeData[i]["start_station_id"])
                startLong = str(bikeData[i]["start_station_longitude"])
                startLat = str(bikeData[i]["start_station_latitude"])
                startName = bikeData[i]["start_station_name"]
                if not startId in stationMap: # first entry for this station
                    stationMap[startId] = stationNo
                    stationNo = stationNo + 1
                    stationsData[startId] = Station(startId, startLong, startLat, startName)
                else: # we already have a Station-object for startId, will check if data are changed and eventually report such changes
                    if stationsData[startId].longitude != startLong or stationsData[startId].latitude != startLat:
                        moveDist = geopy.distance.distance((stationsData[startId].latitude, stationsData[startId].longitude), 
                            (startLat, startLong)).km
                        stationsData[startId].longitude = startLong
                        stationsData[startId].latitude = startLat
                        if settings.REPORT_CHANGES:
                            if stationsData[startId].moved == False:
                                print("* position of station ", startId, "was moved ", "%.3f" % moveDist, "km")
                                stationsData[startId].moved = True
                    if stationsData[startId].stationName != startName:
                        oldName = stationsData[startId].stationName
                        stationsData[startId].stationName = startName
                        if settings.REPORT_CHANGES:
                            if stationsData[startId].renamed == False:
                                print("* name of station ", startId, "was changed from", oldName, " to ", startName )
                                stationsData[startId].renamed = True

                endId = int(bikeData[i]["end_station_id"])
                endLong = str(bikeData[i]["end_station_longitude"])
                endLat = str(bikeData[i]["end_station_latitude"])
                endName = bikeData[i]["end_station_name"]
                if not endId in stationMap:
                    stationMap[endId] = stationNo
                    stationNo = stationNo + 1
                    stationsData[endId] = Station(endId, endLong, endLat, endName)
                else: # we already have a Station-object for endId, will check if data are changed and eventually report such changes
                    if stationsData[endId].longitude != endLong or stationsData[endId].latitude != endLat:
                        moveDist = geopy.distance.distance((stationsData[endId].latitude, stationsData[endId].longitude), 
                            (endLat, endLong)).km
                        stationsData[endId].longitude = endLong
                        stationsData[endId].latitude = endLat
                        if settings.REPORT_CHANGES:
                            if stationsData[endId].moved == False:
                                print("* position of station ", endId, "was moved ", "%.3f" % moveDist, "km")
                                stationsData[endId].moved = True
                    if stationsData[endId].stationName != endName:
                        oldName = stationsData[endId].stationName
                        stationsData[endId].stationName = endName 
                        if settings.REPORT_CHANGES:
                            if stationsData[endId].renamed == False:
                                print("* name of station ", endId, "was changed from ", oldName, "to ", endName)                    
                                stationsData[endId].renamed = True
        print("A total of ", len(set(stationsData)), " stations used, reported on stationsAll.txt")

    stationsList = sortStations(stationsData)  # this step was needed to assure equivalent behaviour under windows and linux. List contains ALL stations  
    activeStationNos = washAndReportStations(stationsList, city)

    dist_matrix_km = [] # km in kilometers
    dm_file = open("init_state/cityBike/data/" + city + "/Distances.txt", "w")
    for rowNo in range(len(activeStationNos)):
        col = 0 
        row = []
        for col in range(len(activeStationNos)):
            # dist = geopy.distance.distance((stationsList[rowNo].latitude, stationsList[rowNo].longitude), 
            #     (stationsList[col].latitude, stationsList[col].longitude)).km
            dist = geopy.distance.distance((stationsList[activeStationNos[rowNo]].latitude, stationsList[activeStationNos[rowNo]].longitude), 
                (stationsList[activeStationNos[col]].latitude, stationsList[activeStationNos[col]].longitude)).km # TODO, puh..., simplify this, ugly
            dist = round(dist, 3)    
            if dist == 0.0 and rowNo != col:
                print("*** ERROR: Distance between two stations is zero", end ="") 
                print(" --- set to 1 km by guessing", rowNo, "", col)
                dist = 1.0
            row.append(dist)
            dm_file.write(str(dist))
            dm_file.write(" ")
            col = col + 1
        dist_matrix_km.append(row)
        dm_file.write("\n")
        rowNo = rowNo + 1    
    print("Distances calculated, stored in Distances.txt and returned thru call")
    return dist_matrix_km

def readStationMap(city):
    stationMap = {} # maps from stationId to station number 
    stationsFile = open("init_state/cityBike/data/" + city + "/stationsAll.txt", "r")
    for line in stationsFile.readlines():
        words = line.split()
        stationMap[words[1]] = int(words[0])
    return stationMap

def readActiveStationMap(city):
    stationMap = {} # maps from stationId to station number 
    stationsFile = open("init_state/cityBike/data/" + city + "/stations.txt", "r")
    for line in stationsFile.readlines():
        words = line.split()
        stationMap[words[1]] = int(words[0])
    return stationMap

def readBikeStartStatus(city):
    verbose = False # local, TODO, make global setting for warnings?
    if city == "Oslo" or city == "Utopia":
#        bikeStatusFile = open("init_state.cityBike/data/Oslo/stationStatus-23-Mar-1513.json", "r")
        bikeStatusFile = open("init_state/cityBike/data/Oslo/stationStatus-26-Apr-1140.json", "r")
        allStatusData = json.loads(bikeStatusFile.read())
        stationData = allStatusData["data"]
        id2no = readActiveStationMap(city)
        bikeStartStatus = []
        for stat in range(len(id2no)): # make list of correct length, fill values below for those found
            bikeStartStatus.append(0)
        for i in range(len(stationData["stations"])):
            stationId = stationData["stations"][i]["station_id"]
            if stationId in id2no:
                stationNo = id2no[stationId]
                noOfBikes = stationData["stations"][i]["num_bikes_available"]
                bikeStartStatus[stationNo] = noOfBikes
            elif verbose:
                print("*** Warning: active station without any tripData, is neglected. StationId: ", stationId)
    else:
        print("*** Error - given city not implemented")
    return bikeStartStatus

def readCapacities(city):
    verbose = False
    dockStartStatus = []
    if city == "Oslo" or city == "Utopia":
        bikeStatusFile = open("init_state/cityBike/data/Oslo/stationStatus-26-Apr-1140.json", "r")
        allStatusData = json.loads(bikeStatusFile.read())
        stationData = allStatusData["data"]
        id2no = readActiveStationMap(city)
        dockStartStatus = []
        for stat in range(len(id2no)): # make list of correct length, fill values below for those found
            dockStartStatus.append(0)
        for i in range(len(stationData["stations"])):
            stationId = stationData["stations"][i]["station_id"]
            if stationId in id2no:
                stationNo = id2no[stationId]
                noOfDocks = stationData["stations"][i]["num_docks_available"]
                noOfBikes = stationData["stations"][i]["num_bikes_available"]
                dockStartStatus[stationNo] = noOfDocks + noOfBikes
            elif verbose:
                print("*** Warning: active station without any tripData, is neglected. StationId: ", stationId)    
    else:
        print("*** Error - readCapacities not implemented for given city")
    return dockStartStatus

def get_initial_state(city="Oslo", week=30, bike_class="Bike", number_of_vans=3, random_seed=1):
    if city == "Oslo" and ( (week < 1) or (week > 53)):
        print("*** Error: week no must be in range 1..53")
    elif city == "Utopia" and (week != 48): # Utopia is used for test purposes
        print("*** Error: week must be 48")
    elif not (city == "Oslo" or city == "Utopia"):
        print("*** Error: given city not implemented ", city)    

    # Read all stations data for city, calculate distances, and remove non-active stations.
    distances = calcDistances(city)

    print("get_initial_state starts analyzing traffic for city: " + city + " for week " + str(week) 
        + ", setting up datastructures ... ") 
    years = [] # Must count no of "year-instances" of the given week that are analyzed
    stationMap = readActiveStationMap(city)
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

    # process all stored trips for given city, count trips and store durations for the given week number, only for 
    # trips with both active start station and end station
    trips = 0 # total number from all tripdata read
    backToStartTrips = 0
    tripDataPath = "init_state/cityBike/data/" + city + "/tripData"
    fileList = os.listdir(tripDataPath)
    for file in fileList:
        jsonFile = open(os.path.join(tripDataPath, file), "r")
        bikeData = json.loads(jsonFile.read())
        for i in range(len(bikeData)):
            startStationNo = -1
            endStationNo = -1
            if bikeData[i]["start_station_id"] in stationMap:
                startStationNo = stationMap[bikeData[i]["start_station_id"]]
            if bikeData[i]["end_station_id"] in stationMap:
                endStationNo = stationMap[bikeData[i]["end_station_id"]]

            if (startStationNo >= 0) and (endStationNo >= 0):
                # both end and start of trip are active stations
                if startStationNo == endStationNo:
                    backToStartTrips += 1  
                year, weekNo, weekDay = yearWeekNoAndDay(bikeData[i]["ended_at"][0:10])
                # print(" ", weekNo, end= "") # debug
                years.append(year)
                hour = int(bikeData[i]["ended_at"][11:13])
                if weekNo == week:
                    arriveCount[endStationNo][weekDay][hour] += 1
                year, weekNo, weekDay = yearWeekNoAndDay(bikeData[i]["started_at"][0:10])
                years.append(year)
                hour = int(bikeData[i]["started_at"][11:13])
                if weekNo == week:
                    leaveCount[startStationNo][weekDay][hour] += 1
                    moveCount[startStationNo][weekDay][hour][endStationNo] += 1    
                    durations[startStationNo][endStationNo].append(bikeData[i]["duration"])

                trips = trips + 1
        print(".", end='') # TODO replace with progress bar
    
    # Calculate average durations, durations in seconds
    avgDuration = []
    for start in range(len(stationMap)):
        avgDuration.append([])
        for end in range(len(stationMap)):
            avgDuration[start].append([])
            sumDuration = 0
            noOfTrips = 0
            for trip in range(len(durations[start][end])):
                tripDuration = durations[start][end][trip]
                noOfTrips += 1
                sumDuration += tripDuration     
            if noOfTrips > 0:
                avgDuration[start][end] = sumDuration/noOfTrips
            else:
                avgDuration[start][end] = (distances[start][end]/settings.SCOOTER_SPEED)*3600

    # Calculate traveltime_matrix, travel-times in minutes
    ttMatrix = []
    for start in range(len(stationMap)):
        ttMatrix.append([])
        for end in range(len(stationMap)):
            averageDuration = avgDuration[start][end]
            if averageDuration == 0:
                if start == end:
                    averageDuration = 7.7777 # TODO, improve, calculate all such positive averageDuarations, and use here
                else:
                    print("*** Error, averageDuration == 0 should not happen") # should be set to default scooter-speed above
            ttMatrix[start].append(averageDuration/60)
         
    ttVanMatrix = []
    for start in range(len(stationMap)):
        ttVanMatrix.append([])
        for end in range(len(stationMap)):
            ttVanMatrix[start].append((distances[start][end]/settings.VEHICLE_SPEED)*60)
                       
    # Calculate arrive and leave-intensities and move_probabilities
    noOfYears = len(set(years))
    arrive_intensities = []  
    leave_intensities = []
    move_probabilities= []
    for station in range(len(stationMap)):
        if station % 20 == 0:
            print(".", end='')
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
                        equalProb = 1.0/len(stationMap)    
                        move_probabilities[station][day][hour].append(equalProb)

    bikeStartStatus = readBikeStartStatus(city)
    dockStartStatus = readCapacities(city)
    totalBikes = 0
    for i in range(len(bikeStartStatus)):
        totalBikes += bikeStartStatus[i]
    write(loggFile, ["Init-state-based-on-traffic:", "trips:", str(trips), "week:", str(week), "years:", str(noOfYears), "bikesAtStart:", str(totalBikes), "city:", city])

    return sim.State.get_initial_state(
        bike_class = bike_class, 
        traveltime_matrix = ttMatrix, 
        traveltime_van_matrix = ttVanMatrix,
        main_depot = False,
        secondary_depots = 0,
        number_of_scooters = bikeStartStatus,
        capacities = dockStartStatus,
        number_of_vans = number_of_vans,
        random_seed = random_seed,
        arrive_intensities = arrive_intensities,
        leave_intensities = leave_intensities,
        move_probabilities = move_probabilities,
    )

