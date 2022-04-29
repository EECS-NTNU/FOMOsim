# parse.py
import sim
import json
import os.path
import geopy.distance
import settings

from init_state.cityBike.helpers import yearWeekNoAndDay 

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
    fileName = "init_state/cityBike/data/" + city + "/stations.txt"
    stationsMap = open(fileName, "w")
    count = 0
    for s in stations:
        stationId = f'{count:>5}'+ f'{s.stationId:>6}' + " " + s.longitude + " " + s.latitude + " " + s.stationName + "\n"
        stationsMap.write(stationId)
        count = count + 1
    stationsMap.close()

def calcDistances(city):
    stationNo = 0 # station numbers found, counts 0,1,2,...
    stations = []
    stationMap = {} # maps from stationId to station number 

    tripDataPath = "init_state/cityBike/data/" + city + "/tripData"
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
        # print(".", end='') # TODO nice, replace with progress bar
    # print("A total of ", len(set(stations)), " stations used, reported on stations.txt")
    reportStations(stations, city)
    dist_matrix_km = [] # km in kilometers
    dm_file = open("init_state/cityBike/data/" + city + "/Distances.txt", "w")
    for rowNo in range(len(stationMap)):
        col = 0 
        row = []
        for col in range(len(stationMap)):
            dist = geopy.distance.distance(
                (stations[rowNo].latitude, stations[rowNo].longitude), 
                (stations[col].latitude, stations[col].longitude)).km
            row.append(dist)
            dm_file.write(str(dist))
            dm_file.write(" ")
            col = col + 1
        dist_matrix_km.append(row)
        dm_file.write("\n")
        rowNo = rowNo + 1    
    # print("Distances calculated, stored in Distances.txt and returned thru call")
    return dist_matrix_km

def readStationMap(city):
    stationMap = {} # maps from stationId to station number 
    stationsFile = open("init_state/cityBike/data/" + city + "/stations.txt", "r")
    for line in stationsFile.readlines():
        words = line.split()
        stationMap[words[1]] = int(words[0])
    return stationMap

def readBikeStartStatus(city):
    if city == "Oslo" or city == "Utopia":
#        bikeStatusFile = open("init_state.cityBike/data/Oslo/stationStatus-23-Mar-1513.json", "r")
        bikeStatusFile = open("init_state/cityBike/data/Oslo/stationStatus-26-Apr-1140.json", "r")
        allStatusData = json.loads(bikeStatusFile.read())
        stationData = allStatusData["data"]
        stationMap = readStationMap(city)
        bikeStartStatus = []
        for stat in range(len(stationMap)):
            bikeStartStatus.append(0)
        for i in range(len(stationData["stations"])):
            station = int(stationData["stations"][i]["station_id"])
            if str(station) in stationMap:
                stationNo = stationMap[str(station)]
                noOfBikes = stationData["stations"][i]["num_bikes_available"]
                bikeStartStatus[stationNo] = noOfBikes
    return bikeStartStatus

def get_initial_state(city, week):
    if city == "Oslo" and ( (week < 1) or (week > 53)):
        print("*** Error: week no must be in range 1..53")
    elif city == "Utopia" and (week != 48):
        print("*** Error: week must be 48")
    elif not (city == "Oslo" or city == "Utopia"):
        print("*** Error: given city not implemented ", city)    

    print("get_initial_state starts analyzing traffic for city: " + city + " for week " + str(week) 
        + ", setting up datastructures ... ", end='') 
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

    # process all stored trips for given city, count trips and store durations for the given week number
    trips = 0
    arrivingBikes = 0
    leavingBikes = 0
    tripDataPath = "init_state/cityBike/data/" + city + "/tripData"
    fileList = os.listdir(tripDataPath)
    for file in fileList:
        jsonFile = open(os.path.join(tripDataPath, file), "r")
        bikeData = json.loads(jsonFile.read())
        for i in range(len(bikeData)):
            year, weekNo, weekDay = yearWeekNoAndDay(bikeData[i]["ended_at"][0:10])
            # print(" ", weekNo, end= "") # debug
            years.append(year)
            hour = int(bikeData[i]["ended_at"][11:13])
            endStationNo = 0 #
            if weekNo == week:
                endStationNo = stationMap[bikeData[i]["end_station_id"]]
                arriveCount[endStationNo][weekDay][hour] += 1
                arrivingBikes += 1
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
    
    # Calculate average durations
    # print("calculate avg trip durations ", end='')
    avgDuration = []
    for start in range(len(stationMap)):
        avgDuration.append([])
        for end in range(len(stationMap)):
            avgDuration[start].append([])
            sumDuration = 0
            for trip in range(len(durations[start][end])):
                sumDuration += durations[start][end][trip]
            avgDuration[start][end] = sumDuration/len(durations)

    # Calculate distance
    # print(" calculate all possible distances ", end='')
    distances = calcDistances(city)

    # Calculate speed matrix
    # print(" calculate speed matrix ", end='')
    speed_matrix = []
    for start in range(len(stationMap)):
        speed_matrix.append([])
        for end in range(len(stationMap)):
            averageDuration = avgDuration[start][end]
            if averageDuration > 0 and start != end:
                speed_matrix[start].append(distances[start][end]/(averageDuration/3600))
            else:
                speed_matrix[start].append(settings.SCOOTER_SPEED)
 
    # Calculate arrive and leave-intensities and move_probabilities
    # print(" calculate intensities ", end='')
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
                        pass
                        # equalProb = 1.0/len(stationMap)    
                        # move_probabilities[station][day][hour].append(equalProb)
                        move_probabilities[station][day][hour].append(0.0)
                        # fiks til 1 delt på antall stasjoner issue 8 ... move_probabilities[station][day][hour].append(0.0) # TODO check this, should set all to zero if no traffic !? 

    loggText = ["trips:", str(trips), "left:", str(leavingBikes), "arrived:", str(arrivingBikes), "week:", str(week), "years:", str(noOfYears), "city:", city]
    bikeStartStatus = readBikeStartStatus(city)
    print(" ") # newline in terminal
    totalBikes = 0
    for i in range(len(bikeStartStatus)):
        totalBikes += bikeStartStatus[i]
    print("Total number of bikes: ", totalBikes) 
    return sim.State.get_initial_state(
        bike_class = "Bike", # TODO helpers.loggLoction will crash if Bike is used here
        distance_matrix = distances,
        speed_matrix = speed_matrix, 
        main_depot = None,
        secondary_depots = [],
        number_of_scooters = bikeStartStatus,
        number_of_vans = 2,
        random_seed = 1,
        arrive_intensities = arrive_intensities,
        leave_intensities = leave_intensities,
        move_probabilities = move_probabilities,
    ), loggText
