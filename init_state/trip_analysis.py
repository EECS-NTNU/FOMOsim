from xml.dom.expatbuilder import parseString
import sim
import json
import gzip
import os
import os.path
import requests
import geopy.distance
from datetime import date, datetime
import numpy as np
import statistics
from statistics import fmean, stdev 
from progress.bar import Bar

import settings
from helpers import yearWeekNoAndDay

tripDataDirectory = "init_state/data/"

def generateYMpairs(fromInclude, toInclude):
    yearMonthPairs = []
    y = fromInclude[0]
    m = fromInclude[1]
    # TODO (nice), robustness, check that there is at least one month
    while True:                 
        yearMonthPairs.append([y, m])
        if m == 12:
            m = 1
            y += 1
        else:
            m += 1
        if (y == toInclude[0]) and (m > toInclude[1]):
            return yearMonthPairs


def log_to_norm(mu_x, stdev_x):
    # variance calculated from stdev
    var_x = stdev_x * stdev_x

    # find mean and variance for the underlying normal distribution
    mu = np.log(mu_x*mu_x / np.sqrt(var_x + mu_x*mu_x))
    var = np.log(1 + var_x/(mu_x*mu_x))

    # stdev calculated from variance
    stdev = np.sqrt(var)

    return (mu, stdev)


def downloadStationInfo(gbfsStart, tripDataPath):
    # check that stationinfo-file has been downloaded once, if not try to do so 
    if not os.path.isfile(f"{tripDataPath}/stationinfo.text.gz"): 
        gbfsTailInfo = "/station_information.json"
        address = gbfsStart + gbfsTailInfo
        stationInfo =  requests.get(address)
        if stationInfo.status_code != 200: # 200 is OK, non-existent files will have status 404
            raise Exception("*** Error: could not read station info from: " + address)
        stationInfoFile = gzip.open(f"{tripDataPath}/stationinfo.text.gz", "wb")
        stationInfoFile.write(stationInfo.text.encode())
        stationInfoFile.close()
        # print("   Info: station information has been read from " + gbfsStart)
  
    if not os.path.isfile(f"{tripDataPath}/stationstatus.text.gz"):
        gbfsTailStatus = "/station_status.json"
        address = gbfsStart + gbfsTailStatus  
        stationStatus =  requests.get(address)
        if stationStatus.status_code != 200: # 200 is OK, non-existent files will have status 404
            raise Exception("*** Error: could not read station status from: " + address)
        stationStatusFile = gzip.open(f"{tripDataPath}/stationstatus.text.gz", "wb")
        stationStatusFile.write(stationStatus.text.encode())
        stationStatusFile.close()
        # print("   Info: station status has been read from " + gbfsStart)
    # else:
    #     print("   Info: stations status was found locally on your computer from an earlier run")

def weekMonths(weekNo): # produce a list of months that can be in a given week no. Note that isocalendar 
                        # does not handle week no = 53 <<== TODO
    if weekNo == 53:
        months = [1,12]
    else:
        months = [] 
        for year in range (2018, date.today().year + 1):
            monday = date.fromisocalendar(year, weekNo, 1)
            sunday = date.fromisocalendar(year, weekNo, 7)
            if monday.month not in months:
                months.append(monday.month)
            if sunday.month not in months:
                months.append(sunday.month)
    return months

def tripAnalysis(tripDataPath, YMpairs, week, trafficMultiplier=1.0):

    class StationLocation: 
        def __init__(self, stationId, longitude, latitude):
            self.stationId = stationId 
            self.longitude = longitude
            self.latitude = latitude

    stationInfoFile = gzip.open(f"{tripDataPath}/stationinfo.text.gz", "rb")
    allData = json.loads(stationInfoFile.read())
    stationInfoData = allData["data"]

    stationStatusFile = gzip.open(f"{tripDataPath}/stationstatus.text.gz", "rb")
    allData = json.loads(stationStatusFile.read())
    stationStatusData = allData["data"]

    stationLocations = {} # dict with key stationId
    stationNumBikes = {}
    stationCapacities = {}
    stationNames = {}
    trafficAtStation = {}

    # Get num_bikes_available from stationStatusData
    for i in range(len(stationStatusData["stations"])):
        id = stationStatusData["stations"][i]["station_id"]
        numBikes = stationStatusData["stations"][i]["num_bikes_available"]
        stationNumBikes[id] = numBikes

    # Read info about active stations
    for i in range(len(stationInfoData["stations"])): # loop thru all active stations
        id = stationInfoData["stations"][i]["station_id"]
        long = stationInfoData["stations"][i]["lon"]
        lat = stationInfoData["stations"][i]["lat"]
        stationLocations[id] = StationLocation(id, long, lat)  
        stationCapacities[id] = stationInfoData["stations"][i]["capacity"]
        stationNames[id] = stationInfoData["stations"][i]["name"]

    assert(len(stationNumBikes) == len(stationLocations))
    

    ###################################################################################################
    # Find stations with traffic for given week, loop thru all months in range given by fromInclude and toInclude, now in YMpairs
    fileList = os.listdir(tripDataPath)
    progress = Bar("Find stations with traffic    ", max = len(fileList))
    trafficAtStation = {} # indexed by id, stores stations with at least one arrival or departure 
    for file in fileList:
        if file.endswith(".json.gz"):
            y = int(file[0:4])
            m = int(file[5:7])

            if [y, m] in YMpairs: 
                if int(file[5:7]) in weekMonths(week):
                    jsonFile = gzip.open(os.path.join(tripDataPath, file), "rb")
                    bikeData = json.loads(jsonFile.read())
                    for i in range(len(bikeData)):
                        dummy1, weekNo, dummy2 = yearWeekNoAndDay(bikeData[i]["started_at"][0:10])
                        if weekNo == week:
                            startId = bikeData[i]["start_station_id"]
                            if startId in stationNames:
                                trafficAtStation[startId] = True 
                        dummy1, weekNo, dummy2 = yearWeekNoAndDay(bikeData[i]["ended_at"][0:10])
                        if weekNo == week:
                            endId = bikeData[i]["end_station_id"]
                            if endId in stationNames:
                                trafficAtStation[endId] = True                     
                    #print("   y:", y, " m:", f'{m:02d}', end="") 
        progress.next()
    progress.finish()

    stationMap = {}
    stationNo = 0
    # withOutTraffic = []
    for stationId in stationNames:
        if stationId in trafficAtStation:
            stationMap[stationId] = stationNo
            stationNo += 1
        # else:
        #     withOutTraffic.append(stationNames[stationId])

    # if len(withOutTraffic) > 0:
    #     print("   Info: These stations without traffic in week ", str(week), " are neglected.")
    #     for name in withOutTraffic:
    #         print(name) 

    arriveCount = []
    leaveCount = []
    moveCount = []
    durations = []
    for station in range(len(stationMap)):
        arriveCount.append([]) # arriving at this station
        leaveCount.append([]) # departing from this station
        moveCount.append([]) # moving from this station to an end-station
        durations.append([]) # traveltime from this station to an end-station
        for s in range(len(stationMap)):
            durations[station].append([])
        for day in range(7): # weekdays here are 0..6
            arriveCount[station].append([])
            leaveCount[station].append([])
            moveCount[station].append([])
            for hour in range(24):
                arriveCount[station][day].append({})
                leaveCount[station][day].append({})
                stationList = []
                for i in range(len(stationMap)):
                    stationList.append(0)
                moveCount[station][day].append(stationList)

    # process all stored trips for given city, store durations for the given week number
    fileList = os.listdir(tripDataPath)
    
    years = [] 

    progress = Bar("Read data from files          ", max = len(fileList))
    for file in fileList:
        if file.endswith(".json.gz"):
            if int(file[5:7]) in weekMonths(week):
                jsonFile = gzip.open(os.path.join(tripDataPath, file), "rb")
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
                        year, weekNo, weekDay = yearWeekNoAndDay(bikeData[i]["ended_at"][0:10])
                        hour = int(bikeData[i]["ended_at"][11:13])
                        if year not in years:
                            years.append(year)
                        if year not in arriveCount[endStationNo][weekDay][hour]:
                            arriveCount[endStationNo][weekDay][hour][year] = 0
                        if weekNo == week:
                            arriveCount[endStationNo][weekDay][hour][year] += 1

                        year, weekNo, weekDay = yearWeekNoAndDay(bikeData[i]["started_at"][0:10])
                        hour = int(bikeData[i]["started_at"][11:13])
                        if year not in years:
                            years.append(year)
                        if year not in leaveCount[startStationNo][weekDay][hour]:
                            leaveCount[startStationNo][weekDay][hour][year] = 0
                        if weekNo == week:
                            if "duration" in bikeData[i]:
                                duration = bikeData[i]["duration"]/60
                            else:
                                started_at = datetime.strptime(bikeData[i]["started_at"][0:19], "%Y-%m-%d %H:%M:%S")
                                ended_at = datetime.strptime(bikeData[i]["ended_at"][0:19], "%Y-%m-%d %H:%M:%S")
                                delta = ended_at - started_at
                                duration = delta.days * 24 * 60 + delta.seconds / 60

                            leaveCount[startStationNo][weekDay][hour][year] += 1
                            moveCount[startStationNo][weekDay][hour][endStationNo] += 1    
                            durations[startStationNo][endStationNo].append(duration)
        progress.next()
    progress.finish()

    noOfYears = len(years)
    if noOfYears == 0:
        raise Exception("*** Sorry, no trip data found for given city and week")

    # Calculate average durations, durations in minutes
    progress = Bar("Calculate durations           ", max = len(stationMap))
    avgDuration = []
    durationStdDev = []

    for s in stationMap:
        start = stationMap[s]
        avgDuration.append([])
        durationStdDev.append([])
        for e in stationMap:
            end = stationMap[e]
            avgDuration[start].append([])
            durationStdDev[start].append([])

            # calculate mean
            if len(durations[start][end]) > 0:
                mean_x = fmean(durations[start][end])
            else:
                distance = geopy.distance.distance((stationLocations[s].latitude, stationLocations[s].longitude), 
                   (stationLocations[e].latitude, stationLocations[e].longitude)).km
                mean_x = (distance/settings.BIKE_SPEED)*60
            if mean_x == 0:
                if start == end:
                    mean_x = 7.7777 # TODO, consider to eventually calculate all such positive averageDuarations, and use here
                else:
                    raise Exception("*** Error, averageDuration == 0 should not happen") 

            # calculate stdev
            if len(durations[start][end]) > 1:
                stdev_x = stdev(durations[start][end], mean_x)            
            else:
                stdev_x = 0
            # convert mean and stdev from lognormal distribution to normal distribution
            mean, st = log_to_norm(mean_x, stdev_x)

            avgDuration[start][end] = mean
            durationStdDev[start][end] = st
        progress.next()
    progress.finish()

    progress = Bar("Calculate traveltime          ", max = len(stationMap))
    ttVehicleMatrix = []
    #ttVehicleMatrixStdDev = []
    for s in stationMap:
        start = stationMap[s]
        ttVehicleMatrix.append([])
        #ttVehicleMatrixStdDev.append([])
        for e in stationMap:
            end = stationMap[e]
            distance = geopy.distance.distance((stationLocations[s].latitude, stationLocations[s].longitude), 
               (stationLocations[e].latitude, stationLocations[e].longitude)).km
            ttVehicleMatrix[start].append((distance/settings.VEHICLE_SPEED)*60)
            #ttVehicleMatrixStdDev[start].append(0) # code prepared to use some variation here
        progress.next()
    progress.finish()
    
    # Calculate arrive and leave-intensities and move_probabilities
    progress = Bar("Calculate intensities         ", max = len(stationMap))
    noOfYears = len(years)
    arrive_intensities = []  
    arrive_intensities_stdev = []  
    leave_intensities = []
    leave_intensities_stdev = []
    move_probabilities= []     
    for station in range(len(stationMap)):
        arrive_intensities.append([])
        arrive_intensities_stdev.append([])
        leave_intensities.append([])
        leave_intensities_stdev.append([])
        move_probabilities.append([])
        for day in range(7):
            arrive_intensities[station].append([])
            arrive_intensities_stdev[station].append([])
            leave_intensities[station].append([])
            leave_intensities_stdev[station].append([])
            move_probabilities[station].append([])
            for hour in range(24):
                arrCount = list(arriveCount[station][day][hour].values())
                for i in range(len(arrCount), noOfYears):
                    arrCount.append(0)

                arrive_intensities[station][day].append(trafficMultiplier * np.mean(arrCount))
                arrive_intensities_stdev[station][day].append(trafficMultiplier * np.std(arrCount))

                lCount = list(leaveCount[station][day][hour].values())
                for i in range(len(lCount), noOfYears):
                    lCount.append(0)

                leave_intensities[station][day].append(trafficMultiplier * np.mean(lCount))
                leave_intensities_stdev[station][day].append(trafficMultiplier * np.std(lCount))
                    
                move_probabilities[station][day].append([])
                for endStation in range(len(stationMap)):
                    movedBikes = moveCount[station][day][hour][endStation]
                    movedBikesTotal = np.sum(lCount)
                    if movedBikesTotal > 0:
#                    if False: # TEST code to enforce equal-prob
                        move_probabilities[station][day][hour].append(movedBikes/movedBikesTotal)
                    else:
                        equalProb = 1.0/len(stationMap)    
                        move_probabilities[station][day][hour].append(equalProb)
        progress.next()
    progress.finish()

    # Create stations

    stations = []
    for i, station_id in enumerate(stationMap):
        station = {}
        station["id"] = i
        station["location"] = (stationLocations[station_id].latitude, stationLocations[station_id].longitude)
        station["is_depot"] = False
        station["capacity"] = stationCapacities[station_id]
        station["num_bikes"] = stationNumBikes[station_id]
        station["leave_intensities"] = leave_intensities[i]
        station["leave_intensities_stdev"] = leave_intensities_stdev[i]
        station["arrive_intensities"] = arrive_intensities[i]
        station["arrive_intensities_stdev"] = arrive_intensities_stdev[i]
        station["move_probabilities"] = move_probabilities[i]
        stations.append(station)

    # Create state

    statedata = {
        "stations" : stations,
        "bike_class" : "Bike",
        "traveltime" : avgDuration,
        "traveltime_stdev" : durationStdDev,
        "traveltime_vehicle" : ttVehicleMatrix,
        "traveltime_vehicle_stdev" : None,
    }

    return statedata