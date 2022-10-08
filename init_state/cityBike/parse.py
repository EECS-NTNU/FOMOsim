# parse.py
from xml.dom.expatbuilder import parseString
import sim
import json
import os
import os.path
import requests
import geopy.distance
import datetime
from datetime import date
import numpy as np
import statistics
from statistics import fmean, stdev 
from progress.bar import Bar

import settings
from init_state.cityBike.helpers import extractCityFromURL, extractCityAndDomainFromURL
from helpers import yearWeekNoAndDay

tripDataDirectory = "init_state/cityBike/data/" # location of tripData

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

def download(url, YMpairs):    

    def loadMonth(yearNo, monthNo, alreadyLoadedFiles):
        fileName = f"{yearNo}-{monthNo:02}.json"
        if fileName in alreadyLoadedFiles:
            if yearNo == datetime.date.today().year  and monthNo == datetime.date.today().month:
                print("   Warning: We found locally stored trip-data for the current month, it will be used, BUT the file should be deleted since it is incomplete")
                return True
        else:
            if yearNo == datetime.date.today().year  and monthNo == datetime.date.today().month:
                print("   Info: will NOT load the current month, only trip-data for months that are expired") 
                return False

        if fileName not in alreadyLoadedFiles:
            # must try to load file         
            address = f"{url}{yearNo}/{monthNo:02}.json"
            data = requests.get(address)
            if data.status_code == 200: # non-existent files will have status 404
                # print(f"downloads {city} {fileName} ...") # debug
                dataOut = open(f"{directory}/{fileName}", "w")
                dataOut.write(data.text)
                dataOut.close()
                return True
            else:
                return False
        else:
            return True    

    city = extractCityFromURL(url)
    directory = f"{tripDataDirectory}{city}"
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    file_list = os.listdir(directory)

    progress = Bar("CityBike 1a/5: Download datafiles   ", max = len(YMpairs))
    notFoundYMpairs = []
    for p in YMpairs:
        if not loadMonth(p[0], p[1], file_list):
            notFoundYMpairs.append(p)
        progress.next()
    if len(notFoundYMpairs) > 0:
        print("\n   Warning: Could not load tripdata from " + url + " for these year/month pairs:", end="") 
        for p in notFoundYMpairs:
            print(" " + str(p[0]) + "/" + str(p[1]), end="")
    progress.finish()

    # check that stationinfo-file has been downloaded once, if not try to do so 
    if not os.path.isfile(f"{directory}/stationinfo.text"): 
        gbfsStart = "https://gbfs.urbansharing.com/"
        gbfsTailInfo = "/station_information.json"
        address = gbfsStart + extractCityAndDomainFromURL(url) + gbfsTailInfo
        stationInfo =  requests.get(address)
        if stationInfo.status_code != 200: # 200 is OK, non-existent files will have status 404
            raise Exception("*** Error: could not read station info from: " + address)
        stationInfoFile = open(f"{directory}/stationinfo.text", "w")
        stationInfoFile.write(stationInfo.text)
        stationInfoFile.close()
        print("   Info: station information has been read from urbansharing.com")
  
    if not os.path.isfile(f"{directory}/stationstatus.text"):
        gbfsTailStatus = "/station_status.json"
        address = gbfsStart + extractCityAndDomainFromURL(url) + gbfsTailStatus  
        stationStatus =  requests.get(address)
        if stationStatus.status_code != 200: # 200 is OK, non-existent files will have status 404
            raise Exception("*** Error: could not read station status from: " + address)
        stationStatusFile = open(f"{directory}/stationstatus.text", "w")
        stationStatusFile.write(stationStatus.text)
        stationStatusFile.close()
        print("   Info: station status has been read from urbansharing.com")
    else:
        print("   Info: stations status was found locally on your computer from an earlier run")

def log_to_norm(mu_x, stdev_x):
    # variance calculated from stdev
    var_x = stdev_x * stdev_x

    # find mean and variance for the underlying normal distribution
    mu = np.log(mu_x*mu_x / np.sqrt(var_x + mu_x*mu_x))
    var = np.log(1 + var_x/(mu_x*mu_x))

    # stdev calculated from variance
    stdev = np.sqrt(var)

    return (mu, stdev)




def get_initial_state(url="https://data.urbansharing.com/oslobysykkel.no/trips/v1/", 
    week=30, fromInclude=[2018, 5], toInclude=[2022,8], trafficMultiplier=1.0, number_of_vehicles=1,  random_seed=1):

    """ Processes selected  trips downloaded for the city, calculates average trip duration for every pair of stations, including
        back-to-start trips. For pairs of stations without any registered trips an average duration is estimated via
        the trip distance and a global average BIKE_SPEED value from settings.py. This gives the travel_time matrix.
        Travel time for the vehicle is based on distance. All selected tripdata is read and used to calculate arrive and leave intensities 
        for every station and move probabilities for every pair of stations. These structures are indexed by station, week and hour.
    """
    class StationLocation: 
        def __init__(self, stationId, longitude, latitude):
            self.stationId = stationId 
            self.longitude = longitude
            self.latitude = latitude

    def weekMonths(weekNo): # produce a list of months that can be in a given week no. Note that isocalendar 
                            # does not handle week no = 53 <<== TODO
        if weekNo == 53:
            months = [1,12]
        else:
            months = [] 
            for year in range (2018, datetime.date.today().year + 1):
                monday = date.fromisocalendar(year, weekNo, 1)
                sunday = date.fromisocalendar(year, weekNo, 7)
                if monday.month not in months:
                    months.append(monday.month)
                if sunday.month not in months:
                    months.append(sunday.month)
        return months

    city = extractCityFromURL(url)
    YMpairs = generateYMpairs(fromInclude, toInclude)
    download(url, YMpairs) 
    years = [] 

    tripDataPath = tripDataDirectory + city

    stationInfoFile = open(f"{tripDataPath}/stationinfo.text", "r")
    allData = json.loads(stationInfoFile.read())
    stationInfoData = allData["data"]

    stationStatusFile = open(f"{tripDataPath}/stationstatus.text", "r")
    allData = json.loads(stationStatusFile.read())
    stationStatusData = allData["data"]

    stationLocations = {} # dict with key stationId
    bikeStartStatus = {}
    stationCapacities = {}
    stationNames = {}
    trafficAtStation = {}

    # Read info about active stations
    for i in range(len(stationInfoData["stations"])): # loop thru all active stations
        id = stationInfoData["stations"][i]["station_id"]
        long = stationInfoData["stations"][i]["lon"]
        lat = stationInfoData["stations"][i]["lat"]
        stationLocations[id] = StationLocation(id, long, lat)  
        stationCapacities[id] = stationInfoData["stations"][i]["capacity"]
        stationNames[id] = stationInfoData["stations"][i]["name"]

        if stationStatusData["stations"][i]["station_id"]  == id: # check that it is same station
            bikeStartStatus[id] = stationStatusData["stations"][i]["num_bikes_available"]
        else:
            raise Exception("Error: stationInfoData and stationStatusData differs")
    
    ###################################################################################################
    # Find stations with traffic for given week, loop thru all months in range given by fromInclude and toInclude, now in YMpairs
    fileList = os.listdir(tripDataPath)
    progress = Bar("CityBike 1b/5: Read data from files, find stations with traffic for given week", max = len(fileList))
    trafficAtStation = {} # indexed by id, stores stations with at least one arrival or departure 
    for file in fileList:
        if file.endswith(".json"):
            y = int(file[0:4])
            m = int(file[5:7])

            if [y, m] in YMpairs: 
                jsonFile = open(os.path.join(tripDataPath, file), "r")
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
                print("   y:", y, " m:", f'{m:02d}', end="") 
        progress.next()
    progress.finish()

    stationMap = {}
    stationNo = 0
    withOutTraffic = []
    for stationId in stationNames:
        if not stationId in trafficAtStation:
            withOutTraffic.append(stationNames[stationId])
        else:
            stationMap[stationId] = stationNo
            stationNo += 1
    if len(withOutTraffic) > 0:
        print("   Info: These stations without traffic in week ", str(week), " are neglected.")
        for name in withOutTraffic:
            print(name) 

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
                arriveCount[station][day].append(0)
                leaveCount[station][day].append(0)
                stationList = []
                for i in range(len(stationMap)):
                    stationList.append(0)
                moveCount[station][day].append(stationList)

    # process all stored trips for given city, store durations for the given week number
    fileList = os.listdir(tripDataPath)
    
    progress = Bar("CityBike 2/5: Read data from files ", max = len(fileList))
    for file in fileList:
        if file.endswith(".json"):
            if int(file[5:7]) in weekMonths(week):
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
                        year, weekNo, weekDay = yearWeekNoAndDay(bikeData[i]["ended_at"][0:10])
                        if year not in years:
                            years.append(year)
                        hour = int(bikeData[i]["ended_at"][11:13])
                        if weekNo == week:
                            arriveCount[endStationNo][weekDay][hour] += 1
                        year, weekNo, weekDay = yearWeekNoAndDay(bikeData[i]["started_at"][0:10])
                        if year not in years:
                            years.append(year)
                        hour = int(bikeData[i]["started_at"][11:13])
                        if weekNo == week:
                            leaveCount[startStationNo][weekDay][hour] += 1
                            moveCount[startStationNo][weekDay][hour][endStationNo] += 1    
                            durations[startStationNo][endStationNo].append(bikeData[i]["duration"]/60) # duration in minutes
        progress.next()
    progress.finish()

    noOfYears = len(years)
    if noOfYears == 0:
        raise Exception("*** Sorry, no trip data found for given city and week")

    # Calculate average durations, durations in minutes
    progress = Bar("CityBike 3/5: Calculate durations  ", max = len(stationMap))
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

    progress = Bar("CityBike 4/5: Calculate traveltime ", max = len(stationMap))
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
    progress = Bar("CityBike 5/5: Calculate intensities", max = len(stationMap))
    noOfYears = len(years)
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
                arrive_intensities[station][day].append(trafficMultiplier * arriveCount[station][day][hour]/noOfYears)
                leave_intensities[station][day].append(trafficMultiplier * leaveCount[station][day][hour]/noOfYears)
                move_probabilities[station][day].append([])
                for endStation in range(len(stationMap)):
                    movedBikes = moveCount[station][day][hour][endStation]
                    movedBikesTotal = leaveCount[station][day][hour]
                    if movedBikesTotal > 0:
#                    if False: # TEST code to enforce equal-prob
                        move_probabilities[station][day][hour].append(movedBikes/movedBikesTotal)
                    else:
                        equalProb = 1.0/len(stationMap)    
                        move_probabilities[station][day][hour].append(equalProb)
        progress.next()
    progress.finish()

    # Create stations
    # capacities must be converted from map to list, to avoid changing code in state.py
    capacitiesList = []
    for stationId in stationMap:
        capacitiesList.append(stationCapacities[stationId])
    
    bikeStartStatusList = []
    totalBikes = 0
    for stationId in stationMap:
        bikesThere = int(bikeStartStatus[stationId])
        bikeStartStatusList.append(bikesThere)
        totalBikes += bikesThere
    if totalBikes == 0:
        print("   Info: No bikes found in stations status file, assume it is set by wrapper.py") 

    stations = sim.State.create_stations(num_stations=len(capacitiesList), capacities=capacitiesList)
    sim.State.create_bikes_in_stations(stations, "Bike", bikeStartStatusList)
    sim.State.set_customer_behaviour(stations, leave_intensities, arrive_intensities, move_probabilities)
    # Create State object and return

    state = sim.State.get_initial_state(stations=stations,
                                        number_of_vehicles=number_of_vehicles,
                                        random_seed=random_seed,
                                        traveltime_matrix=avgDuration,
                                        traveltime_matrix_stddev=durationStdDev,
                                        traveltime_vehicle_matrix=ttVehicleMatrix,
                                        # traveltime_vehicle_matrix_stddev =ttVehicleMatrixStdDev
                                        traveltime_vehicle_matrix_stddev = None
                                        ) 
    return state
