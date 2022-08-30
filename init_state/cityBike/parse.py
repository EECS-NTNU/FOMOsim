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

def download(url):
    newDataFound = False

    def loadMonth(yearNo, monthNo):
        fileName = f"{yearNo}-{monthNo:02}.json"
        loaded = False
        if fileName not in file_list:     
            address = f"{url}{yearNo}/{monthNo:02}.json"
            data = requests.get(address)
            if data.status_code == 200: # non-existent files will have status 404
                # print(f"downloads {city} {fileName} ...") # debug
                dataOut = open(f"{directory}/{fileName}", "w")
                dataOut.write(data.text)
                dataOut.close()
                loaded = True
        return loaded

    city = extractCityFromURL(url)
    directory = f"{tripDataDirectory}{city}"
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    file_list = os.listdir(directory)

    # these loops are a brute-force method to avoid implementing a web-crawler
    progress = Bar("CityBike 1a/5: Download datafiles   ", max = (datetime.date.today().year - 2018) * 12 + datetime.date.today().month - 1)
    for yearNo in range(2018, datetime.date.today().year): # 2018/02 is earliest data from data.urbansharing.com
        for month in range (1, 13):
            if loadMonth(yearNo, month):
                newDataFound = True
            progress.next()
    for month in range (1, datetime.date.today().month):
        if loadMonth(datetime.date.today().year, month):
            newDataFound = True
        progress.next()
    progress.finish()

    if newDataFound:
        # print("downloads station information")
        gbfsStart = "https://gbfs.urbansharing.com/"
        gbfsTailInfo = "/station_information.json"
        address = gbfsStart + extractCityAndDomainFromURL(url) + gbfsTailInfo
        stationInfo =  requests.get(address)
        if stationInfo.status_code != 200: # 200 is OK, non-existent files will have status 404
            raise Exception("*** Error: could not read station info from: " + address)
        stationInfoFile = open(f"{directory}/stationinfo.text", "w")
        stationInfoFile.write(stationInfo.text)
        stationInfoFile.close()

        # print("downloads station status")
        gbfsTailStatus = "/station_status.json"
        address = gbfsStart + extractCityAndDomainFromURL(url) + gbfsTailStatus  
        stationStatus =  requests.get(address)
        if stationStatus.status_code != 200: # 200 is OK, non-existent files will have status 404
            raise Exception("*** Error: could not read station status from: " + address)
        stationStatusFile = open(f"{directory}/stationstatus.text", "w")
        stationStatusFile.write(stationStatus.text)
        stationStatusFile.close()

    return newDataFound            

def log_to_norm(mu_x, stdev_x):
    # variance calculated from stdev
    var_x = stdev_x * stdev_x

    # find mean and variance for the underlying normal distribution
    mu = np.log(mu_x*mu_x / np.sqrt(var_x + mu_x*mu_x))
    var = np.log(1 + var_x/(mu_x*mu_x))

    # stdev calculated from variance
    stdev = np.sqrt(var)

    return (mu, stdev)


def get_initial_state(url="https://data.urbansharing.com/oslobysykkel.no/trips/v1/", week=30, number_of_vehicles=1, random_seed=1):
    """ Processes all stored trips downloaded for the city, calculates average trip duration for every pair of stations, including
        back-to-start trips. For pairs of stations without any registered trips an average duration is estimated via
        the trip distance and a global average BIKE_SPEED value from settings.py. This gives the travel_time matrix.
        Travel time for the vehicle is based on distance. All tripdata is read and used to calculate arrive and leave intensities 
        for every station and move probabilities for every pair of stations. These structures are indexed by station, week and hour.
        Station capacities and number of bikes in use is based on real_time data read at execution time, NOTE this will remove 
        reproducibility of simulations (but it can be re-gained by caching)
    """
    class StationLocation: 
        def __init__(self, stationId, longitude, latitude):
            self.stationId = stationId 
            self.longitude = longitude
            self.latitude = latitude

    def weekMonths(weekNo): # produce a list of months that can be in a given week no. Note that isocalendar 
                            # does not handle week no = 53
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
    # download(url)
    print("***** TESTING --- download new data paused")
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
            raise Exception("Error: stationInfoData and stationStatusData differs, extend code to handle it")
    
    ###################################################################################################
    # # Find stations with traffic for given week, loop thru all years
    fileList = os.listdir(tripDataPath)
    progress = Bar("CityBike 1b/5: Read data from files, find stations with traffic for given week", max = len(fileList))
    trafficAtStation = {} # indexed by id, stores stations with at least one arrival or departure 
    for file in fileList:
        if file.endswith(".json"):
            if int(file[5:7]) in weekMonths(week):
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
        progress.next()
    progress.finish()

    stationMap = {}
    stationNo = 0
    print("These stations without traffic in week ", str(week), " are neglected.")
    for stationId in stationNames:
        if not stationId in trafficAtStation:
            print(stationNames[stationId])
            pass
        else:
            stationMap[stationId] = stationNo
            stationNo += 1
    print()

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

    # DEBUG
    pragmatic = True
    longestSaved = []

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
        progress.next()
    progress.finish()

    totalBikes = 0
    for s in bikeStartStatus:
        totalBikes += int(s)
    if totalBikes == 0:
        raise Exception("*** Sorry, no bikes currently available for given city")

    # Create stations
    # capacities must be converted from map to list, to avoid changing code in state.py
    capacitiesList = []
    for stationId in stationMap:
        capacitiesList.append(stationCapacities[stationId])
    # same for bikeStartStatus
    bikeStartStatusList = []
    for stationId in stationMap:
        bikeStartStatusList.append(bikeStartStatus[stationId])

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
