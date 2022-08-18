# parse.py
import sim
import json
import os
import os.path
import requests
import geopy.distance
import datetime
from datetime import date
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
    progress = Bar("CityBike 1/5: Download datafiles   ", max = (datetime.date.today().year - 2018) * 12 + datetime.date.today().month - 1)
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
            print("*** Error: could not read station info from: " + address)
            return False
        stationInfoFile = open(f"{directory}/stationinfo.text", "w")
        stationInfoFile.write(stationInfo.text)
        stationInfoFile.close()

        # print("downloads station status")
        gbfsTailStatus = "/station_status.json"
        address = gbfsStart + extractCityAndDomainFromURL(url) + gbfsTailStatus  
        stationStatus =  requests.get(address)
        if stationStatus.status_code != 200: # 200 is OK, non-existent files will have status 404
            print("*** Error: could not read station status from: " + address)
            return False
        stationStatusFile = open(f"{directory}/stationstatus.text", "w")
        stationStatusFile.write(stationStatus.text)
        stationStatusFile.close()

    return newDataFound            

def get_initial_state(url="https://data.urbansharing.com/oslobysykkel.no/trips/v1/", week=30, number_of_vehicles=3, random_seed=1):
    """ Calls calcDistances to get an updated status of active stations in the given city. Processes all stored trips
        downloaded for the city, calculates average trip duration for every pair of stations, including
        back-to-start trips. For pair of stations without any registered trips an average duration is estimated via
        the trip distance and a global average BIKE_SPEED value from settings.py. This gives the travel_time matrix.
        Travel time for the vehicle is based on distance. All tripdata is read and used to calculate arrive and leave intensities 
        for every station and move probabilities for every pair of stations. These structures are indexed by station, week and hour.
        Station capacities and number of bikes in use is based on real_time data read at execution time. NOTE this will remove reproducibility of simulations
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
    download(url)

    years = [] 

    tripDataPath = tripDataDirectory + city

    stationInfoFile = open(f"{tripDataPath}/stationinfo.text", "r")
    allData = json.loads(stationInfoFile.read())
    stationInfoData = allData["data"]

    stationStatusFile = open(f"{tripDataPath}/stationstatus.text", "r")
    allData = json.loads(stationStatusFile.read())
    stationStatusData = allData["data"]

    stationNo = 0
    stationMap = {} # maps from station Id to stationNo that is index in most datastructures   
    stationLocations = [] # indexed by stationNo
    bikeStartStatus = []
    stationCapacities = []

    for i in range(len(stationInfoData["stations"])): # loop thru all active stations
        id = stationInfoData["stations"][i]["station_id"]
        long = stationInfoData["stations"][i]["lon"]
        lat = stationInfoData["stations"][i]["lat"]
        stationMap[id] = stationNo
        stationLocations.append(StationLocation(id, long, lat))
        bikeStartStatus.append(stationStatusData["stations"][i]["num_bikes_available"])
        stationCapacities.append(stationInfoData["stations"][i]["capacity"])
        stationNo += 1

    arriveCount = []
    leaveCount = []
    moveCount = []
    durations = []
    noOfStations = len(stationMap)
    for station in range(noOfStations):
        arriveCount.append([]) # arriving at this station
        leaveCount.append([]) # departing from this station
        moveCount.append([]) # moving fro this station to an end-station
        durations.append([]) # traveltime from this station to an end-station
        for s in range(noOfStations):
            durations[station].append([])
        for day in range(7): # weekdays here are 0..6
            arriveCount[station].append([])
            leaveCount[station].append([])
            moveCount[station].append([])
            for hour in range(24):
                arriveCount[station][day].append(0)
                leaveCount[station][day].append(0)
                stationList = []
                for i in range(noOfStations):
                    stationList.append(0)
                moveCount[station][day].append(stationList)

    # process all stored trips for given city, count trips and store durations for the given week number
    trips = 0 # total number from all tripdata read
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
                            durations[startStationNo][endStationNo].append(bikeData[i]["duration"])
                        trips = trips + 1
        progress.next()
    progress.finish()

    noOfYears = len(years)
    if noOfYears == 0:
        raise Exception("*** Sorry, no trip data found for given city and week")

    # Calculate average durations, durations in seconds
    progress = Bar("CityBike 3/5: Calculate durations  ", max = noOfStations)
    avgDuration = []

    for start in range(noOfStations):
        avgDuration.append([])
        for end in range(noOfStations):
            avgDuration[start].append([])
            sumDuration = 0
            noOfTrips = 0
            for trip in range(len(durations[start][end])):
                tripDuration = durations[start][end][trip]
                # print(tripDuration)
                noOfTrips += 1
                sumDuration += tripDuration     
            if noOfTrips > 0:
                avgDuration[start][end] = sumDuration/noOfTrips
            else:
                distance = geopy.distance.distance((stationLocations[start].latitude, stationLocations[start].longitude), 
                        (stationLocations[end].latitude, stationLocations[end].longitude)).km
                avgDuration[start][end] = (distance/settings.BIKE_SPEED)*3600
        progress.next()
    progress.finish()

    # Calculate traveltime_matrix, travel-times in minutes
    ttMatrix = []
    for start in range(noOfStations):
        ttMatrix.append([])
        for end in range(noOfStations):
            averageDuration = avgDuration[start][end]
            if averageDuration == 0:
                if start == end:
                    averageDuration = 7.7777 # TODO, improve, calculate all such positive averageDuarations, and use here
                else:
                    print("*** Error, averageDuration == 0 should not happen") # should be set to default bike-speed above
            ttMatrix[start].append(averageDuration/60)
    
    progress = Bar("CityBike 4/5: Calculate traveltime ", max = noOfStations)
    ttVehicleMatrix = []
    for start in range(noOfStations):
        ttVehicleMatrix.append([])
        for end in range(noOfStations):
            distance = geopy.distance.distance((stationLocations[start].latitude, stationLocations[start].longitude), 
               (stationLocations[end].latitude, stationLocations[end].longitude)).km
            ttVehicleMatrix[start].append((distance/settings.VEHICLE_SPEED)*60)
        progress.next()
    progress.finish()
    
    # Calculate arrive and leave-intensities and move_probabilities
    progress = Bar("CityBike 5/5: Calculate intensities", max = noOfStations)
    noOfYears = len(years)
    arrive_intensities = []  
    leave_intensities = []
    move_probabilities= []
    for station in range(noOfStations):
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
                for endStation in range(noOfStations):
                    movedBikes = moveCount[station][day][hour][endStation]
                    movedBikesTotal = leaveCount[station][day][hour]
                    if movedBikesTotal > 0:
                        move_probabilities[station][day][hour].append(movedBikes/movedBikesTotal)
                    else:
                        equalProb = 1.0/noOfStations    
                        move_probabilities[station][day][hour].append(equalProb)
        progress.next()
    progress.finish()

    totalBikes = 0
    for i in range(len(bikeStartStatus)):
        totalBikes += bikeStartStatus[i]
    if totalBikes == 0:
        raise Exception("*** Sorry, no bikes currently available for given city")

    # Create stations
    stations = sim.State.create_stations(num_stations=len(stationCapacities), capacities=stationCapacities)
    sim.State.create_bikes_in_stations(stations, "Bike", bikeStartStatus)
    sim.State.set_customer_behaviour(stations, leave_intensities, arrive_intensities, move_probabilities)
    # Create State object and return

    state = sim.State.get_initial_state(stations, number_of_vehicles, random_seed, ttMatrix, ttVehicleMatrix) 
    
    return state
