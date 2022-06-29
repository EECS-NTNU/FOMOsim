# parse.py
# AD: - Etter at state-objektet er laget første gang bør det lagres som json-fil
#     - get_initial_state() kan sjekke om det finnes en lagret state, bruk i såfall den
import sim
import json
import os
import os.path
import requests
import geopy.distance
import datetime

import settings
from GUI import loggFile
from helpers import extractCityFromURL, extractCityAndDomainFromURL, yearWeekNoAndDay, write

tripDataDirectory = "init_state/cityBike/data/" # location of tripData

def download(url):
    city = extractCityFromURL(url)
    directory = f"{tripDataDirectory}/{city}"
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    file_list = os.listdir(directory)

    # these loops are a brute-force method to avoid implementing a web-crawler
    for yearNo in range(2018, datetime.date.today().year + 1): # 2018/02 is earliest data from data.urbansharing.com
        for monthNo in range (1, 13): 
            fileName = f"{yearNo}-{monthNo:02}.json"
            if fileName not in file_list:     
                address = f"{url}{yearNo}/{monthNo:02}.json"
                data = requests.get(address)
                if data.status_code == 200: # non-existent files will have status 404
                    print(f"downloads {city} {fileName} ...")
                    dataOut = open(f"{directory}/{fileName}", "w")
                    dataOut.write(data.text)
                    dataOut.close()

def get_initial_state(url="https://data.urbansharing.com/oslobysykkel.no/trips/v1/", week=30, bike_class="Bike", number_of_vehicles=3, random_seed=1):
    """ Calls calcDistances to get an updated status of active stations in the given city. Processes all stored trips
        downloaded for the city, calculates average trip duration for every pair of stations, including
        back-to-start trips. For pair of stations without any registered trips an average duration is estimated via
        the trip distance and a global average SCOOTER_SPEED value from settings.py. This gives the travel_time matrix.
        Travel time for the vehicle is based on distance. All tripdata is read and used to calculate arrive and leave intensities 
        for every station and move probabilities for every pair of stations. These structures are indexed by station, week and hour.
        Station capacities and number of scooters in use is based on real_time data read at execution time. NOTE this will remove reproducibility of simulations
    """
    class StationLocation: 
        def __init__(self, stationId, longitude, latitude):
            self.stationId = stationId 
            self.longitude = longitude
            self.latitude = latitude

    download(url) # checks if new data are available for the city
    city = extractCityFromURL(url)
    print("get_initial_state starts analyzing traffic for city: " + city + " for week " + str(week) + ", setting up datastructures ... ") 
    years = [] # Must count no of "year-instances" of the given week that are analyzed

    gbfsStart = "https://gbfs.urbansharing.com/"
    gbfsTailInfo = "/station_information.json"
    address = gbfsStart + extractCityAndDomainFromURL(url) + gbfsTailInfo                                                                                                # read from 
    stationInfo =  requests.get(address)
    if stationInfo.status_code != 200: # non-existent files will have status 404
        print("*** Error: could not read station info from: " + address)
        return
    allData = json.loads(stationInfo.text)
    stationInfoData = allData["data"]

    gbfsTailStatus = "/station_status.json"
    address = gbfsStart + extractCityAndDomainFromURL(url) + gbfsTailStatus  
    stationStatus =  requests.get(address)
    if stationStatus.status_code != 200: # non-existent files will have status 404
        print("*** Error: could not read station status from: " + address)
        return
    allData = json.loads(stationStatus.text)
    stationStatusData = allData["data"]
    
    stationNo = 0
    stationMap = {} # maps from station Id to stationNo taht is index in most datastructures   
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
    for station in range(len(stationMap)):
        arriveCount.append([]) # arriving at this station
        leaveCount.append([]) # departing from this station
        moveCount.append([]) # moving fro this station to an end-station
        durations.append([]) # traveltime from this station to an end-station
        for s in range(len(stationMap)):
            durations[station].append([])
        for day in range(7):
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

    # process all stored trips for given city, count trips and store durations for the given week number
    trips = 0 # total number from all tripdata read
    tripDataPath = tripDataDirectory + city
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
                year, weekNo, weekDay = yearWeekNoAndDay(bikeData[i]["ended_at"][0:10])
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
        print(".", end='') # TODO replace with progress bar # AD: Enig, progress bar er lett i python
    
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
                distance = geopy.distance.distance((stationLocations[start].latitude, stationLocations[start].longitude), 
                        (stationLocations[end].latitude, stationLocations[end].longitude)).km
                avgDuration[start][end] = (distance/settings.SCOOTER_SPEED)*3600

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
         
    ttVehicleMatrix = []
    for start in range(len(stationMap)):
        ttVehicleMatrix.append([])
        for end in range(len(stationMap)):
            distance = geopy.distance.distance((stationLocations[start].latitude, stationLocations[start].longitude), 
               (stationLocations[end].latitude, stationLocations[end].longitude)).km
            ttVehicleMatrix[start].append((distance/settings.VEHICLE_SPEED)*60)
                       
    # Calculate arrive and leave-intensities and move_probabilities
    noOfYears = len(set(years)) # TODO, inefficient storing long list of years, use set from the start 
    arrive_intensities = []  
    leave_intensities = []
    move_probabilities= []
    for station in range(len(stationMap)):
        if station % 20 == 0: # show progress
            print(".", end='') # TODO LN-AD: progress bar
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

    totalBikes = 0
    for i in range(len(bikeStartStatus)):
        totalBikes += bikeStartStatus[i]
    write(loggFile, ["Init-state-based-on-traffic:", "trips:", str(trips), "week:", str(week), "years:", str(noOfYears), "bikesAtStart:", str(totalBikes), "city:", city])

    # AD: Hadde vært kjekt å cache ferdigprosesserte data her, så slipper man å kalkulere alt på nytt hver gang
    # AD: Du kan kalle State.save() her, og State.load() på begynnelsen av funksjonen   1) PRØV Å BRUKE DENNE først   2)  hvis min egen save load, så flytt koden inn her fra gui
    # AD: Lurer på om denne bør splittes opp i flere funksjoner, har blitt ganske stor og stygg  ASBJØRN TENKER på, ikke Lasse
    return sim.State.get_initial_state(
        bike_class = bike_class, 
        traveltime_matrix = ttMatrix, 
        traveltime_vehicle_matrix = ttVehicleMatrix,
        main_depot = False,
        secondary_depots = 0,
        number_of_scooters = bikeStartStatus,
        capacities = stationCapacities,
        number_of_vehicles = number_of_vehicles,
        random_seed = random_seed,
        arrive_intensities = arrive_intensities,
        leave_intensities = leave_intensities,
        move_probabilities = move_probabilities,
    )
