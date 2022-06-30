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
from datetime import date
import jsonpickle
import zipfile
from zipfile import ZipFile

import settings
from GUI import loggFile
from helpers import extractCityFromURL, extractCityAndDomainFromURL, yearWeekNoAndDay, write

tripDataDirectory = "init_state/cityBike/data/" # location of tripData
savedStatesDirectory = "init_state/cityBike/savedStates/" # location of saved states

def download(url):
    newData = False
    def loadMonth(monthNo):
        fileName = f"{yearNo}-{monthNo:02}.json"
        if fileName not in file_list:     
            address = f"{url}{yearNo}/{monthNo:02}.json"
            data = requests.get(address)
            if data.status_code == 200: # non-existent files will have status 404
                print(f"downloads {city} {fileName} ...")
                dataOut = open(f"{directory}/{fileName}", "w")
                dataOut.write(data.text)
                dataOut.close()
                newData = True

    city = extractCityFromURL(url)
    directory = f"{tripDataDirectory}/{city}"
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    file_list = os.listdir(directory)

    # these loops are a brute-force method to avoid implementing a web-crawler
    for yearNo in range(2018, datetime.date.today().year + 1): # 2018/02 is earliest data from data.urbansharing.com
        if yearNo < datetime.date.today().year:
            for month in range (1, 13):
                loadMonth(month)
        else: # will not load tripdata for current month since we do not want incomplete month-files (Reason is mainly the use of file existence in loadMonth)
            for month in range (1, datetime.date.today().month):
                loadMonth(month)
    return newData            


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

    if not download(url): # TODO WORK IN PROGRESS // checks if new data are available for the city, true if new data was loaded, and in that case state should not
                          # be restored from saved state
        # look in cache
        # if found in cache, load
        pass
        # savedStateFileName = task[1] + ".json"
        #     # print("before" + dateAndTimeStr()) # code for measuring time reduction when using ZIP
        #     loadStateFile = open(loadFileName, "r") # from JSON alternative
        #     string = loadStateFile.read()
        #     # with zipfile.ZipFile(loadFileName, mode = "r") as archive:
        #     #     string = archive.read(loadFileName) # NOTE, compression can be really efficient here

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

    # process all stored trips for given city, count trips and store durations for the given week number
    trips = 0 # total number from all tripdata read
    tripDataPath = tripDataDirectory + city
    fileList = os.listdir(tripDataPath)
    
    for file in fileList:
        if int(file[5:7]) in weekMonths(week):
            print("Reads from: ", file)
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
        # TODO progress bar # AD: Enig, progress bar er lett i python, men Behov redusert nå ? 
    
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

    # Create stations
    stations = sim.State.create_stations(num_stations=len(stationCapacities), capacities=stationCapacities)
    sim.State.create_bikes_in_stations(stations, bike_class, bikeStartStatus)
    sim.State.set_customer_behaviour(stations, leave_intensities, arrive_intensities, move_probabilities)

    # Create State object and return

    state = sim.State.get_initial_state(stations, number_of_vehicles, random_seed, ttMatrix, ttVehicleMatrix) 
    
    # if not os.path.isdir(savedStatesDirectory):
    #     os.makedirs(savedStatesDirectory, exist_ok=True) # first time
        
    # fileName = f"{savedStatesDirectory}/saved.json"
    # saveStateFile = open(fileName, "w")
    # saveStateFile.write(jsonpickle.encode(state))
    # saveStateFile.close()
    
    # # myZip = ZipFile(filename=f"{savedStatesDirectory}/saved.json", arcname = f"{savedStatesDirectory}/{city}_{week}.json""w")
    # # myZip.close()    

    return state
