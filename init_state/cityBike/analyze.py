# analyze.py_compile
# 
import os
import json
import matplotlib.pyplot as plt

from GUI.GUIhelpers import userError 

def openVisual1(): # TODO prototype code to serve a request from Steffen May 27. - 2022, should be cleaned, improved or removed 
    # perStation = True
    allStations = True
    stationId2NameMap = {} # map storing stationNames, key is stationId  
    stationIdList = []
    stationsFile = open("init_state/cityBike/data/Oslo/stations.txt", "r")
    for line in stationsFile.readlines():
        words = line.split()
        stationId = words[1]
        stationIdList.append(int(stationId))
        stationName = words[4] 
        for i in range(len(words) - 5):
            stationName += ("_" + words[i+5]) 
        stationId2NameMap[stationId] = stationName

#    if perStation:
        # stationList = [377, 386, 391, 483]
        # for s in stationList: ...
    if allStations: # NOTE this takes many hours, half a day, produce 15000 files, 265 Mbytes
        print("*** WARNING: this takes many hours, half a day, produce 15000 files, 265 Mbytes")
        for s in stationIdList:
            starts = []
            ends = []
            for h in range(24):
                starts.append([])
            for h in range(24):
                ends.append([])

            tripDataPath = "init_state/cityBike/data/Oslo/tripData"
            fileList = os.listdir(tripDataPath)
            for file in fileList:
                jsonFile = open(os.path.join(tripDataPath, file), "r")
                bikeData = json.loads(jsonFile.read())
                for i in range(len(bikeData)):
                    if s == int(bikeData[i]["start_station_id"]):
                        startHour = bikeData[i]["started_at"][11:13]
                        startMinute = bikeData[i]["started_at"][14:16]
                        starts[int(startHour)].append(int(startMinute))

                for i in range(len(bikeData)):
                    if s == int(bikeData[i]["start_station_id"]):
                        endHour = bikeData[i]["ended_at"][11:13]
                        endMinute = bikeData[i]["ended_at"][14:16]
                        ends[int(endHour)].append(int(endMinute))
            
            fig, axs = plt.subplots(8,3, sharex = True, sharey = True)
            fig.suptitle(stationId2NameMap[str(s)])
            for row in range(8):
                for col in range(3):
                    h = row*3 + col
                    axs[row, col].hist(starts[h], bins=60)
                    axs[row, col].set_title("Hour == " + str(h), fontsize=9, x = 0.3, y = 0.4)
            plt.savefig("HourDistribution_starts_all_" + stationId2NameMap[str(s)] + ".jpg")
            plt.close()

            fig, axs = plt.subplots(8,3, sharex = True, sharey = True)
            fig.suptitle(stationId2NameMap[str(s)])
            for row in range(8):
                for col in range(3):
                    h = row*3 + col
                    axs[row, col].hist(ends[h], bins=60)
                    axs[row, col].set_title("Hour == " + str(h), fontsize=9, x = 0.3, y = 0.4)
            plt.savefig("HourDistribution_ends_all_" + stationId2NameMap[str(s)] + ".jpg")
            plt.close()
           
            for h in range(24):
                plt.hist(starts[h], bins=60)
                plt.title(stationId2NameMap[str(s)] + " Hour = " + str(h))
                plt.grid(True)
                plt.savefig("HourDistribution_starts_" + stationId2NameMap[str(s)] + "_" + str(h) + ".jpg")
                plt.close()
            
            for h in range(24):
                plt.hist(ends[h], bins=60)
                plt.title(stationId2NameMap[str(s)] + " Hour = " + str(h))
                plt.grid(True)
                plt.savefig("HourDistribution_ends_"+  stationId2NameMap[str(s)] + "_" + str(h) + ".jpg")
                plt.close()       
    return

def openVisual2(resultFile):
    # TODO bikes is used also for scooters
    class Trip:
        def __init__(self, startTime, endTime, fromStation, toStation):
            self.startTime = startTime
            self.endTime = endTime
            self.fromStation = fromStation
            self.toStation = toStation     

    class Bike:
        def __init__(self, id):
            self.id = id
            self.startTime = -1
            self.endTime = -1
            self.fromStation = -1
            self.toStation = -1
    if resultFile == "":
        userError("empty resultFile")
        return
    traffic = open(resultFile, "r")
    trips = []
    lostTrips = 0
    allBikes = {}
    bikesParked = []
    bikesParkedPrevious = []
    lastDeparture = -1
    lastDepartureTime = -1
    lastArrival = -1
    lastArrivalTime = -1

    lines = traffic.readlines()
    lineNo = 0
    while lineNo < len(lines):
        words = lines[lineNo].split()
        if words[0] == "Time:" :
            time = words[1]
        elif words[0] == "Locations:" :
            noOfLocs = int(words[1])
            bikesParked = []
            for loc in range(noOfLocs):
                lineNo +=1
                words = lines[lineNo].split()
                bikes = int(words[1])
                for i in range(bikes):
                    bikeId = words[2 + 2*i]
                    if allBikes.get(bikeId) == None:
                        allBikes[bikeId] = Bike(bikeId)
                        allBikes[bikeId].fromStation = words[0]
                    bikesParked.append(bikeId)

            if lastDeparture >= 0 : # find bike used in last departure
                usedBike = set(bikesParkedPrevious) - set(bikesParked)
                if len(list(usedBike)) > 0 :
                    allBikes[list(usedBike)[0]].fromStation = str(lastDeparture)
                    allBikes[list(usedBike)[0]].startTime = str(lastDepartureTime)
                else:
                    pass
                    # TODO MARK *A print("lost trip at station " + str(lastDeparture) + " at time " + str(lastDepartureTime)) 
                lastDeparture = -1
            elif lastArrival >= 0 : # find bike that arrived  
                usedBike = set(bikesParked) - set(bikesParkedPrevious)
                if len(list(usedBike)) > 0 :
                    allBikes[list(usedBike)[0]].toStation = str(lastArrival)
                    allBikes[list(usedBike)[0]].endTime = str(lastArrivalTime)
                    trips.append(Trip(allBikes[list(usedBike)[0]].startTime, allBikes[list(usedBike)[0]].endTime,
                                allBikes[list(usedBike)[0]].fromStation, allBikes[list(usedBike)[0]].toStation))

            bikesParkedPrevious = bikesParked
        elif words[0] == "Departure-at-time:" :
            lastDepartureTime = int(words[1])
            lastDeparture = int(words[3])
            #### må bruke koden for å finne hvilken sykkel ble brukt, er gitt av endring i bikesParked 
            # fra n til neste gang, og det kan sjekkes mot departedfrom, IKKE HELT LETT , se på tid 40-48 sist
        elif words[0] == "In-use:" :
            pass
            # check that all in the in-use list have got a fromStation
            for bike in range(len(words) - 1):
                bikeId = "ID-" + words[bike+1]
                #print("Bike in use is no " + str(bikeId) + " left-station: " + str(allBikes[bikeId].fromStation)
                #    + " at-time: " + str(allBikes[bikeId].startTime))
            # bikesUsed.append(words[1]) # TODO extend to handle more than one bike in use
            pass
        elif words[0] == "Arrival-at-time:":
            lastArrival = int(words[3])
            lastArrivalTime = int(words[1])    
        elif words[0] == "LostTrip-at-time:" :
            lostTrips += 1
        elif words[0] == "usedTime(s):":
            secondsUsed = float(words[1])

        lineNo +=1

    xValues = []
    yValues = []
    for t in range(len(trips)):
        xValues.append(trips[t].endTime)
        yValues.append(trips[t].toStation)
    completedTrips = len(trips)
    totalTrips = completedTrips + lostTrips    
    print("#trips: " + str(totalTrips) + ", ", end='')
    if totalTrips > 0:
        print("%.1f"%((completedTrips/totalTrips)*100), "% completed,", end = '')
    if secondsUsed > 0:
        print(" TPS: ", "%.1f" % (totalTrips/secondsUsed) )    
    plt.plot(xValues, yValues)
    plt.show()
    
def openVisual3():
    plt.plot([1, 2, 3, 4], [1, 4, 9, 3])
    plt.show()

def openVisual4():
    print("Visual Test 4, accumulated traffic during 53 weeks in Oslo")
    osloTripsFile = open("GUI/results/sessionLog-Oslo-Init-1-53.txt", "r")
    textLines = osloTripsFile.readlines()
    data = []
    for i in range(len(textLines)):
        line = []
        for word in textLines[i].split():
             line.append(word)
        data.append(int(line[6]))  
    plt.plot(data)
    plt.show()
    pass

def plotSpeeds(list):
    plt.hist(list, bins=25)
    plt.show()