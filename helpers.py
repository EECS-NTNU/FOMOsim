# helpers.py

import shutil
import os

from datetime import datetime, date

###############################################################################
# time and date

def timeInMinutes(days=0, hours=0, minutes=0): 
    return 60*24*days + 60*hours + minutes

def yearWeekNoAndDay(dateString):
    y, w, d = datetime.strptime(dateString, "%Y-%m-%d").isocalendar() # returns day as 1..7
    d -= 1 # adjust day to ramge 0..6
    return (y, w, d)

def timeInHoursAndMinutes(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return str(h) + ":" + str(m)

def printTime():
    print("Time =", datetime.now().strftime("%H:%M:%S"))

def dateAndTimeStr():
    return datetime.now().strftime("%Y-%m-%d-%H_%M_%S") # : between H and M, and M and S replaced by _ to allow use in filenames

def readTime():
    return datetime.now().strftime("%H:%M:%S")
 
###############################################################################
# logging

trafficLoggDir = "log_files/" 

os.makedirs(trafficLoggDir, exist_ok=True)
trafficLogg = open(trafficLoggDir + "traffic.txt", "w")  

def writeWords(file, words): # writes list of words to file and flush
    for i in range(len(words)):
        file.write(words[i])
        file.write(" ") 
    file.write("\n")    
    file.flush()    

#------------------------------------------------------------------------------

def loggTime(time):
    words = ["Time: ", str(time)]       
    writeWords(trafficLogg, words)    

def loggLocations(state):
    writeWords(trafficLogg, ["Locations:", str(len(state.locations))])
    for loc in range(len(state.locations)):
        words = [str(loc), str(len(state.locations[loc].scooters))]
        for sco in state.locations[loc].scooters.values():
            scooterId = "ID-" + str(sco.id)
            scooterBatteryStat = str(sco.battery)
            words.append(scooterId)
            words.append(scooterBatteryStat)      
        writeWords(trafficLogg, words)
    words = []    
    for scooter in state.scooters_in_use.values():
        words.append(str(scooter.id))
    if len(words) > 0:
        writeWords(trafficLogg, ["In-use:"] + words)        

def loggDepartures(stationId, timelist):
    loggMsg = ["Trips-generated-for-station:", str(stationId)]
    for t in timelist:
        loggMsg.append(str(t))
    writeWords(trafficLogg, loggMsg)    

def loggEvent(event, times=[]):
    string = event.__repr__()
    words = string.split()
    if words[0] == "<GenerateScooterTrips":
        pass
    elif words[0] == "<ScooterDeparture":
        time = words[3][0:len(words[3])-1]
        fromLocation = words[7][0:len(words[7])-1]
        writeWords(trafficLogg, ["Departure-at-time:", time, "from:", fromLocation ]) 
    elif words[0] == "<ScooterArrival":
        time = words[3][0:len(words[3])-1]
        toLocation = words[7][0:len(words[7])-1]
        writeWords(trafficLogg, ["Arrival-at-time:", time, "at:", toLocation ])
    elif words[0] == "<LostTrip":
        time = words[3][0:len(words[3])-1]
        writeWords(trafficLogg, ["LostTrip-at-time:", time])
    elif words[0] == "<VehicleArrival":
        time = words[3][0:len(words[3])-1]
        writeWords(trafficLogg, ["VehicleArrival-at-time:", time])
    else:
        print(f"*** ERROR: Tried to log unknown event {words[0][1:]} ")
