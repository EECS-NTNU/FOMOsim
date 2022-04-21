# helpers.py

from datetime import datetime, date
from GUI import trafficLogg

def strip(removeStr, wholeStr):
    start = wholeStr.find(removeStr)
    if start == 0:
        return wholeStr[len(removeStr):len(wholeStr)]
    else:
        print("*** ERROR could not remove ", removeStr)

def yearWeekNoAndDay(dateString):
    year, month, day = map(int, dateString.split('-'))
    date1 = date(year, month, day)
    return year, int(date1.isocalendar()[1]), date1.weekday()

def timeInHoursAndMinutes(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return str(h) + ":" + str(m)

def printTime():
    print("Time =", datetime.now().strftime("%H:%M:%S"))

def dateAndTimeStr():
    return datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

def readTime():
    return datetime.now().strftime("%H:%M:%S")
 
def write(file, words): # writes list of words to file and flush
    for i in range(len(words)):
        file.write(words[i])
        file.write(" ") 
    file.write("\n")    
    file.flush()    

def loggTime(time):
    words = ["Time: ", str(time)]       
    write(trafficLogg, words)    

def loggLocations(state):
    write(trafficLogg, ["Locations:", str(len(state.locations))])
    for loc in range(len(state.locations)):
        words = [str(loc), str(len(state.locations[loc].scooters))]
        for sco in range(len(state.locations[loc].scooters)):
            scooterId = "ID-" + str(state.locations[loc].scooters[sco].id)
            scooterBatteryStat = str(state.locations[loc].scooters[sco].battery)
            words.append(scooterId)
            words.append(scooterBatteryStat)      
        write(trafficLogg, words)
    words = []    
    for i in range(len(state.scooters_in_use)):
        words.append(str(state.scooters_in_use[i].id))
    if len(words) > 0:
        write(trafficLogg, ["In-use:"] + words)        

def loggEvent(event):
    string = event.__repr__()
    words = string.split()
    if words[0] == "<GenerateScooterTrips":
        time = words[3][0:len(words[3])-1]
        write(trafficLogg, ["GenerateTrips-at-time:", time]) 
    elif words[0] == "<ScooterDeparture":
        time = words[3][0:len(words[3])-1]
        fromLocation = words[7][0:len(words[7])-1]
        write(trafficLogg, ["Departure-at-time:", time, "from:", fromLocation ]) 
    elif words[0] == "<ScooterArrival":
        time = words[3][0:len(words[3])-1]
        toLocation = words[7][0:len(words[7])-1]
        write(trafficLogg, ["Arrival-at-time:", time, "at:", toLocation ])
    elif words[0] == "<LostTrip":
        time = words[3][0:len(words[3])-1]
        write(trafficLogg, ["LostTrip-at-time:", time])
    elif words[0] == "<VehicleArrival":
        time = words[3][0:len(words[3])-1]
        write(trafficLogg, ["VehicleArrival-at-time:", time])
    else:
        pass
        print("*** ERROR: Tried to logg unknown event ??? ")
        
def fixComputerName(string):
    if string == "LAPTOP-SBB45R3V":
        return string + "(Lasse-PC1)"
    else:
        return string + "(better name needed?)"    

# def tripStatsClose():
#     tripStatsLogFile.write("\nlogg file closed")
#     tripStatsLogFile.close()

# def loggTripStart(time, state):
#     tripStatsLogFile.write("Time " + str(time) + ":")
#     for s in state.locations:
#         tripStatsLogFile.write(" " + str(len(s.scooters)))
#     tripStatsLogFile.flush()

# def loggTripEnd():
#     tripStatsLogFile.write(" -bike- ==> ")
#     tripStatsLogFile.write(" after \n")
#     tripStatsLogFile.flush()
