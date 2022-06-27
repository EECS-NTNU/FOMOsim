# helpers.py
import shutil

from datetime import datetime, date
from GUI import trafficLogg, trafficLoggDir

# AD: Mye her (alt?) er ikke spesifikt for cityBike, bør flyttes ut til rota, eller til der de brukes

# AD: Denne kan du erstatte med re.sub()
def strip(removeStr, wholeStr): # strips from front
    start = wholeStr.find(removeStr)
    if start == 0:
        return wholeStr[len(removeStr):len(wholeStr)]
    else:
        print("*** ERROR could not remove ", removeStr)

def yearWeekNoAndDay(dateString):
    # AD: Bruke strptime, ikke parse stringen selv
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
    return datetime.now().strftime("%Y-%m-%d-%H_%M_%S") # : between H and M, and M and S replaced by _ to allow use in filenames

def readTime():
    return datetime.now().strftime("%H:%M:%S")
 
def write(file, words): # writes list of words to file and flush
    for i in range(len(words)):
        file.write(words[i])
        file.write(" ") 
    file.write("\n")    
    file.flush()    

def loggWrite(words):
    write(trafficLogg, words)

def saveTrafficLogg(timeStamp):
    shutil.copy(trafficLoggDir + "traffic.txt", trafficLoggDir + "traffic_" + timeStamp + ".txt")

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

def loggDepartures(stationId, timelist):
    loggMsg = ["Trips-generated-for-station:", str(stationId)]
    for t in timelist:
        loggMsg.append(str(t))
    write(trafficLogg, loggMsg)    

def loggEvent(event, times=[]):
    string = event.__repr__()
    words = string.split()
    if words[0] == "<ScooterDeparture":
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

# AD: Dette er tungvint, rart og vanskelig å vedlikeholde.  Er det ikke bedre å endre navn på PC-ene?
def fixComputerName(string): # Used for storing name of computer when logging simualtion results and performance
                             # also used to distinquish execution on WinPC in contrast to linux, in cases where it is needed
    if string == "LAPTOP-SBB45R3V":
        return string + "(Lasse-PC1)"
    elif string == "DESKTOP-CTHMSMJ":
        return string + "(Lasse-PC2)"
    elif string == "lasse-PC":
        return "Lasse-PC3"
    else:
        return string + "(unknown)"    

# AD: Bedre da å refaktorere slik at samme funksjon brukes begge steder
def get_duration(days, hour): # was get_time in Asbjørns recent main.py
    return 60*24*days + 60*hour
