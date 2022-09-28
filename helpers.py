# helpers.py

import shutil
import os
import numpy as np
<<<<<<< Updated upstream
=======
import itertools
>>>>>>> Stashed changes

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
# file locking

try:
    import fcntl

    def lock(filename):
        print("Waiting for lock", filename)
        lockname = filename + ".LOCK"
        fd = open(lockname, 'w+')
        fcntl.lockf(fd, fcntl.LOCK_EX)
        print("Got lock")
        return (fd,lockname)

    def unlock(handle):
        handle[0].close()
    #    os.remove(handle[1])

except ModuleNotFoundError:
    def lock(filename):
        return filename

    def unlock(handle):
        pass

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
        words = [str(loc), str(len(state.locations[loc].bikes))]
        for sco in state.locations[loc].bikes.values():
            bikeId = "ID-" + str(sco.id)
            bikeBatteryStat = str(sco.battery)
            words.append(bikeId)
            words.append(bikeBatteryStat)      
        writeWords(trafficLogg, words)
    words = []    
    for bike in state.bikes_in_use.values():
        words.append(str(bike.id))
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
    if words[0] == "<GenerateBikeTrips":
        pass
    elif words[0] == "<BikeDeparture":
        time = words[3][0:len(words[3])-1]
        fromLocation = words[7][0:len(words[7])-1]
        writeWords(trafficLogg, ["Departure-at-time:", time, "from:", fromLocation ]) 
    elif words[0] == "<BikeArrival":
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


###############################################################################
# weight analysis

def get_feasible_range(rest,lower,upper,delta):
    if rest < lower:
        range_output = [0]
    else:
        if rest <= upper:
            ub = rest
        else:
            ub = upper
        range_output = np.arange(lower,ub+delta,delta)
    return range_output
    

def get_criticality_weights(delta, w1_range, w2_range,w3_range,w4_range):
    weights = list()
    precision = 3
    for w1 in np.arange(w1_range[0],w1_range[1]+delta,delta):
        rest = 1-w1
        for w2 in get_feasible_range(rest,w2_range[0],w2_range[1],delta):
            rest = 1-w1-w2
            for w3 in get_feasible_range(rest,w3_range[0],w3_range[1],delta):
                w4 = 1-w1-w2-w3
                values = (w1,w2,w3,w4)
                weights.append([round(value,precision) for value in values])
    return weights
<<<<<<< Updated upstream
=======

def get_criticality_weights2(num_weights):
    
    all_weights = []
    weights_base = np.repeat(0,num_weights)
    #single measure
    for i in range(num_weights):
        weight = copy.copy(weights_base)
        weight[i] = 1
        all_weights.append(weight)
    
    #combination of two:
    factors = [1/x for x in [2,3,4]]
    subsets = list(itertools.combinations([0,1,2,3], 2))
    for (i,j) in subsets:
        
    num_weights = 2
    #flat
    factor_all = 1/num_weights
    #linear increase
    
    
    
    for i in range(num_weights):
        for factor in factors:
            weight = copy.copy(weights_base)
            weight[i] = 
    
    
    
    #flat strategy 
>>>>>>> Stashed changes
