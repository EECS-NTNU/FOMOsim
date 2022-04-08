# helpers.py

from datetime import datetime, date
from GUI import loggFile

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

def loggLocations(state):
    words = []
    for loc in range(len(state.locations)):
        atLocation = []
        for sco in range(len(state.locations[loc].scooters)):
            scooterId = "ID-" + str(state.locations[loc].scooters[sco].id)
            scooterBatteryStat = str(state.locations[loc].scooters[sco].battery)
            atLocation.append(scooterId)
            atLocation.append(scooterBatteryStat)
        words = ["Location:", str(loc)]
        words += atLocation        
        write(loggFile, words)    

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
