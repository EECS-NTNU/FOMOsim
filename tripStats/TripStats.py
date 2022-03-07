# TripStats.py

tripStatsLogFile = open("tripStats/logs/logg.txt", "w")

def tripStatsInit(name):
    tripStatsLogFile.write("tripStats logg file opened for ")
    tripStatsLogFile .write(name + "\n")

def tripStatsClose():
    tripStatsLogFile.write("\nlogg file closed")
    tripStatsLogFile.close()

def loggTripStart(time, state):
    tripStatsLogFile.write("Time " + str(time) + ":")
    for s in state.locations:
        tripStatsLogFile.write(" " + str(len(s.scooters)))

    tripStatsLogFile.flush()

def loggTripEnd():
    tripStatsLogFile.write(" -bike- ==> ")
    tripStatsLogFile.write(" after \n")
    tripStatsLogFile.flush()
