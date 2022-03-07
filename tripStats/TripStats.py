# TripStats.py

tripStatsLogFile = open("tripStats/logs/logg.txt", "w")

def tripStatsInit(name):
    tripStatsLogFile.write("tripStats logg file opened for ")
    tripStatsLogFile .write(name + "\n")

def tripStatsClose():
    tripStatsLogFile.write("\nlogg file closed")
    tripStatsLogFile.close()

def loggTrip():
    tripStatsLogFile.write("logg called")
