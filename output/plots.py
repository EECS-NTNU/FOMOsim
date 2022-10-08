# plots.py 
import json 
import os
import matplotlib.pyplot as plt
from init_state.cityBike.parse import tripDataDirectory
from helpers import yearWeekNoAndDay
from progress.bar import Bar
    
def cityPerWeekStats(p, textFile, city):
    # TODO, no error handling yet. Assume data is loaded
    tripDataPath = tripDataDirectory + city
    fileList = os.listdir(tripDataPath)
    yearsWeek = [] # for each week, count how many years we have data for that week
    tripsPerWeek = []
    for w in range(1, 54):
        yearsWeek.append([])
        tripsPerWeek.append(0)
    progress = Bar("Reading traffic for " + city, max = len(fileList))
    for file in fileList:
        if file.endswith(".json"):
            jsonFile = open(os.path.join(tripDataPath, file), "r")
            bikeData = json.loads(jsonFile.read())
            for i in range(len(bikeData)):
                year, weekNo, weekDay = yearWeekNoAndDay(bikeData[i]["ended_at"][0:10])
                index = weekNo - 1 # weeks 1..53 are stored in array 0..52
                if year not in yearsWeek[index]:
                    yearsWeek[index].append(year)
                tripsPerWeek[index] += 1
        progress.next()

    textFile.write("Traffic per week for " + city + ":\n")    
    xValues = []
    data = []
    for w in range(len(tripsPerWeek)):
        if len(yearsWeek[w]) > 0:
            textFile.write("Week " + str(w+1) + ": " + str(tripsPerWeek[w]/len(yearsWeek[w])))
            data.append(tripsPerWeek[w]/len(yearsWeek[w]))
            textFile.write(" (Total " + str(tripsPerWeek[w]) + " over " + str(len(yearsWeek[w])) + "years)\n")
        else:
            textFile.write("Week: " + str(w+1) + str(tripsPerWeek[w]) + "\n")
            data.append(tripsPerWeek[w])
        xValues.append(w+1)
    textFile.write("\n")       
    p.plot(xValues, data)

def cityTrafficStats():
    print("Generating traffic statistics for all cities with stored traffic data")
    fig, p = plt.subplots()
    f = open(f"{tripDataDirectory}/CityTraffic.txt", "w")
    labels = []
    cityPerWeekStats(p, f, "oslobysykkel")
    labels.append("Oslo")
    cityPerWeekStats(p, f, "bergenbysykkel")
    labels.append("Bergen")
    cityPerWeekStats(p, f, "trondheimbysykkel")
    labels.append("Trondheim")
    cityPerWeekStats(p, f, "edinburghcyclehire")
    labels.append("Edinburgh")
    # cityPerWeekStats("oslovintersykkel") # skipped due to very little trip data
    p.set_ylabel("Trips pr. week")
    p.set_xlabel("Week no (1..53)")
    p.legend(labels)
    plt.savefig(f"{tripDataDirectory}/CityTrafficPerWeek.pdf")
    plt.show()
    print("Plot is saved at CityTrafficPerWeek.pdf and numbers at CityTraffic.txt")
    