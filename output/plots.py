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
    
def lostTripsPlot(cities, policies, starv, serr, cong, cerr):
    fig, plots = plt.subplots(nrows=1, ncols=len(cities), sharey=True)
    subPlots = [] # fix that subPlots returns a list only when more than one plot. Tried squeezy=False but it did not work for this purpose 
    if len(cities) == 1:
        subPlots.append(plots)
    else:
        for p in plots:
            subPlots.append(p)

    fig.suptitle("FOMO simulator - lost trips results\nImprovement from baseline (left bar) in % ", fontsize=15)
    w = 0.3
    pos = []
    for city in range(len(cities)):
        pos.append([])
        for i in range(len(policies)): # IMPROVED by more readable code here
            pos[city].append(starv[city][i] + cong[city][i])
        baseline = starv[city][0] + cong[city][0] # first policy is always baseline
        
        policyLabels = [] # fix policy labels
        for i in range(len(policies)):
            if i > 0:
                improved =  ( ((starv[city][i] + cong[city][i]) - baseline)/baseline)*100.0
                policyLabels.append(policies[i] + "\n(" + "{:.1f}".format(improved) + "%)")
            else:
                policyLabels.append(policies[i]) # label for baseline

        subPlots[city].bar(policyLabels, starv[city], w, label='Starvation')
        subPlots[city].errorbar(policyLabels, starv[city], yerr = serr[city], fmt='none', ecolor='black')
        subPlots[city].bar(policyLabels, cong[city], w, bottom=starv[city], label='Congestion')

        delta = 0.03 # skew the upper error-bar horisontally with delta to avoid that they can overwrite each other
        policiesPlussDelta = []
        for i in range(len(policies)):
            policiesPlussDelta.append(i + delta)
        subPlots[city].errorbar(policiesPlussDelta, pos[city], yerr= cerr[city], fmt='none', ecolor='black')
        subPlots[city].set_xlabel(cities[city])
        subPlots[city].set_ylabel("Violations (% of total number of trips)")
        subPlots[city].legend()

    plt.show()