# plots.py 
import json 
import os
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import init_state
import numpy as np
from helpers import yearWeekNoAndDay
from progress.bar import Bar
    
def cityPerWeekStats(p, textFile, city, weeksToMark, color):
    # TODO, no error handling yet. Assume data is loaded
    tripDataPath = init_state.tripDataDirectory + city
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
    lineLayout = '-' + color + 'D'
    # adjust weeksToMark since plotted results are in array [0..52] while weekNo is in [1..53]
    for i in range(len(weeksToMark)):
        weeksToMark[i] = weeksToMark[i] - 1
    p.plot(xValues, data, lineLayout, markevery=weeksToMark)
    pass

def cityTrafficStats(): # call this from main to produce figure used in paper
    print("Generating traffic statistics for all cities with stored traffic data")
    fig, p = plt.subplots()
    f = open(f"{init_state.tripDataDirectory}/CityTraffic.txt", "w")
    labels = []
    cityPerWeekStats(p, f, "oslobysykkel", [10, 22, 34, 46], 'b')
    labels.append("Oslo")
    cityPerWeekStats(p, f, "bergenbysykkel", [8, 25, 35, 45], 'r')
    labels.append("Bergen")
    cityPerWeekStats(p, f, "trondheimbysykkel", [17, 21, 34, 44], 'g')
    labels.append("Trondheim")
    cityPerWeekStats(p, f, "edinburghcyclehire", [10, 22, 31, 50], 'm')
    labels.append("Edinburgh")
    # cityPerWeekStats(p, f, "oslovintersykkel", [12, 33], 'b') # skipped due to very little trip data
    # labels.append("Oslo-vinter") 
    p.set_ylabel("Trips pr. week")
    p.set_xlabel("Week no (1..53)")
    p.legend(labels)
    plt.savefig(f"{init_state.tripDataDirectory}/CityTrafficPerWeek.pdf")
    plt.show()
    print("Plot is saved at CityTrafficPerWeek.pdf and numbers at CityTraffic.txt")
    
def lostTripsPlot(cities, policies, starv, serr, cong, cerr):
    # After 14 october cities is called instanceNames, and policies is analysisNames in the calling code. TODO could renew code to reflect this, if we get time to do so
    # This applies to other procedures also
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
        subPlots[city].legend()
    subPlots[city].set_ylabel("Violations (% of total number of trips)")

    plt.show()

def Surface3Dplot(bikes, policyNames, values, title):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    policiesPosition = range(len(policyNames))
    bikes, policiesPosition = np.meshgrid(bikes, policiesPosition)
    values2D = bikes + policiesPosition*100 # 2D numpyarray
    Z = values2D # Z behaves as a pointer/reference to values2D
    for p in range(len(policyNames)):  # copy results from 2D list to numpyarray
        for b in range(len(bikes[p])):
            Z[p][b] = values[p][b] 
    ax.set_xlabel("number of bikes")
    ax.set_ylabel("rebalancing policy")
    ax.set_yticks(range(len(policyNames)))
    ax.set_yticklabels(policyNames)
    norm = colors.Normalize(vmin=0, vmax=100)
    scmap = plt.cm.ScalarMappable(norm = norm, cmap="coolwarm")
    surf = ax.plot_surface(bikes, policiesPosition, Z, facecolors=scmap.to_rgba(Z), linewidth=0, antialiased=False, shade = False)
    ax.set_zlim([0, 100])
    # https://stackoverflow.com/questions/6390393/matplotlib-make-tick-labels-font-size-smaller
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    fig.colorbar(scmap, shrink=0.3, aspect=5, )
    fig.suptitle(title)
    return fig, ax

def Surface3DplotFraction(bikes, policyNames, starv, cong, title):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    policiesPosition = range(len(policyNames))
    bikes, policiesPosition = np.meshgrid(bikes, policiesPosition)
    Z = bikes + policiesPosition # 2D numpyarray, addition is not used, but gives Z the right shape, Z behaves now as a reference to a 2D-matrix
    Rint = np.copy(Z)
    R = Rint.astype(float) 
    for p in range(len(policyNames)): # copy results from 2D list to numpyarray
        for b in range(len(bikes[p])):
            Z[p][b] = starv[p][b] + cong[p][b] # Z-value to plot, with zero at zero, in range 0 - 120 or more
    for p in range(len(policyNames)): # copy results from 2D list to numpyarray
        for b in range(len(bikes[p])):
            R[p][b] = (starv[p][b]/Z[p][b])*100 # Starvation in % of Lost trips  
    norm = colors.Normalize(vmin=0, vmax=100)
    norm2 = colors.Normalize(vmin=0.0, vmax=100)
    scmap = plt.cm.ScalarMappable(norm = norm, cmap="coolwarm")
    scmap2 = plt.cm.ScalarMappable(norm = norm2, cmap=cm.BrBG)
    surf = ax.plot_surface(bikes, policiesPosition, Z, facecolors=scmap.to_rgba(Z), shade=False)
    surf2 = ax.plot_surface(bikes, policiesPosition, (R*0)-100, facecolors=scmap2.to_rgba(R), shade=False)
    ax.set_zlim(-100, 100) 
    ax.set_xlabel("number of bikes", weight='bold')
    ax.set_ylabel("rebalancing policy",  weight='bold')
    ax.set_yticks(range(len(policyNames)))
    ax.set_yticklabels(policyNames)
    # https://stackoverflow.com/questions/6390393/matplotlib-make-tick-labels-font-size-smaller
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    # yaxis-set_(policies, policyNames)
    cb = fig.colorbar(scmap, shrink=0.2, aspect=5, location = "right")
    cb2 = fig.colorbar(scmap2, shrink=0.2, aspect=5, location = "bottom")
    cb.set_label(label="Lost trips in " + '%' + " of total trips", size = 9)
    cb2.set_label(label = "Congestion - starvation ratio", size = 9)
    fig.suptitle(title)
    return fig, ax

def Surface3DplotTripsProfit(bikes, policyNames, trips, profit, title):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    policiesPosition = range(len(policyNames))
    bikes, policiesPosition = np.meshgrid(bikes, policiesPosition)
    Z = bikes + policiesPosition # 2D numpyarray, addition is not used, but gives Z the right shape, Z behaves now as a reference to a 2D-matrix
    Rint = np.copy(Z)
    R = Rint.astype(float) 
    for p in range(len(policyNames)): # copy results from 2D list to numpyarray
        for b in range(len(bikes[p])):
            Z[p][b] = trips[p][b]  # Z-value to plot, with zero at zero, in range 0 - 120 or more
    for p in range(len(policyNames)):
        for b in range(len(bikes[p])):
            R[p][b] = profit[p][b] 
    norm = colors.Normalize(vmin=0, vmax=100)
    norm2 = colors.Normalize(vmin=0.0, vmax=100)
    scmap = plt.cm.ScalarMappable(norm = norm, cmap="coolwarm")
    scmap2 = plt.cm.ScalarMappable(norm = norm2, cmap=cm.BrBG)
    surf = ax.plot_surface(bikes, policiesPosition, Z, facecolors=scmap.to_rgba(Z), shade=False) # upper surface
    surf2 = ax.plot_surface(bikes, policiesPosition, (R*0)-100, facecolors=scmap2.to_rgba(R), shade=False) # lower flat surface
    ax.set_zlim(-100, 100) 
    ax.set_xlabel("number of bikes", weight='bold')
    ax.set_ylabel("rebalancing policy",  weight='bold')
    ax.set_yticks(range(len(policyNames)))
    ax.set_yticklabels(policyNames)
    # https://stackoverflow.com/questions/6390393/matplotlib-make-tick-labels-font-size-smaller
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    # yaxis-set_(policies, policyNames)
    cb = fig.colorbar(scmap, shrink=0.2, aspect=5, location = "right")
    cb2 = fig.colorbar(scmap2, shrink=0.2, aspect=5, location = "bottom")
    cb.set_label(label="Trips/200", size = 9)
    cb2.set_label(label = "Profit/180kNOK", size = 9)
    fig.suptitle(title)
    return fig, ax
