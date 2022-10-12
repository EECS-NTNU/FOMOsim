
import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt

def readVector(textLine):
    values = []
    for v in textLine.split():
        values.append(v)
    return values

results = open("output/costModel/simulatedTrips.txt")
policies = []
trips = []
starv = []
cong = []

bikes = readVector(results.readline())
while True:
    policyName = results.readline().rstrip()
    if policyName != "-end-":
        policies.append(policyName) # read result for one policy
        trips.append(readVector(results.readline()))
        starv.append(readVector(results.readline()))
        cong.append(readVector(results.readline()))
    else:
        break

incomeTrip = 20
def plot_function(starvation, congestion):
    fig, p = plt.subplots()
    for case in range(len(trips)):
        y = []
        for i in range(len(trips[case])):
            y.append(int(trips[case][i])*incomeTrip - int(starv[case][i])*starvation - int(cong[case][i])*congestion)
        p.plot(bikes,y)
        p.set_ylabel("Income (NOK)")
        p.set_xlabel("Number of bikes")
        p.set_xticks(range(len(bikes)))
        p.set_xticklabels(bikes)
        p.legend(policies)
    plt.show()

interact(plot_function, starvation = 2, congestion = 4)

print("bye bye")