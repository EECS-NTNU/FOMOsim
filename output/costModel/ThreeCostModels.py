import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual
import matplotlib.pyplot as plt

def readVector(textLine):
    values = []
    for v in textLine.split():
        values.append(v)
    return values

results = open("output/costModel/simulatedTripsPaperFig.txt")
bikes = readVector(results.readline())
policyName = results.readline().rstrip()
trips = readVector(results.readline())
starv = readVector(results.readline())
cong = readVector(results.readline())

fig, p = plt.subplots()

incomeTrip = 20
cases = [[0, 0], [8, 2], [2, 8], [5, 5]]

caseNames = []
for case in cases:
    income = []
    starvCost = case[0]
    congCost = case[1]
    caseNames.append("Starvation=" + str(starvCost) + ", Congestion=" + str(congCost))
    for i in range(len(trips)):
        income.append(int(trips[i])*incomeTrip - int(starv[i])*starvCost - int(cong[i])*congCost)
    p.plot(bikes, income)

p.set_ylabel("Income (NOK)")
p.set_xlabel("Number of bikes")
p.set_xticks(range(len(bikes)))
p.set_xticklabels(bikes)
for label in p.get_xticklabels():
    label.set_rotation(40)
    label.set_fontsize('x-small')
p.legend(caseNames)

plt.show()

print("bye bye")