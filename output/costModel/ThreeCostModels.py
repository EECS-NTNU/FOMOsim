import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual
import matplotlib.pyplot as plt

def readVector(textLine):
    values = []
    for v in textLine.split():
        values.append(v)
    return values

results = open("output/costModel/simulatedTrips.txt")
bikes = readVector(results.readline())
policyName = results.readline().rstrip()
trips = readVector(results.readline())
starv = readVector(results.readline())
cong = readVector(results.readline())

fig, p = plt.subplots()

incomeTrip = 20
cases = [[0, 0], [10, 2], [2, 10], [6, 6]]

caseNames = []
for case in cases:
    income = []
    starvCost = case[0]
    congCost = case[1]
    caseNames.append("S=" + str(starvCost) + "  - C=" + str(congCost))
    for i in range(len(trips)):
        income.append(int(trips[i])*incomeTrip - int(starv[i])*starvCost - int(cong[i])*congCost)
    p.plot(bikes, income)

p.set_ylabel("Income (NOK)")
p.set_xlabel("Number of bikes")
p.set_xticks(range(len(bikes)))
p.set_xticklabels(bikes)
p.legend(caseNames)

plt.show()

print("bye bye")