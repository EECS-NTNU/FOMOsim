#!/bin/python3
"""
FOMO simulator example
"""
from settings import *
import init_state
import init_state.cityBike
import target_state
import policies
import policies.fosen_haldorsen
import policies.haflan_haga_spetalen
import policies.gleditsch_hagen
import sim
import output
from helpers import timeInMinutes

DURATION = timeInMinutes(hours=1)
instance = 'EH_W22'
INSTANCE_DIRECTORY="instances"
analysis = dict(name="equalprob",
         target_state="equal_prob_target_state",
         policy="GreedyPolicy",
         policyargs={},
         numvehicles=1,
         day=0,
         hour=6)

def main():
    tstate = None
    if "target_state" in analysis:
        tstate = getattr(target_state, analysis["target_state"])
    initial_state = init_state.read_initial_state(INSTANCE_DIRECTORY + "/" + instance,
                                                          target_state=tstate,
                                                          number_of_stations=analysis.get("numstations", None),
                                                          number_of_bikes=analysis.get("numbikes", None),
                                                          )
    if analysis["numvehicles"] > 0:
        policyargs = analysis["policyargs"]
    policy = getattr(policies, analysis["policy"])(**policyargs)
    initial_state.set_vehicles([policy]*analysis["numvehicles"])

    simulator = sim.Simulator(
        initial_state = initial_state,
        start_time = timeInMinutes(days=analysis["day"], hours=analysis["hour"]),
        duration = DURATION,
        verbose = True,
    )
    simulator.run()

    # Output
    print(f"Simulation time = {DURATION} minutes")
    print(f"Total requested trips = {simulator.metrics.get_aggregate_value('trips')}")
    print(f"Starvations = {simulator.metrics.get_aggregate_value('starvation')}")
    print(f"Congestions = {simulator.metrics.get_aggregate_value('congestion')}")

    WEEK = int(instance[4:len(instance)])   # extracts week number from instance name
    output.write_csv(simulator, "output.csv", week=WEEK, hourly = False)
    # output.visualize_trips([simulator], title=("Week " + str(WEEK)), week=WEEK)
    # output.visualize_starvation([simulator], title=("Week " + str(WEEK)), week=WEEK)
    # output.visualize_congestion([simulator], title=("Week " + str(WEEK)), week=WEEK)
    output.visualize_heatmap([simulator], "trips")

if __name__ == "__main__":
    main()
