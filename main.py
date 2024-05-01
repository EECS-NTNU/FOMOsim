"""
FOMO simulator example
"""
from settings import *
from init_state import read_initial_state
# import init_state.json_source
# import init_state.csv_source
import target_state
import policies
# import policies.fosen_haldorsen
# import policies.haflan_haga_spetalen
# import policies.gleditsch_hagen
# import policies.inngjerdingen_moeller
# import policies.hlm
import policies.hlv_master
import sim
import output
import demand
from helpers import timeInMinutes
import time
import random

# from output.plots import cityTrafficStats

START_TIME = timeInMinutes(hours=7)
DURATION = timeInMinutes(days=2)
INSTANCE = 'TD_W34'
WEEK = 34

def main(seed):

    ###############################################################################
    # Get initial state

    # the following is for creating a new initial state from trip data
    # state = init_state.get_initial_state(source=init_state.json_source,
    #                                      name="Trondheim",
    #                                      urlHistorical="https://data.urbansharing.com/trondheimbysykkel.no/trips/v1/",
    #                                      urlGbfs="https://gbfs.urbansharing.com/trondheimbysykkel.no",
    #                                      week=34)

    # start_time = time.time()

    # the following is for reading a precalculated initial state from a json file
    state = read_initial_state(sb_jsonFilename = SB_INSTANCE_FILE, ff_jsonFilename = FF_INSTANCE_FILE)
    state.set_seed(seed)
    
    ###############################################################################
    # print("Policy: BS_PILOT, Seed:", seed)
    # policy = policies.hlv_master.BS_PILOT()
    # state.set_sb_vehicles([policy])

    print("Policy: BS_PILOT_FF, Seed:", seed)
    policy_ff = policies.hlv_master.BS_PILOT_FF()
    state.set_ff_vehicles([policy_ff])

    # print("Policy: FF_Collab2, Seed:", seed)
    # policy2_ff = policies.hlv_master.FF_Collab2()
    # state.set_ff_vehicles([policy2_ff]) 
    
    # print("Policy: SB_Collab2, Seed:", seed)
    # policy2_sb = policies.hlv_master.SB_Collab2()
    # state.set_sb_vehicles([policy2_sb]) 
    
    # print("Policy: Collab3, Seed:", seed)
    # policy3 = policies.hlv_master.Collab3()
    # state.set_vehicles([policy3])

    # print("Policy: Collab4, Seed:", seed)
    # policy4 = policies.hlv_master.Collab4()
    # state.set_vehicles([policy4]) 

    ###############################################################################
    # Set up target state
    tstate = target_state.HLVTargetState(FF_TARGET_STATE_FILE)

    ###############################################################################
    # Set up demand

    dmand = demand.Demand()

    ###############################################################################
    # Set up simulator

    simulator = sim.Simulator(
        initial_state = state,
        target_state = tstate,
        demand = dmand,
        start_time = START_TIME,
        duration = DURATION,
        verbose = True,
    )
    simulator.run()

    # Output to console
    print(f"Simulation time = {DURATION} minutes")
    print(f"Total requested trips = {simulator.metrics.get_aggregate_value('trips')}")
    print(f"Bike departures = {simulator.metrics.get_aggregate_value('bike departure')}")
    print(f"Escooter departures = {simulator.metrics.get_aggregate_value('escooter departure')}")
    print(f"Bike arrival = {simulator.metrics.get_aggregate_value('bike arrival')}")
    print(f"Escooter arrival = {simulator.metrics.get_aggregate_value('escooter arrival')}")
    print(f"Vehicle arrivals = {simulator.metrics.get_aggregate_value('vehicle arrivals')}")
    print(f"Starvations = {simulator.metrics.get_aggregate_value('starvations')}")
    print(f"Roaming for bikes = {simulator.metrics.get_aggregate_value('roaming for bikes')}")
    print(f"Roaming distance for bikes = {round(simulator.metrics.get_aggregate_value('roaming distance for bikes'), 2)} km")
    print(f"Congestions = {simulator.metrics.get_aggregate_value('long congestions')}")
    print(f"Roaming distance for locks = {round(simulator.metrics.get_aggregate_value('roaming distance for locks'), 2)} km")
    print(f"help pickup = {round(simulator.metrics.get_aggregate_value('num helping bike pickups'), 2)}")
    print(f"help delivery = {round(simulator.metrics.get_aggregate_value('num helping bike deliveries'), 2)}")

    print("Failed events =", simulator.metrics.get_aggregate_value('failed events'))
    print("Starvations =", simulator.metrics.get_aggregate_value('starvations'))
    print("Escooter starvations =", simulator.metrics.get_aggregate_value('escooter starvations'))
    print("Bike starvations =", simulator.metrics.get_aggregate_value('bike starvations'))
    print("Battery starvations =", simulator.metrics.get_aggregate_value('battery starvations'))
    print("Battery violations =", simulator.metrics.get_aggregate_value('battery violations'))
    print("Long congestions =", simulator.metrics.get_aggregate_value('long congestions'))
    print("Short congestions =", simulator.metrics.get_aggregate_value('short congestions'))

    print("Number of bike deliveries =", simulator.metrics.get_aggregate_value('num bike deliveries'))
    print("Number of bike pickups =", simulator.metrics.get_aggregate_value('num bike pickups'))
    print("Number of battery swaps =", simulator.metrics.get_aggregate_value('num battery swaps'))
    print("Number of escooter deliveries =", simulator.metrics.get_aggregate_value('num escooter deliveries'))
    print("Number of escooter pickups =", simulator.metrics.get_aggregate_value('num escooter pickups'))
    print("Number of helping bike deliveries =", simulator.metrics.get_aggregate_value('num helping bike deliveries'))
    print("Number of helping bike pickups =", simulator.metrics.get_aggregate_value('num helping bike pickups'))
    print("Number of helping escooter deliveries =", simulator.metrics.get_aggregate_value('num helping escooter deliveries'))
    print("Number of helping escooter pickups =", simulator.metrics.get_aggregate_value('num helping escooter pickups'))
    # results_visualizer = policies.inngjerdingen_moeller.manage_results.VisualizeResults(simulator)
    # results_visualizer.visualize_violations_and_roaming()
    # results_visualizer.visualize_total_roaming_distances()
    # results_visualizer.visualize_average_roaming_distances()
    # results_visualizer.visualize_share_of_events()

# If comparissons between roaming=True and roaming=False: 
    # print(f"Different station choices = {simulator.metrics.get_aggregate_value('different_station_choice')}")
    # print(f"Different pickup quantities = {simulator.metrics.get_aggregate_value('different_pickup_quantity')}")
    # print(f"Different deliver quantities = {simulator.metrics.get_aggregate_value('different_deliver_quantity')}")
    # print(f"Number of overlaps = {simulator.metrics.get_aggregate_value('overlap')}")
    # print(f"Number of identical choices = {simulator.metrics.get_aggregate_value('same_choice')}")
    # print(f"Number of subproblems solved = {simulator.metrics.get_aggregate_value('number_of_subproblems')}")
    
    # Output to file

    # output.write_csv(simulator, "output.csv", hourly = False)

    # Plot to screen

    # output.visualize([simulator.metrics], metric="trips")
    # output.visualize([simulator.metrics], metric="starvation")
    # output.visualize([simulator.metrics], metric="congestion")
    # output.visualize_heatmap([simulator], metric="trips")
    
    # output.visualize([simulator.metrics], metric="roaming for bikes")
    # output.visualize([simulator.metrics], metric="roaming distance for bikes")
    # output.visualize([simulator.metrics], metric="roaming distance for locks")

# If comparissons between roaming=True and roaming=False : 
    # output.visualize([simulator.metrics], metric="different_station_choice")
    # output.visualize([simulator.metrics], metric="different_pickup_quantity")
    # output.visualize([simulator.metrics], metric="different_deliver_quantity")
    # output.visualize([simulator.metrics], metric="number_of_subproblems")
    
    
# show travel times for a given bike
    # bikes = simulator.state.get_all_bikes()
    # bikes = sorted(bikes, key=lambda bike: bike.metrics.getLen("travel_time"), reverse=True)
    # print(f"Bike {bikes[11].bike_id}: {bikes[11].metrics.getSum('travel_time')} {bikes[11].metrics.getSum('travel_time_congested')}")
    # output.visualize([bikes[11].metrics], metric="travel_time")
    # output.visualize([bikes[11].metrics], metric="travel_time_congested")

if __name__ == "__main__":
    # seed_list = [random.randint(1, 3000) for _ in range(10)]
    seed_list = [10]
    for seed in seed_list:
        main(seed)