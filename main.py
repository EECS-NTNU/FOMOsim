#!/bin/python3

import copy
import time
from progress.bar import IncrementalBar
import numpy as np

import sim
import clustering.scripts
import policies
from visualization.visualizer import visualize_analysis

import globals

initial_state = clustering.scripts.get_initial_state(
    500,
    5,
    number_of_vans=1,
    number_of_bikes=0,
)

id = 0
stations = []
for statId in range(5):
    scooters = []
    for scooterId in range(10):
        scooters.append(sim.Scooter(scooter_id=id+scooterId))
    id += 10
    stations.append(sim.Station(statId, scooters, trip_intensity_per_iteration=1.0))
    
depots = [sim.Depot(depot_id=5+i) for i in range(2)]
vehicles = [sim.Vehicle(i, stations[i], 10, 10) for i in range(3)]
state = sim.State(stations, depots, vehicles)

distance_matrix = []
for location in state.locations:
    neighbour_distance = []
    for neighbour in state.locations:
        if location == neighbour:
            neighbour_distance.append(0.0)
        else:
            neighbour_distance.append(1)
    distance_matrix.append(neighbour_distance)
state.distance_matrix = distance_matrix

move_prob = np.zeros((len(stations), len(stations)), dtype="float64")
state.set_probability_matrix(move_prob)

simulator = sim.Simulator(
    10080,
    policies.RebalancingPolicy(),
#    policies.DoNothing(),
    state,
    verbose=True,
    visualize=False,
    label="World",
)
simulator.run()

visualize_analysis([simulator])
