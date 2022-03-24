#!/bin/python3

import copy
import time
from progress.bar import IncrementalBar
import numpy as np

import sim
import clustering.scripts
import policies
import policies.fosen_haldorsen
from visualization.visualizer import visualize_analysis

PERIOD = 960 # 16 hours

simulators = []

###############################################################################

# Set up initial state

# Lasse: Jeg ser for meg å kalle en funksjon som ligner på følgende:
# state = lasses_pakke.get_initial_state(datadir="data/oslo", week=12)

# Din funksjon må da sette opp tilstanden ved å bruke sim.State.get_initial_state()
# Du ser et eksempel på det her:

# Først setter vi opp noen matriser
arrive_intensities = [] # 3D matrise som indekseres [station][day][hour]
leave_intensities = []  # 3D matrise som indekseres [station][day][hour]
move_probabilities = [] # 4D matrise som indekseres [from-station][day][hour][to-station]
for station in range(4): # eksempelet har 4 stasjoner
    arrive_intensities.append([])
    leave_intensities.append([])
    move_probabilities.append([])
    for day in range(7):
        arrive_intensities[station].append([])
        leave_intensities[station].append([])
        move_probabilities[station].append([])
        for hour in range(24):
            arrive_intensities[station][day].append(2) # fra denne stasjonen på gitt tidspunkt drar det 2 sykler i timer
            leave_intensities[station][day].append(2)  # fra denne stasjonen på gitt tidspunkt kommer det 2 sykler i timer
            move_probabilities[station][day].append([1/3, 1/3, 1/3, 1/3]) # sannsynlighetsfordeling for å dra til de forskjellige stasjonene
            move_probabilities[station][day][hour][station] = 0 # null i sannsynlighet for å bli på samme plass

state = sim.State.get_initial_state(
    bike_class = "Bike",
    distance_matrix = [ # km
        [0, 4, 2, 3],
        [4, 0, 5, 1],
        [2, 5, 0, 4],
        [3, 1, 4, 0],
    ],
    speed_matrix = [ # km/h
        [15, 15, 15, 15],
        [15, 15, 15, 15],
        [15, 15, 15, 15],
        [15, 15, 15, 15],
    ],
    main_depot = None,
    secondary_depots = [],
    number_of_scooters = [1, 3, 2, 4],
    number_of_vans = 2,
    random_seed = 1,
    arrive_intensities = arrive_intensities,
    leave_intensities = leave_intensities,
    move_probabilities = move_probabilities,
)
    
###############################################################################

# Set up first simulator
simulators.append(sim.Simulator(
    PERIOD,
    policies.DoNothing(),
    copy.deepcopy(state),
    verbose=True,
    label="DoNothing",
))

# Run first simulator
simulators[-1].run()

###############################################################################

# Visualize results
visualize_analysis(simulators)
