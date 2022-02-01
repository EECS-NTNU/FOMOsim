#!/bin/python3

import copy
import time
from progress.bar import IncrementalBar

import classes
import clustering.scripts
import decision.value_functions
from visualization.visualizer import visualize_analysis

import globals

if __name__ == "__main__":
  NUM_SCOOTERS = 500
  NUM_CLUSTERS = 5
  NUM_VANS = 2
  DURATION = 10080

  donothing = classes.World(
    DURATION,
    decision.DoNothing(),
    clustering.scripts.get_initial_state(
      NUM_SCOOTERS,
      NUM_CLUSTERS,
      number_of_vans=NUM_VANS,
      number_of_bikes=0,
    ),
    verbose=True,
    visualize=False,
    label="DoNothing",
  )
  donothing.run()

  rebalancing = classes.World(
    DURATION,
    decision.RebalancingPolicy(),
    clustering.scripts.get_initial_state(
      NUM_SCOOTERS,
      NUM_CLUSTERS,
      number_of_vans=NUM_VANS,
      number_of_bikes=0,
    ),
    verbose=True,
    visualize=False,
    label="Rebalancing",
  )
  rebalancing.run()

  visualize_analysis([donothing, rebalancing])
