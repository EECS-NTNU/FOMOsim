#!/bin/python3
"""
FOMO simulator, create jobs to run on cluster
"""
import os
import shutil
import json

import init_state
import init_state.cityBike

from helpers import *

INSTANCE_DIRECTORY="instances"

###############################################################################

# Enter instance definition here.  For numbikes and numstations, enter 'None' to use dataset default
instances = [
    dict(name="Oslo",        url="https://data.urbansharing.com/oslobysykkel.no/trips/v1/",        numbikes=2000, numstations=None, week=33, day=0, hour=6),
    dict(name="Bergen",      url="https://data.urbansharing.com/bergenbysykkel.no/trips/v1/",      numbikes=1000, numstations=None, week=33, day=0, hour=6),
    dict(name="Trondheim",   url="https://data.urbansharing.com/trondheimbysykkel.no/trips/v1/",   numbikes=1000, numstations=None, week=33, day=0, hour=6),
#    dict(name="Oslo-vinter", url="https://data.urbansharing.com/oslovintersykkel.no/trips/v1/",    numbikes=400,  numstations=None, week=7,  day=0, hour=6),
#    dict(name="Edinburgh",   url="https://data.urbansharing.com/edinburghcyclehire.com/trips/v1/", numbikes=200,  numstations=None, week=20, day=0, hour=6),
]

###############################################################################

if __name__ == "__main__":

    for instance in instances:
        init_state.create_and_save_state(INSTANCE_DIRECTORY + "/" + instance["name"],
                                         source=init_state.cityBike,
                                         url=instance["url"],
                                         week=instance["week"],
                                         )
