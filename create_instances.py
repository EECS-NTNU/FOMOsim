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

    dict(name="Oslo",
         url="https://data.urbansharing.com/oslobysykkel.no/trips/v1/",
         numbikes=2000,
         numstations=None,
         week=33),

    dict(name="Bergen",
         url="https://data.urbansharing.com/bergenbysykkel.no/trips/v1/",
         numbikes=1000,
         numstations=None,
         week=33),

    dict(name="Trondheim",
         url="https://data.urbansharing.com/trondheimbysykkel.no/trips/v1/",
         numbikes=1000,
         numstations=None,
         week=33),

    # dict(name="Oslo-vinter",
    #      url="https://data.urbansharing.com/oslovintersykkel.no/trips/v1/",
    #      numbikes=400,
    #      numstations=None,
    #      week=7),

    dict(name="Edinburgh",
         url="https://data.urbansharing.com/edinburghcyclehire.com/trips/v1/",
         numbikes=200,
         numstations=None,
         week=20),

]

###############################################################################

if __name__ == "__main__":

    for instance in instances:
        init_state.create_and_save_state(INSTANCE_DIRECTORY + "/" + instance["name"],
                                         source=init_state.cityBike,
                                         url=instance["url"],
                                         number_of_bikes=instance["numbikes"],
                                         number_of_stations=instance["numstations"],
                                         week=instance["week"],
                                         )
