#!/bin/python3
"""
FOMO simulator, create jobs to run on cluster
"""

import init_state
import init_state.cityBike

from helpers import *

INSTANCE_DIRECTORY="instances"

###############################################################################

instances = [

    dict(city="Oslo",
         abbrv="OS",
         url="https://data.urbansharing.com/oslobysykkel.no/trips/v1/",
         mapdata=("oslo.png", (10.6365, 10.8631, 59.8843, 59.9569)),
         numbikes=2000,
         numstations=None,
         weeks=[10,22,31,50]),

    dict(city="Bergen",
         abbrv="BG",
         url="https://data.urbansharing.com/bergenbysykkel.no/trips/v1/",
         mapdata=("bergen.png", (5.2484, 5.3953, 60.3501, 60.4346)),
         numbikes=1200,
         numstations=None,
         weeks=[8,25,35,45]),

    dict(city="Trondheim",
         abbrv="TD",
         url="https://data.urbansharing.com/trondheimbysykkel.no/trips/v1/",
         mapdata=("trondheim.png", (10.3339, 10.4808, 63.3930, 63.4597)),
         numbikes=1000,
         numstations=None,
         weeks=[17,21,34,44]),

    dict(city="Edinburgh",
         abbrv="EH",
         url="https://data.urbansharing.com/edinburghcyclehire.com/trips/v1/",
         mapdata=("edinburgh.png", (-3.2592, -3.1122, 55.9109, 55.9936)),
         numbikes=200,
         numstations=None,
         weeks=[10,22,31,50]),

]

###############################################################################

if __name__ == "__main__":

    for instance in instances:
        for week in instance["weeks"]:
            instance_name = instance["abbrv"]+'_W'+str(week)
            init_state.create_and_save_state(instance_name, 
                                             INSTANCE_DIRECTORY + "/" + instance_name,
                                             source=init_state.cityBike,
                                             url=instance["url"],
                                             number_of_bikes=instance["numbikes"],
                                             number_of_stations=instance["numstations"],
                                             week=week,
                                             mapdata=instance["mapdata"],
                                             )
