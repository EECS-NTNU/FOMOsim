#!/bin/python3
"""
FOMO simulator, create jobs to run on cluster
"""

import init_state
import init_state.cityBike
import init_state.csv_reader

from helpers import *

INSTANCE_DIRECTORY="instances"

###############################################################################

instances = [

    dict(source=init_state.csv_reader,
         city="NewYork",
         abbrv="NY",
         urlHistorical="https://s3.amazonaws.com/tripdata/",
         urlGbfs="http://gbfs.citibikenyc.com/gbfs/en",
         filename_format="%Y%m-citibike-tripdata.csv.zip",
         mapdata=None,
         numbikes=None,
         numstations=None,
         weeks=[31]),

    dict(source=init_state.csv_reader,
         city="Boston",
         abbrv="BO",
         urlHistorical="https://s3.amazonaws.com/hubway-data/",
         urlGbfs="https://gbfs.bluebikes.com/gbfs/en",
         filename_format="%Y%m-bluebikes-tripdata.zip",
         mapdata=None,
         numbikes=None,
         numstations=None,
         weeks=[31]),

    dict(source=init_state.csv_reader,
         city="Chicago",
         abbrv="CH",
         urlHistorical="https://divvy-tripdata.s3.amazonaws.com/",
         urlGbfs="https://gbfs.divvybikes.com/gbfs/en",
         filename_format="%Y%m-divvy-tripdata.zip",
         mapdata=None,
         numbikes=None,
         numstations=None,
         weeks=[31]),

    # dict(source=init_state.cityBike,
    #      city="Oslo",
    #      abbrv="OS",
    #      urlHistorical="https://data.urbansharing.com/oslobysykkel.no/trips/v1/",
    #      mapdata=("oslo.png", (10.6365, 10.8631, 59.8843, 59.9569)),
    #      numbikes=2885,
    #      numstations=None,
    #      weeks=[10,22,31,50]),

    # dict(source=init_state.cityBike,
    #      city="Bergen",
    #      abbrv="BG",
    #      urlHistorical="https://data.urbansharing.com/bergenbysykkel.no/trips/v1/",
    #      mapdata=("bergen.png", (5.2484, 5.3953, 60.3501, 60.4346)),
    #      numbikes=1000,
    #      numstations=None,
    #      weeks=[8,25,35,45]),

    # dict(source=init_state.cityBike,
    #      city="Trondheim",
    #      abbrv="TD",
    #      urlHistorical="https://data.urbansharing.com/trondheimbysykkel.no/trips/v1/",
    #      mapdata=("trondheim.png", (10.3339, 10.4808, 63.3930, 63.4597)),
    #      numbikes=768,
    #      numstations=None,
    #      weeks=[17,21,34,44]),

    # dict(source=init_state.cityBike,
    #      city="Edinburgh",
    #      abbrv="EH",
    #      urlHistorical="https://data.urbansharing.com/edinburghcyclehire.com/trips/v1/",
    #      mapdata=("edinburgh.png", (-3.2592, -3.1122, 55.9109, 55.9936)),
    #      numbikes=140,
    #      numstations=None,
    #      weeks=[10,22,31,50]),

]

###############################################################################

if __name__ == "__main__":

    for instance in instances:
        for week in instance["weeks"]:
            instance_name = instance["abbrv"]+'_W'+str(week)

            if instance["source"] == init_state.csv_reader:
                init_state.create_and_save_state(name=instance_name, 
                                                 city=instance["city"],
                                                 filename=INSTANCE_DIRECTORY + "/" + instance_name,
                                                 source=instance["source"],
                                                 urlHistorical=instance["urlHistorical"],
                                                 urlGbfs=instance["urlGbfs"],
                                                 filename_format=instance["filename_format"],
                                                 number_of_bikes=instance["numbikes"],
                                                 number_of_stations=instance["numstations"],
                                                 week=week,
                                                 mapdata=instance["mapdata"],
                                                 )

            else:
                init_state.create_and_save_state(name=instance_name,
                                                 city=instance["city"],
                                                 filename=INSTANCE_DIRECTORY + "/" + instance_name,
                                                 source=instance["source"],
                                                 urlHistorical=instance["urlHistorical"],
                                                 urlGbfs=instance["urlGbfs"],
                                                 number_of_bikes=instance["numbikes"],
                                                 number_of_stations=instance["numstations"],
                                                 week=week,
                                                 mapdata=instance["mapdata"],
                                                 )
