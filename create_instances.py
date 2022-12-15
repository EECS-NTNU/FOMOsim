#!/bin/python3
"""
FOMO simulator, create jobs to run on cluster
"""

import init_state
import init_state.json_source
import init_state.csv_source

from helpers import *

INSTANCE_DIRECTORY="instances"

###############################################################################

instances = [

    dict(source=init_state.csv_source,
         city="NewYork",
         abbrv="NY",
         urlHistorical="https://s3.amazonaws.com/tripdata/",
         urlGbfs="http://gbfs.citibikenyc.com/gbfs/en",
         filename_format="%Y%m-citibike-tripdata.csv.zip",
         numbikes=None,
         numstations=None,
         weeks=[31]),

    dict(source=init_state.csv_source,
         city="Boston",
         abbrv="BO",
         urlHistorical="https://s3.amazonaws.com/hubway-data/",
         urlGbfs="https://gbfs.bluebikes.com/gbfs/en",
         filename_format="%Y%m-bluebikes-tripdata.zip",
         numbikes=None,
         numstations=None,
         weeks=[31]),

    dict(source=init_state.csv_source,
         city="Chicago",
         abbrv="CH",
         urlHistorical="https://divvy-tripdata.s3.amazonaws.com/",
         urlGbfs="https://gbfs.divvybikes.com/gbfs/en",
         filename_format="%Y%m-divvy-tripdata.zip",
         numbikes=None,
         numstations=None,
         weeks=[31]),

    dict(source=init_state.json_source,
         city="Oslo",
         abbrv="OS",
         urlHistorical="https://data.urbansharing.com/oslobysykkel.no/trips/v1/",
         urlGbfs="https://gbfs.urbansharing.com/oslobysykkel.no",
         numbikes=2885,
         numstations=None,
         weeks=[10,22,31,50]),

    dict(source=init_state.json_source,
         city="Bergen",
         abbrv="BG",
         urlHistorical="https://data.urbansharing.com/bergenbysykkel.no/trips/v1/",
         urlGbfs="https://gbfs.urbansharing.com/bergenbysykkel.no",
         numbikes=1000,
         numstations=None,
         weeks=[8,25,35,45]),

    dict(source=init_state.json_source,
         city="Trondheim",
         abbrv="TD",
         urlHistorical="https://data.urbansharing.com/trondheimbysykkel.no/trips/v1/",
         urlGbfs="https://gbfs.urbansharing.com/trondheimbysykkel.no",
         numbikes=768,
         numstations=None,
         weeks=[17,21,34,44]),

    dict(source=init_state.json_source,
         city="Edinburgh",
         abbrv="EH",
         urlHistorical="https://data.urbansharing.com/edinburghcyclehire.com/trips/v1/",
         urlGbfs="https://gbfs.urbansharing.com/edinburghcyclehire.com",
         numbikes=140,
         numstations=None,
         weeks=[10,22,31,50]),

]

###############################################################################

if __name__ == "__main__":

    for instance in instances:
        for week in instance["weeks"]:
            instance_name = instance["abbrv"]+'_W'+str(week)

            print(f"\n\nGenerating instance {instance_name}\n")

            if instance["source"] == init_state.csv_source:
                init_state.create_and_save_state(name=instance_name, 
                                                 city=instance["city"],
                                                 instance_directory=INSTANCE_DIRECTORY,
                                                 source=instance["source"],
                                                 urlHistorical=instance["urlHistorical"],
                                                 urlGbfs=instance["urlGbfs"],
                                                 filename_format=instance["filename_format"],
                                                 number_of_bikes=instance["numbikes"],
                                                 number_of_stations=instance["numstations"],
                                                 week=week,
                                                 )

            else:
                init_state.create_and_save_state(name=instance_name,
                                                 city=instance["city"],
                                                 instance_directory=INSTANCE_DIRECTORY,
                                                 source=instance["source"],
                                                 urlHistorical=instance["urlHistorical"],
                                                 urlGbfs=instance["urlGbfs"],
                                                 number_of_bikes=instance["numbikes"],
                                                 number_of_stations=instance["numstations"],
                                                 week=week,
                                                 )
