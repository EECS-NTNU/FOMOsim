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

cities = ["Oslo","Bergen","Trondheim","Edinburgh"]
abbrvs = {"Oslo": 'OS',
          "Bergen": 'BG',
          "Trondheim":'TD' ,
          "Edinburgh":'EH'
          }
urls = {"Oslo": "https://data.urbansharing.com/oslobysykkel.no/trips/v1/",
          "Bergen": "https://data.urbansharing.com/bergenbysykkel.no/trips/v1/",
          "Trondheim": "https://data.urbansharing.com/trondheimbysykkel.no/trips/v1/",
          "Edinburgh":"https://data.urbansharing.com/edinburghcyclehire.com/trips/v1/"
          }
weeks = {"Oslo": [10,22,31,50],
          "Bergen": [8,25,35,45],
          "Trondheim":[17,21,34,44] ,
          "Edinburgh":[10,22,31,50]
          }

numbikes = {"Oslo": 2000,
          "Bergen": 1200,
          "Trondheim":1000 ,
          "Edinburgh":200
          }

instances = []

for city in cities:
     for week in weeks[city]:
          instances.append(dict(
               city=city,
               url=urls[city],
               numbikes=numbikes[city],
               numstations=None,
               week=week
          ))

# # Enter instance definition here.  For numbikes and numstations, enter 'None' to use dataset default
# instances = [

#     dict(city="Oslo",
#          url="https://data.urbansharing.com/oslobysykkel.no/trips/v1/",
#          numbikes=2000,
#          numstations=None,
#          week=33),

#     dict(city="Bergen",
#          url="https://data.urbansharing.com/bergenbysykkel.no/trips/v1/",
#          numbikes=1200,
#          numstations=None,
#          week=33),

#     dict(city="Trondheim",
#          url="https://data.urbansharing.com/trondheimbysykkel.no/trips/v1/",
#          numbikes=1000,
#          numstations=None,
#          week=33),

#     # dict(city="Oslo-vinter",
#     #      url="https://data.urbansharing.com/oslovintersykkel.no/trips/v1/",
#     #      numbikes=400,
#     #      numstations=None,
#     #      week=7),

#     dict(city="Edinburgh",
#          url="https://data.urbansharing.com/edinburghcyclehire.com/trips/v1/",
#          numbikes=200,
#          numstations=None,
#          week=20),

# ]

###############################################################################

if __name__ == "__main__":

    for instance in instances:
        instance_name = abbrvs[instance["city"]]+'_W'+str(instance["week"])
        init_state.create_and_save_state(instance_name, 
                                         INSTANCE_DIRECTORY + "/" + instance_name,
                                         source=init_state.cityBike,
                                         url=instance["url"],
                                         number_of_bikes=instance["numbikes"],
                                         number_of_stations=instance["numstations"],
                                         week=instance["week"],
                                         )
