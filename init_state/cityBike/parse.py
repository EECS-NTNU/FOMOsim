# parse.py

from xml.dom.expatbuilder import parseString
import sim
import json
import os
import os.path
import requests
import geopy.distance
import datetime
from datetime import date
import numpy as np
import statistics
from statistics import fmean, stdev 
from progress.bar import Bar

import settings
import init_state
from helpers import yearWeekNoAndDay

tripDataDirectory = "init_state/data/" # location of tripData

def download(url, YMpairs, tripDataPath):    

    def loadMonth(yearNo, monthNo, alreadyLoadedFiles):
        fileName = f"{yearNo}-{monthNo:02}.json"
        if fileName in alreadyLoadedFiles:
            if yearNo == datetime.date.today().year  and monthNo == datetime.date.today().month:
                print("   Warning: We found locally stored trip-data for the current month, it will be used, BUT the file should be deleted since it is incomplete")
                return True
        else:
            if yearNo == datetime.date.today().year  and monthNo == datetime.date.today().month:
                print("   Info: will NOT load the current month, only trip-data for months that are expired") 
                return False

        if fileName not in alreadyLoadedFiles:
            # must try to load file         
            address = f"{url}{yearNo}/{monthNo:02}.json"
            data = requests.get(address)
            if data.status_code == 200: # non-existent files will have status 404
                # print(f"downloads {city} {fileName} ...") # debug
                dataOut = open(f"{tripDataPath}/{fileName}", "w")
                dataOut.write(data.text)
                dataOut.close()
                return True
            else:
                return False
        else:
            return True    

    file_list = os.listdir(tripDataPath)

    progress = Bar("Download datafiles   ", max = len(YMpairs))
    notFoundYMpairs = []
    for p in YMpairs:
        if not loadMonth(p[0], p[1], file_list):
            notFoundYMpairs.append(p)
        progress.next()
    if len(notFoundYMpairs) > 0:
        print("   Warning: Could not load tripdata from " + url + " for these year/month pairs:", end="") 
        for p in notFoundYMpairs:
            print(" " + str(p[0]) + "/" + str(p[1]), end="")
        print()
    progress.finish()

def get_initial_state(city, urlHistorical, urlGbfs, week, fromInclude=[2018, 5], toInclude=[2022,8], trafficMultiplier=1.0):

    """ Processes selected  trips downloaded for the city, calculates average trip duration for every pair of stations, including
        back-to-start trips. For pairs of stations without any registered trips an average duration is estimated via
        the trip distance and a global average BIKE_SPEED value from settings.py. This gives the travel_time matrix.
        Travel time for the vehicle is based on distance. All selected tripdata is read and used to calculate arrive and leave intensities 
        for every station and move probabilities for every pair of stations. These structures are indexed by station, week and hour.
    """

    tripDataPath = tripDataDirectory + city
    if not os.path.isdir(tripDataPath):
        os.makedirs(tripDataPath, exist_ok=True)

    YMpairs = init_state.generateYMpairs(fromInclude, toInclude)

    download(urlHistorical, YMpairs, tripDataPath) 
    init_state.downloadStationInfo(urlGbfs, tripDataPath)

    return init_state.parse_json(tripDataPath, YMpairs, week, trafficMultiplier)
