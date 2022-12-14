import requests
import os
import zipfile
import io
import os.path
from pathlib import Path
import csv
import json

import init_state

tripDataDirectory = "init_state/data/"

def save_json(csvf, filename):
    csvReader = csv.DictReader(csvf)
    trips = []
    for row in csvReader:
        trip = {}

        if "Start Time" in row: trip["started_at"] = row["Start Time"]
        if "Stop Time" in row: trip["ended_at"] = row["Stop Time"]
        if "Start Station ID" in row: trip["start_station_id"] = row["Start Station ID"]
        if "Start Station Name" in row: trip["start_station_name"] = row["Start Station Name"]
        if "Start Station Latitude" in row: trip["start_station_latitude"] = row["Start Station Latitude"]
        if "Start Station Longitude" in row: trip["start_station_longitude"] = row["Start Station Longitude"]
        if "End Station ID" in row: trip["end_station_id"] = row["End Station ID"]
        if "End Station Name" in row: trip["end_station_name"] = row["End Station Name"]
        if "End Station Latitude" in row: trip["end_station_latitude"] = row["End Station Latitude"]
        if "End Station Longitude" in row: trip["end_station_longitude"] = row["End Station Longitude"]

        if "starttime" in row: trip["started_at"] = row["starttime"]
        if "stoptime" in row: trip["ended_at"] = row["stoptime"]
        if "start station id" in row: trip["start_station_id"] = row["start station id"]
        if "start station name" in row: trip["start_station_name"] = row["start station name"]
        if "start station latitude" in row: trip["start_station_latitude"] = row["start station latitude"]
        if "start station longitude" in row: trip["start_station_longitude"] = row["start station longitude"]
        if "end station id" in row: trip["end_station_id"] = row["end station id"]
        if "end station name" in row: trip["end_station_name"] = row["end station name"]
        if "end station latitude" in row: trip["end_station_latitude"] = row["end station latitude"]
        if "end station longitude" in row: trip["end_station_longitude"] = row["end station longitude"]

        if "started_at" in row: trip["started_at"] = row["started_at"]
        if "ended_at" in row: trip["ended_at"] = row["ended_at"]
        if "start_station_id" in row: trip["start_station_id"] = row["start_station_id"]
        if "start_station_name" in row: trip["start_station_name"] = row["start_station_name"]
        if "start_lat" in row: trip["start_station_latitude"] = row["start_lat"]
        if "start_lng" in row: trip["start_station_longitude"] = row["start_lng"]
        if "end_station_id" in row: trip["end_station_id"] = row["end_station_id"]
        if "end_station_name" in row: trip["end_station_name"] = row["end_station_name"]
        if "end_lat" in row: trip["end_station_latitude"] = row["end_lat"]
        if "end_lng" in row: trip["end_station_longitude"] = row["end_lng"]

        assert("started_at" in trip)
        assert("ended_at" in trip)
        assert("start_station_id" in trip)
        assert("end_station_id" in trip)

        trips.append(trip)

    # save to disk
    dataOut = open(filename, "wb")
    dataOut.write(json.dumps(trips).encode())
    dataOut.close()

def download(url, city, filename_format, YMpairs, directory):
    for year, month in YMpairs:
        filename = f"{directory}/{year}-{month:02}.json"
        if not os.path.exists(filename):
            address = url + filename_format.replace('%Y', f"{year}").replace('%m', f"{month:02}")
            print(f"Downloading {address}");
            data = requests.get(address)
            if data.status_code == 200:
                if filename_format.endswith("zip"):
                    z = zipfile.ZipFile(io.BytesIO(data.content))
                    for zf in z.infolist():
                        if zf.filename.endswith("csv"):
                            # unzip first csv file in zip-archive
                            csvf = io.StringIO(z.read(z.infolist()[0]).decode())
                            save_json(csvf, filename)
                            break
                    z.close()

def get_initial_state(city, urlHistorical, urlGbfs, week, filename_format, fromInclude=[2017, 1], toInclude=[2022,11], trafficMultiplier=1.0):

    tripDataPath = f"{tripDataDirectory}{city}"
    if not os.path.isdir(tripDataPath):
        os.makedirs(tripDataPath, exist_ok=True)

    YMpairs = init_state.generateYMpairs(fromInclude, toInclude)

    download(urlHistorical, city, filename_format, YMpairs, tripDataPath)
    init_state.downloadStationInfo(urlGbfs, tripDataPath)

    return init_state.parse_json(tripDataPath, YMpairs, week, trafficMultiplier)
