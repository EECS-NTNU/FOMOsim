import requests
import os
import zipfile
import io
import os.path
from pathlib import Path
import csv
import json
from urllib.parse import urlparse

import init_state

tripDataDirectory = "init_state/csv_reader/data/"

def download(url, city, filename_format, YMpairs, directory):
    # TODO: YMpairs

    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    for year in range(2017, 2019):
        for month in range(1, 12):

            filename = f"{directory}/{year}-{month:02}.json"
            if not os.path.exists(filename):
                address = url + filename_format.replace('%Y', f"{year}").replace('%m', f"{month:02}")
                print(f"Downloading {address}");
                data = requests.get(address)
                if data.status_code == 200:
                    z = zipfile.ZipFile(io.BytesIO(data.content))
                    for zf in z.infolist():
                        if zf.filename.split(".")[-1] == "csv":
                            # unzip first csv file in zip-archive
                            csvf = io.StringIO(z.read(z.infolist()[0]).decode())

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
                                if "Trip Duration" in row: trip["duration"] = row["Trip Duration"]

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
                                if "tripduration" in row: trip["duration"] = row["tripduration"]

                                trips.append(trip)

                            # save to disk
                            dataOut = open(filename, "wb")
                            dataOut.write(json.dumps(trips).encode())
                            dataOut.close()

                            break

                    z.close()

def get_initial_state(urlHistorical, urlGbfs, week, filename_format, fromInclude=[2018, 5], toInclude=[2022,8]):
    city = urlparse(urlGbfs).hostname
    tripDataPath = f"{tripDataDirectory}{city}"

    YMpairs = init_state.generateYMpairs(fromInclude, toInclude)

    download(urlHistorical, city, filename_format, YMpairs, tripDataPath)
    init_state.downloadStationInfo(urlGbfs, tripDataPath)

    return init_state.parse_json(tripDataPath, YMpairs, week)
