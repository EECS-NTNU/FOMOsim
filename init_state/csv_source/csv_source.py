import requests
import os
import zipfile
import io
import os.path
from pathlib import Path
from datetime import datetime
import csv
import json
import gzip
from progress.bar import Bar

import init_state

def save_json(csvf, filename):
    with gzip.open(filename, "wb") as file:
        file.write("[\n".encode())
        csvReader = csv.DictReader(csvf)
        first = True
        for row in csvReader:
            if first: first = False
            else: file.write(",\n".encode())

            trip = {}

            if "Start Time" in row: trip["started_at"] = row["Start Time"]
            if "Stop Time" in row: trip["ended_at"] = row["Stop Time"]
            if "Start Station ID" in row: trip["start_station_id"] = row["Start Station ID"]
            if "End Station ID" in row: trip["end_station_id"] = row["End Station ID"]

            if "starttime" in row: trip["started_at"] = row["starttime"]
            if "stoptime" in row: trip["ended_at"] = row["stoptime"]
            if "start station id" in row: trip["start_station_id"] = row["start station id"]
            if "end station id" in row: trip["end_station_id"] = row["end station id"]

            if "started_at" in row: trip["started_at"] = row["started_at"]
            if "ended_at" in row: trip["ended_at"] = row["ended_at"]
            if "start_station_id" in row: trip["start_station_id"] = row["start_station_id"]
            if "end_station_id" in row: trip["end_station_id"] = row["end_station_id"]

            # if ("Fecha_Retiro" in row) and ("Hora_Retiro" in row):
            #     date = datetime.strptime(row["Fecha_Retiro"] + " " + row["Hora_Retiro"], "%d/%m/%Y %H:%M:%S")
            #     trip["started_at"] = date.strftime("%Y-%m-%d %H:%M:%S")
            # if ("Fecha_Arribo" in row) and ("Hora_Arribo" in row):
            #     date = datetime.strptime(row["Fecha_Arribo"] + " " + row["Hora_Arribo"], "%d/%m/%Y %H:%M:%S")
            #     trip["ended_at"] = date.strftime("%Y-%m-%d %H:%M:%S")
            # if "Ciclo_Estacion_Retiro" in row: trip["start_station_id"] = row["Ciclo_Estacion_Retiro"]
            # if "Ciclo_Estacion_Arribo" in row: trip["end_station_id"] = row["Ciclo_Estacion_Arribo"]

            if(("started_at" not in trip) or
               ("ended_at" not in trip) or
               ("start_station_id" not in trip) or
               ("end_station_id" not in trip)):
                print("Not enough data in trip.  Row:")
                print(row)
                raise Exception("Not enough data")

            file.write(json.dumps(trip, indent=4).encode())

        file.write("]\n".encode())

def download(week, url, city, filename_format, YMpairs, directory):
    progress = Bar("Download datafiles            ", max = len(YMpairs))

    for year, month in YMpairs:
        if month in init_state.weekMonths(week):
            filename = f"{directory}/{year}-{month:02}.json.gz"
            if not os.path.exists(filename):
                address = url + filename_format.replace('%Y', f"{year}").replace('%m', f"{month:02}")
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
                    if filename_format.endswith("csv"):
                        csvf = io.StringIO(data.content.decode())
                        save_json(csvf, filename)
        progress.next()
    progress.finish()

def get_initial_state(city, urlHistorical, urlGbfs, week, filename_format, fromInclude=[2017, 1], toInclude=[2022,11], trafficMultiplier=1.0):

    tripDataPath = f"{init_state.tripDataDirectory}{city}"
    if not os.path.isdir(tripDataPath):
        os.makedirs(tripDataPath, exist_ok=True)

    YMpairs = init_state.generateYMpairs(fromInclude, toInclude)

    download(week, urlHistorical, city, filename_format, YMpairs, tripDataPath)
    init_state.downloadStationInfo(urlGbfs, tripDataPath)

    return init_state.tripAnalysis(tripDataPath, YMpairs, week, trafficMultiplier)
