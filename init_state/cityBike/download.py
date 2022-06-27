# download.py

import os
import requests
import re
from init_state.cityBike.parse import tripDataDirectory

def extractCityFromURL(url):
    name = re.sub("https://data.urbansharing.com/","",url)
    name = re.sub(".no/trips/v1/","", name)
    name = re.sub(".com/trips/v1/","", name)
    return name

def download(url):
    city = extractCityFromURL(url)
    directory = tripDataDirectory + "/" + city
    if not os.path.isdir(directory):
        os.mkdir(directory) 
    file_list = os.listdir(directory)

    yearNo = 2018 # 2018/02 is earliest data from data.urbansharing.com
    while yearNo < 2024: # TODO nice-to-have improve stop iteration by reading current year
        for monthNo in range (12): # these loops are a brute-force method to avoid implementing a web-crawler
            monthNo += 1    
            if monthNo < 10:  
                monthStr = "0" + str(monthNo)
            else:
                monthStr = str(monthNo)
            fileName = str(yearNo) + "-" + monthStr + ".json"
            if fileName not in file_list:     
                address = url + str(yearNo) + "/" + monthStr + ".json"
                data = requests.get(address)
                if data.status_code == 200: # non-existent files will have status 404
                    print("downloads " + city + " " + fileName + " ...")
                    dataOut = open(directory + "/" + str(yearNo) + "-" + monthStr  + ".json", "w")
                    dataOut.write(data.text)
                    dataOut.close()
        yearNo += 1    

