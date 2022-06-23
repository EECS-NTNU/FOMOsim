# download.py

import os
import requests

def oslo(fromIncluded, toIncluded):
    for i in range(int(fromIncluded), int(toIncluded) + 1):
        monthNo = 1 + ((i + 2) % 12) # 1 is April
        yearNo = (i + 2)//12 + 2019
        if monthNo < 10:  
            monthStr = "0" + str(monthNo)
        else:
            monthStr = str(monthNo)
        address = "https://data.urbansharing.com/oslobysykkel.no/trips/v1/" + str(yearNo) + "/" + monthStr + ".json"
        print(address, "...", end = '')
        data = requests.get(address)
        if not os.path.isdir("init_state/cityBike/data/Oslo/tripData"):
            os.mkdir("init_state/cityBike/data/Oslo/tripData")
        dataOut = open("init_state/cityBike/data/Oslo/tripData/Oslo-" + str(yearNo) + "-" + monthStr  + ".json", "w")
        dataOut.write(data.text)
        dataOut.close()
        print(" downloaded")