# download.py

import requests

from init_state.cityBike.helpers import strip 

def oslo(fromIncluded, toIncluded):
    first = int(strip("From:", fromIncluded))
    last = int(strip("To:", toIncluded))
    for i in range(first, last + 1):
        monthNo = 1 + ((i + 2) % 12) # 1 is April
        yearNo = (i + 2)//12 + 2019
        if monthNo < 10:  
            monthStr = "0" + str(monthNo)
        else:
            monthStr = str(monthNo)
        address = "https://data.urbansharing.com/oslobysykkel.no/trips/v1/" + str(yearNo) + "/" + monthStr + ".json"
        print(address, "...", end = '')
        data = requests.get(address)
        dataFileName = "Oslo-" + str(i)    
        dataOut = open("init_state.cityBike/data/Oslo/tripData/Oslo-" + str(yearNo) + "-" + monthStr  + ".json", "w")
        dataOut.write(data.text)
        dataOut.close()
        print(" downloaded")
    print("Finito")