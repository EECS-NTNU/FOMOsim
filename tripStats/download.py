# download.py

import requests
from helpers import strip 

def oslo(fromIncluded, toIncluded):
#    print("Oslo called with parameters", fromIncluded, toIncluded) 
    first = int(strip("From:", fromIncluded))
    last = int(strip("To:", toIncluded))
    for i in range(first, last + 1):
        monthNo = 1 + ((i + 2) % 12) # 1 is April
        yearNo = (i + 2)//12 + 2019
        # print(yearNo, monthNo)
        if monthNo < 10: # refactor into function zeroPad
            monthStr = "0" + str(monthNo)
        else:
            monthStr = str(monthNo)
        address = "https://data.urbansharing.com/oslobysykkel.no/trips/v1/" + str(yearNo) + "/" + monthStr + ".json"
        print(address, "...", end = '')
        data = requests.get(address)
        dataFileName = "Oslo-" + str(i)    
        dataOut = open("tripStats/data/Oslo/tripData/Oslo-" + str(yearNo) + "-" + monthStr  + ".json", "w")
        dataOut.write(data.text)
        dataOut.close()
        print(" downloaded")
    print("Finito")