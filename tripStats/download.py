# download.py

import requests


def strip(removeStr, wholeStr):
    start = wholeStr.find(removeStr)
    if start == 0:
        return wholeStr[len(removeStr):len(wholeStr)]
    else:
        print("*** ERROR could not remove ", removeStr)

def oslo(fromIncluded, toIncluded):
#    print("Oslo called with parameters", fromIncluded, toIncluded) 
    first = int(strip("From:", fromIncluded))
    last = int(strip("To:", toIncluded))
    for i in range(first, last + 1):
        monthNo = 1 + ((i + 2) % 12) # 1 is April
        yearNo = (i + 2)//12 + 2019
        # print(yearNo, monthNo)
        if monthNo < 10:
            monthStr = "0" + str(monthNo)
        else:
            monthStr = str(monthNo)
        address = "https://data.urbansharing.com/oslobysykkel.no/trips/v1/" + str(yearNo) + "/" + monthStr + ".json"
        print(address, "...", end = '')
        data = requests.get(address)
        dataFileName = "tripData-" + str(i)    
        dataOut = open("tripStats/data/Oslo/" + dataFileName + ".json", "w")
        dataOut.write(data.text)
        dataOut.close()
        print(" downloaded")
    print("Finito")