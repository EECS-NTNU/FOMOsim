# download.py

def oslo(fromIncluded, toIncluded):
    print("Oslo called with parameters", fromIncluded, toIncluded)
    first = int(fromIncluded)
    last = int(toIncluded)

# data = requests.get("https://data.urbansharing.com/oslobysykkel.no/trips/v1/2022/01.json") # 15378 trips
#     dataOut = open("out.json", "w")
#     dataOut.write(data.text)
#     dataOut.close()