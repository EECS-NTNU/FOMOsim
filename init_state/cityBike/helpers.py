import re

def extractCityFromURL(url):
    name = re.sub("https://data.urbansharing.com/","",url)
    name = re.sub(".no/trips/v1/","", name)
    name = re.sub(".com/trips/v1/","", name)
    return name
    
def extractCityAndDomainFromURL(url):
    name = re.sub("https://data.urbansharing.com/","",url)
    name = re.sub("/trips/v1/","", name)
    return name
