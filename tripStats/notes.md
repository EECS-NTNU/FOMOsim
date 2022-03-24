# City-bike trip statistics

## File structure

* package folder name **tripStats**
* subfolder **data** contains trip data and more for different cities
  * **data/Oslo** from Oslo City Bike Data
  * **data/Utopia** contains various syntetic data for testing

## Code

* **parse.py**
  *  **calcDistance**(_city_) reads all files in subfolder data/_city_
    * Assumes all files are .json files in the correct format
    * Finds all stations used as start or end of a trip
    * Writes stations overview in file _stations.txt_

* note start and end might be different hour, weekday and weekNo

* distance matrix
  * rounded up or down to nearest km
  * calcDistances() is time consuming since it reads all trips, but it found 270 stations. Reading one busy month gfound typically 253-254 stations

* Oslo City Bike data stores longitude and latitude of stations, not altitude.
  * A distancse matrix should be symmetrical if distance is "birds flight" (luftlinje). One-way roads etc. could make it assymmetric. Difference in altitude will give assymmetric travel times, therefore average travel-times give indirectly altitude-information :-)

