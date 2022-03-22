# City-bike trip statistics

* foldername tripStats

* distance matrix
  * rounded up or down to nearest km
  * calcDistances() is time consuming since it reads all trips, but it found 270 stations. Reading one busy month gfound typically 253-254 stations

* Oslo City Bike data stores longitude and latitude of stations, not altitude.
  * A distancse matrix should be symmetrical if distance is "birds flight" (luftlinje). One-way roads etc. could make it assymmetric. Difference in altitude will give assymmetric travel times, therefore average travel-times give indirectly altitude-information :-)

