# City-bike trip statistics

## Til Asbjørn

* Jeg har ikke helt snøring på moduler, from, import m.m., må derfor desverre (via kommentering) bruke to ulike varianter av import fra tripStats i parse.py
* Initiell stasjons-status i antall sykler er "hardkodet" til 23 mars 2022 kl. 1513, se **readBikeStartStatus()**

## File structure

* package folder name **tripStats**
* subfolder **data** contains trip data and more for different cities
  * **data/Oslo** from Oslo City Bike Data
    * subfolder **tripData** stores all dowloaded tripData
  * **data/Utopia** contains various syntetic data for testing
    * _work not started_

## Code

* **parse.py**
  * **calcDistance**(_city_) reads all files in subfolder data/_city_/tripData
    * Assumes all files are .json files in the correct format
    * Finds all stations used as start or end of a trip
    * Writes stations overview in file _stations.txt_
  * **get_initial_state**(_city, week_)
    * Reads out all trip data for a given city and week no, counts arriving bikes for all end-stations and leaving bikes and destinations for trips initiated at all start stations. Broken down into weekday and hour
    * Also records durations of all trips for every startstation/endstation-pair _starting_ in the given week Calculates average speed matrix for trips in that week in
    
    * **Status:**
      * jeg må kommentere ut speed_matrix i kallet fra main da min branch har eldre versjon av State
      * kræsjer der move_probabilites skal plugges inn, kanskje pga. gammel kode, eller for at det brukes numpy der? 

## Notes

* note start and end might be different hour, weekday and weekNo

* distance matrix
  * calcDistances() is time consuming since it reads all trips

* Oslo City Bike data stores longitude and latitude of stations, not altitude.
  * A distancse matrix should be symmetrical if distance is "birds flight" (luftlinje). One-way roads etc. could make it assymmetric. Difference in altitude will give assymmetric travel times, therefore average travel-times give indirectly altitude-information :-)