# City-bike trip statistics

## Til Asbjørn

* Kode-eksemplet du la ut for meg i main.py var svært nyttig!!
* Initiell stasjons-status i antall sykler er "hardkodet" til 23 mars 2022 kl. 1513, se **readBikeStartStatus()** i parse.py
* funksjonen oslo() i download.py brukes for å laste ned JSON filer for Oslo, de som finnes er nummerert 1 - 35
* jeg har fjernet (men tatt vare på :-) dashboard-modulen (GUI), for å levere bare den koden du ba om

## File structure

* package folder name **init_state.cityBike**
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
    * Reads out all trip data for a given city and week no, counts arriving bikes for all end-stations and leaving bikes and destinations for trips initiated at all start stations. Broken down into weekday and hour. Also records durations of all trips for every startstation/endstation-pair _starting_ in the given week
    * Calculated Distance to ensure up to data data, by calling calcDistance
    * Calculates average speed matrix for trips in that week using average duration and distance
    * Calculates arrive and leave intensities, and move probabilities
* **Status fra testing in main.py**
  * jeg må kommentere ut speed_matrix i kallet fra main da min branch har eldre versjon av State
  * kræsjet der move_probabilites skal plugges inn, men fikk det i orden ved å kopiere over til numpy-array
    * TODO, bør kanskje bruke numpy fra starten av !?

## Notes

* note that start and end might be in different hour, weekday, or even weekNo
* distance matrix
  * calcDistances() is time consuming since it reads all trips
* Oslo City Bike data stores longitude and latitude of stations, not altitude.
  * A distance matrix should be symmetrical if distance is "birds flight" (luftlinje). One-way roads etc. could make it assymmetric. Difference in altitude will give assymmetric travel times, therefore average travel-times give indirectly altitude-information :-)
