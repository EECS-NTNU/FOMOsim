package org.gleditsch_hagen.classes;

public class StationVisit {

    private Station station;
    private double loadingQuantity;
    private double visitTime;
    private double loadAfterVisit;


    //Constructor

    public StationVisit() {
    }

    public StationVisit(StationVisit stationVisit) {
        station = stationVisit.getStation();
    }

    //GETTERS AND SETTERS

    public double getLoadingQuantity() {
        return loadingQuantity;
    }

    public void setLoadingQuantity(double loadingQuantity) {
        this.loadingQuantity = loadingQuantity;
    }

    public double getVisitTime() {
        return visitTime;
    }

    public void setVisitTime(double visitTime) {
        this.visitTime = visitTime;
    }

    public Station getStation() {
        return station;
    }

    public void setStation(Station station) {
        this.station = station;
    }

    public double getLoadAfterVisit() {
        return loadAfterVisit;
    }

    public void setLoadAfterVisit(double loadAfterVisit) {
        this.loadAfterVisit = loadAfterVisit;
    }
}
