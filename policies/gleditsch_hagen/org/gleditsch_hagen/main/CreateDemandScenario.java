package org.gleditsch_hagen.main;

import org.gleditsch_hagen.classes.Input;
import org.gleditsch_hagen.classes.Station;
import org.gleditsch_hagen.functions.RandomDraws;
import org.json.JSONException;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;


public class CreateDemandScenario {

    private ArrayList<Integer> stationIdList;
    private HashMap<Integer, Station> stations;
    private ArrayList<ArrayList<Double>> arrivalTimes;
    private String outPutFileName;
    private int testInstance = 3;
    private double startTime = 7;
    private double stopTime = 11;


    public static void main(String[] args) throws IOException, JSONException {
        CreateDemandScenario simulation = new CreateDemandScenario();                   //Read input data
        simulation.CreateScenarios(simulation);
    }

    private void CreateScenarios(CreateDemandScenario simulation) throws FileNotFoundException, UnsupportedEncodingException {

        for (int i = 1; i <= 10; i ++) {
            outPutFileName = "simulation_Instance"+ testInstance + "_Nr" + i + ".txt";
            simulation.run(this.startTime, this.stopTime);
            simulation.printArrivalTimes();

        }

    }


    //Constructor
    private CreateDemandScenario() throws IOException, JSONException {
        Input input = new Input(this.testInstance, this.startTime);
        stationIdList = input.getStationIdList();
        stations = input.getStations();
        arrivalTimes = new ArrayList<>();
    }

    private void run(double startTime, double stopTime){
        for (Station station : stations.values()) {
            simulate(station, startTime, stopTime);
        }

        //Sort list by arrival time
        Collections.sort(this.arrivalTimes, new Comparator<ArrayList<Double>>() {
            @Override
            public int compare(ArrayList<Double> arrival1, ArrayList<Double> arrival2) {
                double diff = arrival1.get(0) - (arrival2.get(0));

                if( diff < 0 ){
                    return -1;
                } else if ( diff > 0 ) {
                    return 1;
                }
                return 0;
            }
        });
    }

    private void simulate(Station station, double startTime, double stopTime){

        double currentTime = startTime;

        while (currentTime < stopTime) {
            double bikeWantedMedian = station.getBikeWantedMedian(currentTime);
            double bikeWantedStandardDeviation = station.getBikeWantedStd(currentTime);
            int numberOfBikesWanted = (int) RandomDraws.drawNormal(bikeWantedMedian, bikeWantedStandardDeviation);
            drawArrivals(station.getId(),-1.00, numberOfBikesWanted, currentTime);
            currentTime++;
        }

        currentTime = startTime;

        while (currentTime < stopTime) {
            double bikeReturnedMedian = station.getBikeReturnedMedian(currentTime);
            double bikeReturnedSd = station.getBikeReturnedStd(currentTime);
            int numberOfBikesReturned = (int) RandomDraws.drawNormal(bikeReturnedMedian, bikeReturnedSd);
            drawArrivals(station.getId(),1.00, numberOfBikesReturned, currentTime);
            currentTime++;
        }
    }

    //This method draws arrivals and saves input in arrivalTimes
    private void drawArrivals(int stationId, double stationLoad, int numberOfdraws, double currentTime) {
        for(int i=0; i<numberOfdraws; i++) {
            ArrayList<Double> arrivalTimeEntry = new ArrayList<>();
            arrivalTimeEntry.add(RandomDraws.drawArrivalTimes(currentTime));    //Arrival time in second, real time
            arrivalTimeEntry.add((double) stationId);                           //Station ID
            arrivalTimeEntry.add(stationLoad);                                  //-1 if wanted, 1 if returned
            arrivalTimes.add(arrivalTimeEntry);
        }
    }

    private void printArrivalTimes() throws FileNotFoundException, UnsupportedEncodingException {
        String filename = outPutFileName;
        PrintWriter writer = new PrintWriter(filename, "UTF-8");
        for (ArrayList<Double> arrivalTime: arrivalTimes) {
            writer.println(arrivalTime.get(0) + ", " + arrivalTime.get(1).intValue() + ", " + arrivalTime.get(2));
        }
        writer.close();
    }


}
