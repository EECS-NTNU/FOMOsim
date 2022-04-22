package org.gleditsch_hagen.main;


import org.gleditsch_hagen.classes.Input;
import org.gleditsch_hagen.classes.Station;
import java.io.*;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

import static java.lang.Math.round;
import static java.lang.StrictMath.floor;

public class CreateTestInstance {

    static ArrayList<Station> highCongestion = new ArrayList<>();
    static ArrayList<Station> lowCongestion = new ArrayList<>();
    static ArrayList<Station> highStarvation = new ArrayList<>();
    static ArrayList<Station> lowStarvation = new ArrayList<>();
    static ArrayList<Station> allStations;
    static double startHour = 17.0;
    static int instanceSize = 50;
    static HashMap<Station, Integer> testInstance = new HashMap<>();
    static String filename = "stationInitialInstance2-17.txt";

    private static void divideStations(Input input) throws FileNotFoundException {
        allStations = input.getStationListWithDemand();

        //Add stations to appropriate group
        for (Station station: allStations) {
            if (station.getNetDemand(startHour) >= 15) { //Legger til stasjon i highCongestion
                highCongestion.add(station);
            } else if(station.getNetDemand(startHour) >= 0 && station.getNetDemand(startHour) < 15){ //Legger til stasjon i low congestion
                lowCongestion.add(station);
            } else if(station.getNetDemand(startHour) > -15 && station.getNetDemand(startHour) < 0) { //Legger til stasjon i highStarvation
                lowStarvation.add(station);
            } else if(station.getNetDemand(startHour) <= -15) { //Legger til stasjon i lowStarvation
                highStarvation.add(station);
            }
        }
    }

    private  static void pickTestInstance() {
        //60% picks from high, 40% picks from low
        int picksFromHigh = (int) floor(instanceSize * 0.30);
        int picksFromLow = (int) floor(instanceSize * 0.20);

        /*
        if (picksFromHigh*2 + picksFromLow*2 != instanceSize) {
            picksFromHigh = picksFromHigh + (instanceSize - (picksFromHigh+picksFromLow));
        }
        */

        Random randomNumber = new Random();
        //testInstance = new HashMap<>();

        //Add stations from High Congested stations
        if (picksFromHigh >= highCongestion.size()) {
            for (Station station : highCongestion) {
                testInstance.put(station, randomNumber.nextInt(station.getCapacity() + 1));
            }
        } else {
            for (int i = 0; i < picksFromHigh; i++) {
                Station station = highCongestion.get(randomNumber.nextInt(highCongestion.size()));
                testInstance.put(station, randomNumber.nextInt(station.getCapacity()+1));
                highCongestion.remove(station);
            }
        }

        //Add stations from High Starved stations
        if (picksFromHigh >= highStarvation.size()) {
            for (Station station : highStarvation) {
                testInstance.put(station, randomNumber.nextInt(station.getCapacity() + 1));
            }
        } else {
            for (int i = 0; i < picksFromHigh; i++) {
                Station station = highStarvation.get(randomNumber.nextInt(highStarvation.size()));
                testInstance.put(station, randomNumber.nextInt(station.getCapacity()+1));
                highStarvation.remove(station);
            }
        }

        //Add stations from Low Congested stations
        if (picksFromLow >= lowCongestion.size()) {
            for (Station station : lowCongestion) {
                testInstance.put(station, randomNumber.nextInt(station.getCapacity() + 1));
            }
        } else {
            for (int i = 0; i < picksFromLow; i++) {
                Station station = lowCongestion.get(randomNumber.nextInt(lowCongestion.size()));
                testInstance.put(station, randomNumber.nextInt(station.getCapacity()+1));
                lowCongestion.remove(station);
            }
        }

        //Add stations from Low Starved stations
        if (picksFromLow >= lowStarvation.size()) {
            for (Station station : lowStarvation) {
                testInstance.put(station, randomNumber.nextInt(station.getCapacity() + 1));
            }
        } else {
            for (int i = 0; i < picksFromLow; i++) {
                Station station = lowStarvation.get(randomNumber.nextInt(lowStarvation.size()));
                testInstance.put(station, randomNumber.nextInt(station.getCapacity()+1));
                lowStarvation.remove(station);
            }
        }

        //Add more stations if test instance is too  small
        if (testInstance.size() < instanceSize) {
            int picksLeft = instanceSize - testInstance.size();
            for(int i = 0; i < picksLeft; i++) {
                Station newStation = allStations.get(randomNumber.nextInt(allStations.size()));
                while (testInstance.containsKey(newStation)) {
                    newStation = allStations.get(randomNumber.nextInt(allStations.size()));
                }
                testInstance.put(newStation, randomNumber.nextInt(newStation.getCapacity()+1));
            }
        }
    }

    private static void writeTextFile(String filename) throws IOException {
        PrintWriter writer = new PrintWriter(filename, "UTF-8");
        for (Station station: testInstance.keySet()) {
            writer.write(station.getId() + ", " + testInstance.get(station));
            writer.println();
        }
        writer.close();
    }



    public static void main(String[] args) throws IOException {
        Input input = new Input(startHour);
        divideStations(input);
        pickTestInstance();
        writeTextFile(filename);
        System.out.println("Test instance successfully created.");
    }



}
