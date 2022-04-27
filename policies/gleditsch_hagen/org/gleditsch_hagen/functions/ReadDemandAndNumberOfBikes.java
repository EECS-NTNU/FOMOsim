package org.gleditsch_hagen.functions;

import org.gleditsch_hagen.classes.Station;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class ReadDemandAndNumberOfBikes {


    public static HashMap<Integer, Station> readStationInformation(ArrayList<Integer> stationIdList, String demandFile, String initialStationFile) throws FileNotFoundException {
        HashMap<Integer, Station> stations = new HashMap<>();

        //Read demand.txt file
        File inputFile = new File(demandFile);
        Scanner in = new Scanner(inputFile);
        int previousValueRead = 0;
        HashMap<Integer, Integer> stationInitialNUmberOfBikesMap = readNumberOfBikes(initialStationFile);
        Station station = new Station(-1, -1);

        while (in.hasNextLine()){
            String line = in.nextLine();
            Scanner element = new Scanner(line).useDelimiter("\\s*;\\s*");
            if (element.hasNextInt()) {
                int stationId = element.nextInt();

                if (stationIdList.contains(stationId)) {
                    if (previousValueRead != stationId) {

                        previousValueRead = stationId;
                        station = new Station(stationId, stationInitialNUmberOfBikesMap.get(stationId));
                        stations.put(station.getId(), station);
                    }

                    station.setPlace(element.next());                                   //title
                    station.setCapacity((int) Double.parseDouble(element.next()));      //nr of slots
                    double hour = element.nextDouble();                                 //hour

                    //Read mean and standard deviation for arrival and departure
                    station.setBikeWantedMedian(hour, Double.parseDouble(element.next()));
                    station.setBikeWantedStd(hour, Double.parseDouble(element.next()));
                    station.setBikeReturnedMedian(hour, Double.parseDouble(element.next()));
                    station.setBikeReturnedStd(hour, Double.parseDouble(element.next()));
                    station.setOptimalState(hour, Double.parseDouble(element.next()));
                }
            }
            element.close();
        }
        in.close();
        return stations;

    }

    // Reads inputfile stationInitial and returns a hashmap with station id vs. initial number of bikes.
    public static HashMap<Integer, Integer> readNumberOfBikes(String initialStationFile) throws FileNotFoundException {
        HashMap<Integer, Integer> stationToNumberOfBikesMap = new HashMap<>();

        File inputFile = new File(initialStationFile);
        Scanner in = new Scanner(inputFile);
        while (in.hasNextLine()){
            String line = in.nextLine();
            Scanner element = new Scanner(line).useDelimiter("\\s*,\\s*");
            if (element.hasNextInt()) {
                int stationId = element.nextInt();
                int initialLoad = element.nextInt();
                stationToNumberOfBikesMap.put(stationId, initialLoad);
            }
        }
        in.close();
        return stationToNumberOfBikesMap;
    }


    public static ArrayList<Station> readDemandInformationForGeneratingInstances(String demandFile, double hour) throws FileNotFoundException {
        ArrayList<Station> stations = new ArrayList<>();

        //Read demand.txt file
        File inputFile = new File(demandFile);
        Scanner in = new Scanner(inputFile);
        int previousValueRead = 0;
        Station station = new Station(-1, -1);

        while (in.hasNextLine()){
            String line = in.nextLine();
            Scanner element = new Scanner(line).useDelimiter("\\s*;\\s*");
            if (element.hasNextInt()) {
                int stationId = element.nextInt();

                if (previousValueRead != stationId) {

                    previousValueRead = stationId;
                    station = new Station(stationId);
                    stations.add(station);
                }

                element.next();
                station.setCapacity((int) Double.parseDouble(element.next()));
                double hourInThisLine = element.nextDouble();

                // Only interested if correct hour
                if (hourInThisLine == hour) {

                    //Set demand
                    station.setBikeWantedMedian(hour, Double.parseDouble(element.next()));
                    element.next();
                    station.setBikeReturnedMedian(hour, Double.parseDouble(element.next()));
                    element.next();

                }
            }
            element.close();
        }
        in.close();
        return stations;

    }

}
