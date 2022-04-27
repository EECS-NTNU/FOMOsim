package org.gleditsch_hagen.functions;

import org.gleditsch_hagen.classes.Station;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class ReadCoordinates {

    public static void lookUpCoordinates(HashMap<Integer, Station> stations, ArrayList<Integer> stationIdList) throws FileNotFoundException {
        File inputFile = new File("stationCoordinates.txt");
        Scanner in = new Scanner(inputFile);
        while (in.hasNextLine()){
            String line = in.nextLine();
            Scanner element = new Scanner(line);
            if (element.hasNextInt()) {
                int stationId = element.nextInt();
                if (stationIdList.contains(stationId)) {
                    double latitude =  Double.parseDouble(element.next());
                    double longitude = Double.parseDouble(element.next());
                    stations.get(stationId).setLatitude(latitude);
                    stations.get(stationId).setLongitude(longitude);
                }
            }
        }
        in.close();
    }

}
