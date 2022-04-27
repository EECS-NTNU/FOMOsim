package org.gleditsch_hagen.functions;

import org.gleditsch_hagen.classes.Station;
import org.gleditsch_hagen.classes.Vehicle;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.Scanner;

public class ReadVehicleInput {

    public static HashMap<Integer, Vehicle> readVehicleInput(String vehicleInitialFile) throws FileNotFoundException {

        HashMap<Integer, Vehicle> vehicles = new HashMap<>();

        //Read vehicleInitial2.txt file
        File inputFile = new File(vehicleInitialFile);
        Scanner in = new Scanner(inputFile);
        while (in.hasNextLine()){
            String line = in.nextLine();
            Scanner element = new Scanner(line);
            if (element.hasNextInt()) {
                int id = element.nextInt();
                Vehicle vehicle = new Vehicle(id);
                int nextStationInitial = element.nextInt();
                vehicle.setNextStationInitial(nextStationInitial);
                double timeToNextStationInitial = (Double.parseDouble(element.next()));
                vehicle.setTimeToNextStationInitial(timeToNextStationInitial);
                int load = element.nextInt();
                vehicle.setInitialLoad(load);
                int capacity = element.nextInt();
                vehicle.setCapacity(capacity);

                vehicles.put(id, vehicle);
            }
            element.close();
        }
        in.close();
        return vehicles;
    }
}
