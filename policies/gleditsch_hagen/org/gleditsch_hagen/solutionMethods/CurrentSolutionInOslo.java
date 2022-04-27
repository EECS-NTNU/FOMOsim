package org.gleditsch_hagen.solutionMethods;

import org.gleditsch_hagen.classes.*;
import org.gleditsch_hagen.functions.ReadClusterList;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;

public class CurrentSolutionInOslo {

    private ArrayList<VehicleArrival> vehicleArrivals = new ArrayList<>();

    //Constructor
    public CurrentSolutionInOslo(Input input) throws IOException {

        generateVehicleArrivals(input);

    }

    private void generateVehicleArrivals(Input input) throws IOException {


        for (Vehicle vehicle : input.getVehicles().values()) {

            ArrayList<StationVisit> stationVisits = vehicle.createRoutesCurrentSolution(input);
            int numberOfStationVisitsInRoute = stationVisits.size();

            for (int i = 0; i < numberOfStationVisitsInRoute; i++) {

                int stationId = stationVisits.get(i).getStation().getId();
                int stationLoad = (int) stationVisits.get(i).getLoadingQuantity();
                double time = stationVisits.get(i).getVisitTime() + input.getCurrentMinute();
                int vehicleId = vehicle.getId();
                int nextStationId;
                double timeNextVisit;

                boolean lastStationVisit = i == numberOfStationVisitsInRoute-1;
                if (!lastStationVisit) {
                    nextStationId = stationVisits.get(i+1).getStation().getId();
                    timeNextVisit = stationVisits.get(i+1).getVisitTime() + input.getCurrentMinute();
                } else {
                    nextStationId = 0;
                    timeNextVisit = 0 + input.getCurrentMinute();
                }
                boolean firstVisit = i==0;

                VehicleArrival vehicleArrival = new VehicleArrival(stationId, stationLoad, time, vehicleId, nextStationId, timeNextVisit, firstVisit);
                vehicleArrivals.add(vehicleArrival);
            }


        }

        //Sort list by arrival time
        Collections.sort(vehicleArrivals, new Comparator<VehicleArrival>() {
            @Override
            public int compare(VehicleArrival vehicleArrival1, VehicleArrival vehicleArrival2) {
                double diff = vehicleArrival1.getTime() - vehicleArrival2.getTime();

                if( diff < 0 ){
                    return -1;
                } else if ( diff > 0 ) {
                    return 1;
                }
                return 0;
            }
        });

    }



    public ArrayList<VehicleArrival> getVehicleArrivals() {
        return vehicleArrivals;
    }

}
