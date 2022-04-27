package org.gleditsch_hagen.xpress;

import org.gleditsch_hagen.classes.StationVisit;
import org.gleditsch_hagen.classes.Vehicle;
import org.gleditsch_hagen.classes.VehicleArrival;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;


public class ReadXpressResult {

    public static ArrayList<VehicleArrival> readVehicleArrivals(double currentTime) throws FileNotFoundException {

        ArrayList<VehicleArrival> vehicleArrivals = new ArrayList<>();

        File inputFile = new File("outputRoutesXpress.txt");
        Scanner in = new Scanner(inputFile);
        while (in.hasNextLine()) {
            String line = in.nextLine();
            Scanner element = new Scanner(line);
            if (element.hasNextInt()) {

                int stationId = roundToInteger(Double.parseDouble(element.next()));
                int stationLoad = roundToInteger(Double.parseDouble(element.next()));
                double time = Double.parseDouble(element.next());
                int vehicle = roundToInteger(Double.parseDouble(element.next()));
                int nextStationId = roundToInteger(Double.parseDouble(element.next()));
                double TimeNextStation = Double.parseDouble(element.next());
                boolean firstVisit = ((element.nextInt()) == 1);



                VehicleArrival vehicleArrival = new VehicleArrival(stationId, stationLoad, time + currentTime, vehicle, nextStationId, TimeNextStation+currentTime, firstVisit);

                vehicleArrivals.add(vehicleArrival);
            }
        }
        in.close();


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

        return vehicleArrivals;
    }

    private static int roundToInteger(double valueToBeRounded) {

        int rounded;
        double leftOver = valueToBeRounded % 1;
        boolean roundUp = leftOver > 0.5;

        if (roundUp) {
            rounded = (int) Math.ceil(valueToBeRounded);
        } else {
            rounded = (int) Math.floor(valueToBeRounded);
        }

        return rounded;
    }

    public static ArrayList<VehicleArrival> readVehicleArrivalsVersion3(HashMap<Integer, Vehicle> vehicles, double currentTime) throws FileNotFoundException {

        ArrayList<VehicleArrival> vehicleArrivals = new ArrayList<>();

        File inputFile = new File("outputRoutesXpress.txt");
        Scanner in = new Scanner(inputFile);
        while (in.hasNextLine()) {
            String line = in.nextLine();
            Scanner element = new Scanner(line);
            if (element.hasNextInt()) {

                int vehicleId = roundToInteger(Double.parseDouble(element.next()));
                int routeNumber = roundToInteger(Double.parseDouble(element.next()))-1;

                ArrayList<ArrayList<StationVisit>> initializedRoutes = vehicles.get(vehicleId).getInitializedRoutes();
                ArrayList<StationVisit> stationVisitsInRoute = initializedRoutes.get(routeNumber);
                int numberOfStationVisitInRoute = stationVisitsInRoute.size();

                for (int i = 0; i < numberOfStationVisitInRoute; i++ ) {

                    StationVisit stationVisit = stationVisitsInRoute.get(i);

                    int stationId = stationVisit.getStation().getId();
                    int stationLoad = (int) stationVisit.getLoadingQuantity();
                    double time = stationVisit.getVisitTime();

                    int nextStationId = 0;
                    double timeNextStation = 0;

                    //If not last station visit in route
                    if (i < numberOfStationVisitInRoute-1) {
                        nextStationId = stationVisitsInRoute.get(i+1).getStation().getId();
                        timeNextStation = stationVisitsInRoute.get(i+1).getVisitTime();
                    }

                    boolean firstVisit = (i == 0);

                    VehicleArrival vehicleArrival = new VehicleArrival(stationId, stationLoad, time + currentTime, vehicleId, nextStationId, timeNextStation + currentTime, firstVisit);

                    vehicleArrivals.add(vehicleArrival);
                }


            }
        }
        in.close();

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

        return vehicleArrivals;

    }

    public static double readObjectiveValue() throws FileNotFoundException, IllegalArgumentException {

        File inputFile = new File("outputObjectiveXpress.txt");
        Scanner in = new Scanner(inputFile);
        double objectiveValue = 0;

        while (in.hasNextLine()) {
            String line = in.nextLine();
            Scanner element = new Scanner(line);

            if (element.hasNext()) {
                objectiveValue = Double.parseDouble(element.next());
            } else {
                throw new IllegalArgumentException("No objective value from Xpress");
            }

        }
        in.close();

        return objectiveValue;
    }
}
