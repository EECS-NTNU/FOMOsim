package org.gleditsch_hagen.xpress;

import org.gleditsch_hagen.classes.*;
import org.gleditsch_hagen.enums.SolutionMethod;
import org.gleditsch_hagen.functions.TimeConverter;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.sql.Time;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;


public class WriteXpressFiles {

    public static void printTimeDependentInput (Input input, SolutionMethod solutionMethod)
            throws FileNotFoundException, UnsupportedEncodingException {
        String filename = input.getTimedependentInputFile();
        PrintWriter writer = new PrintWriter(filename, "UTF-8");

        switch (solutionMethod) {

            case HEURISTIC_VERSION_1:
                printWeights(input, writer);
                printLengthOfPlanningHorizon(input.getTimeHorizon(), writer);
                printMaxVisit(input.getMaxVisit(), writer);
                printMaxRoute(input.getVehicles().values(), writer);
                printVehicleInitialInformation(input.getVehicles().values(), input, writer);
                printStationInitialInformation(input.getStations().values(), writer, input.getCurrentMinute());
                printInteriorRepresentation(input, writer);
                break;

            case HEURISTIC_VERSION_2:
                printWeights(input, writer);
                printLengthOfPlanningHorizon(input.getTimeHorizon(), writer);
                printMaxVisit(input.getMaxVisit(), writer);
                printMaxRoute(input.getVehicles().values(), writer);
                printVehicleInitialInformation(input.getVehicles().values(), input, writer);
                printStationInitialInformation(input.getStations().values(), writer, input.getCurrentMinute());
                printInteriorRepresentation(input, writer);
                printLoadFromHeuristic(input, writer);
                printArrivalTimeFromHeuristic(input, writer);
                break;

            case HEURISTIC_VERSION_3:
                printWeights(input, writer);
                printMaxRoute(input.getVehicles().values(), writer);
                printInteriorRepresentationOnlyOrigin(input, writer);
                printObjectiveValues(input, writer);
                break;

            case EXACT_METHOD:
                printWeights(input, writer);
                printLengthOfPlanningHorizon(input.getTimeHorizon(), writer);
                printMaxVisit(input.getMaxVisit(), writer);
                printVehicleInitialInformation(input.getVehicles().values(), input, writer);
                printStationInitialInformation(input.getStations().values(), writer, input.getCurrentMinute());
                break;
        }

        writer.close();

    }

    private static void printObjectiveValues(Input input, PrintWriter writer) {


        //Violations if no visits
        writer.println();
        double totalViolationsIfNoVisit = 0;
        double totalDeviationsIfNoVisit = 0;

        for (Station station: input.getStations().values()) {

            double initialLoad = station.getLoad();
            double demandPerMinute = station.getNetDemand(TimeConverter.convertMinutesToHourRounded(input.getCurrentMinute()))/60;
            double loadAtHorizon = initialLoad + demandPerMinute*input.getTimeHorizon();
            double optimalState = station.getOptimalState(TimeConverter.convertMinutesToHourRounded(input.getCurrentMinute()));

            if (loadAtHorizon > station.getCapacity()) {
                totalViolationsIfNoVisit += loadAtHorizon-station.getCapacity();
                loadAtHorizon = station.getCapacity();
            }
            if (loadAtHorizon < 0) {
                totalViolationsIfNoVisit += -loadAtHorizon;
                loadAtHorizon = 0;
            }
            double diffFromOptimalState = Math.abs(optimalState-loadAtHorizon);
            totalDeviationsIfNoVisit += diffFromOptimalState;

        }
        writer.println("totalViolationsIfNoVisit : " + totalViolationsIfNoVisit);


        //Deviations if no visits
        writer.println();
        writer.println("totalDeviationIfNoVisit : " + totalDeviationsIfNoVisit);


        //Starvation prevented
        writer.println();
        writer.println("starvationPrevented : [");
        for (Vehicle vehicle : input.getVehicles().values()) {
            for (int route = 0; route < vehicle.getInitializedRoutes().size(); route++ ) {
                double starvationPrevented = findStarvationsPrevented(vehicle, route, input.getCurrentMinute(), input.getTimeHorizon());
                writer.println("( " + vehicle.getId() + " " + (route+1) + " ) " + starvationPrevented);
            }
            writer.println();
        }
        writer.println("]");


        //Congestion prevented
        writer.println();
        writer.println("congestionPrevented : [");
        for (Vehicle vehicle : input.getVehicles().values()) {
            for (int route = 0; route < vehicle.getInitializedRoutes().size(); route++ ) {
                double congestionsPrevented = findCongestionsPrevented(vehicle, route, input.getCurrentMinute(), input.getTimeHorizon());
                writer.println("( " + vehicle.getId() + " " + (route+1) + " ) " + congestionsPrevented);
            }
            writer.println();
        }
        writer.println("]");


        //Deviation improvements
        writer.println();
        writer.println("deviationImproved : [");
        for (Vehicle vehicle : input.getVehicles().values()) {
            for (int route = 0; route < vehicle.getInitializedRoutes().size(); route++ ) {
                double deviationImprovement = findDeviationImprovement(vehicle, route, input.getCurrentMinute(), input.getTimeHorizon());
                writer.println("( " + vehicle.getId() + " " + (route+1) + " ) " + deviationImprovement);
            }
            writer.println();
        }
        writer.println("]");


        //Deviation reward
        writer.println();
        writer.println("deviationReward : [");
        for (Vehicle vehicle : input.getVehicles().values()) {
            for (int route = 0; route < vehicle.getInitializedRoutes().size(); route++ ) {
                double deviationLastVisit = findDeviationLastVisit(vehicle, route, input.getCurrentMinute(), input.getTimeHorizon());
                writer.println("( " + vehicle.getId() + " " + (route+1) + " ) " + deviationLastVisit);
            }
            writer.println();
        }
        writer.println("]");


        //Driving time penalty
        writer.println();
        writer.println("drivingTimePenalty : [");
        for (Vehicle vehicle : input.getVehicles().values()) {
            for (int route = 0; route < vehicle.getInitializedRoutes().size(); route++ ) {
                double timeExceedingTimeHorizonAtLastStationVisit = findTimeExceedingTimeHorizonAtLastStationVisit(vehicle, route, input.getTimeHorizon());
                writer.println("( " + vehicle.getId() + " " + (route+1) + " ) " + timeExceedingTimeHorizonAtLastStationVisit);
            }
            writer.println();
        }
        writer.println("]");
    }


    private static double findTimeExceedingTimeHorizonAtLastStationVisit(Vehicle vehicle, int route, double timeHorizon) {
        double drivingTimeToLastStation = 0;

        int numberOfStationVisitInRoute = vehicle.getInitializedRoutes().get(route).size();
        double visitTimeLastStation = vehicle.getInitializedRoutes().get(route).get(numberOfStationVisitInRoute-1).getVisitTime();

        if (visitTimeLastStation > timeHorizon) {
            drivingTimeToLastStation = visitTimeLastStation-timeHorizon;
        }

        return drivingTimeToLastStation;

    }

    private static double findDeviationLastVisit(Vehicle vehicle, int route, double currentMinute, double timeHorizon) {
        double deviationLastStation = 0;

        //Deviation at last station visit if last station visit is after time horizon
        int numberOfStationVisitInRoute = vehicle.getInitializedRoutes().get(route).size();
        double visitTimeLastStation = vehicle.getInitializedRoutes().get(route).get(numberOfStationVisitInRoute-1).getVisitTime();

        if (visitTimeLastStation > timeHorizon) {

            double timeAtTimeHorizonHourRounded = TimeConverter.convertMinutesToHourRounded(currentMinute + timeHorizon);
            double currentHourRounded = TimeConverter.convertMinutesToHourRounded(currentMinute);

            StationVisit lastStationVisit = vehicle.getInitializedRoutes().get(route).get(vehicle.getInitializedRoutes().get(route).size()-1);
            double demand = lastStationVisit.getStation().getNetDemand(currentHourRounded);
            double optimalLoadAtHorizon = lastStationVisit.getStation().getOptimalState(timeAtTimeHorizonHourRounded);
            double stationInitialLoad = lastStationVisit.getStation().getLoad();
            int stationCapacity = lastStationVisit.getStation().getCapacity();


            double loadAtHorizon = stationInitialLoad + demand/60*timeHorizon;
            if (loadAtHorizon > stationCapacity) {
                loadAtHorizon = stationCapacity;
            } else if (loadAtHorizon < 0) {
                loadAtHorizon = 0;
            }
            deviationLastStation = Math.abs(loadAtHorizon-optimalLoadAtHorizon);
        }

        return deviationLastStation;
    }

    private static double findDeviationImprovement(Vehicle vehicle, int route, double currentMinute, double timeHorizon) {
        double timeAtTimeHorizonHourRounded = TimeConverter.convertMinutesToHourRounded(currentMinute + timeHorizon);
        double currentHourRounded = TimeConverter.convertMinutesToHourRounded(currentMinute);
        double deviationImprovement = 0;

        for (StationVisit stationVisit : vehicle.getInitializedRoutes().get(route)) {
            double demand = stationVisit.getStation().getNetDemand(currentHourRounded);
            double visitTime = stationVisit.getVisitTime();

            //If station is visited before time horizon, deviation might have been improved
            if (visitTime < timeHorizon) {

                double optimalLoadAtHorizon = stationVisit.getStation().getOptimalState(timeAtTimeHorizonHourRounded);
                int stationCapacity = stationVisit.getStation().getCapacity();
                double stationInitialLoad = stationVisit.getStation().getLoad();
                double loadingQuantity = stationVisit.getLoadingQuantity();

                //Deviation if no visit
                double loadAtHorizonIfNoVisit = stationInitialLoad + demand/60*timeHorizon;
                if (loadAtHorizonIfNoVisit > stationCapacity) {
                    loadAtHorizonIfNoVisit = stationCapacity;
                } else if (loadAtHorizonIfNoVisit < 0) {
                    loadAtHorizonIfNoVisit = 0;
                }
                double deviationWithNoVisit = Math.abs(loadAtHorizonIfNoVisit-optimalLoadAtHorizon);

                //Deviation with station visit
                double loadRightBeforeVisit = stationInitialLoad + demand/60*visitTime;
                if (loadRightBeforeVisit > stationCapacity) {
                    loadRightBeforeVisit = stationCapacity;
                } else if (loadRightBeforeVisit < 0) {
                    loadRightBeforeVisit = 0;
                }
                double loadAtTimeHorizonWithVisit = loadRightBeforeVisit + loadingQuantity + demand/60*(timeHorizon - visitTime);
                if (loadAtTimeHorizonWithVisit > stationCapacity) {
                    loadAtTimeHorizonWithVisit = stationCapacity;
                } else if (loadAtTimeHorizonWithVisit < 0) {
                    loadAtTimeHorizonWithVisit = 0;
                }
                double deviationWithVisit = Math.abs(loadAtTimeHorizonWithVisit-optimalLoadAtHorizon);

                deviationImprovement += deviationWithNoVisit-deviationWithVisit;
            }

        }

        return deviationImprovement;
    }

    private static double findCongestionsPrevented(Vehicle vehicle, int route, double currentMinute, double timeHorizon) {

        double currentHourRounded = TimeConverter.convertMinutesToHourRounded(currentMinute);
        double congestionsPrevented = 0;

        for (StationVisit stationVisit : vehicle.getInitializedRoutes().get(route)) {
            double demand = stationVisit.getStation().getNetDemand(currentHourRounded);
            double visitTime = stationVisit.getVisitTime();

            //If pick up station and station has been visited within time horizon, we might have prevented some congestions
            if (demand > 0 && visitTime < timeHorizon) {
                int stationCapacity = stationVisit.getStation().getCapacity();
                double stationInitialLoad = stationVisit.getStation().getLoad();
                double loadingQuantity = stationVisit.getLoadingQuantity();         //This will be a negative number
                double congestionIfNoVisit = 0;
                double congestionBeforeVisit = 0;
                double congestionAfterVisit = 0;

                //Congestions if no visits
                double loadAtHorizonIfNoVisit = stationInitialLoad + demand/60*timeHorizon;
                if (loadAtHorizonIfNoVisit > stationCapacity) {
                    congestionIfNoVisit = loadAtHorizonIfNoVisit - stationCapacity;
                }

                if (congestionIfNoVisit > 0) {
                    //Congestions with visit within timehorizon
                    //Congestions before visit
                    double loadRightBeforeVisit = stationInitialLoad + demand/60*visitTime;
                    if (loadRightBeforeVisit > stationCapacity) {
                        congestionBeforeVisit = loadRightBeforeVisit-stationCapacity;
                        loadRightBeforeVisit = stationCapacity;
                    }

                    //Congestions after visit
                    double loadAtHorizon = loadRightBeforeVisit+loadingQuantity+demand/60*(timeHorizon-visitTime);
                    if (loadAtHorizon > stationCapacity) {
                        congestionAfterVisit = loadAtHorizon-stationCapacity;
                    }
                }

                congestionsPrevented += congestionIfNoVisit - (congestionBeforeVisit + congestionAfterVisit);
            }
        }

        return congestionsPrevented;
    }


    private static double findStarvationsPrevented(Vehicle vehicle, int route, double currentMinute, double timeHorizon) {

        double currentHourRounded = TimeConverter.convertMinutesToHourRounded(currentMinute);
        double starvationsPrevented = 0;

        for (StationVisit stationVisit : vehicle.getInitializedRoutes().get(route)) {
            double demand = stationVisit.getStation().getNetDemand(currentHourRounded);
            double visitTime = stationVisit.getVisitTime();

            //If delivery station and station has been visited within time horizon, we might have prevented some starvations
            if (demand < 0 && visitTime < timeHorizon) {
                int stationCapacity = stationVisit.getStation().getCapacity();
                double stationInitialLoad = stationVisit.getStation().getLoad();
                double loadingQuantity = stationVisit.getLoadingQuantity();         //This will be a positive number
                double starvationIfNoVisit = 0;
                double starvationBeforeVisit = 0;
                double starvationAfterVisit = 0;

                //Starvations if no visits
                double loadAtHorizonIfNoVisit = stationInitialLoad + demand/60*timeHorizon;
                if (loadAtHorizonIfNoVisit < 0) {
                    starvationIfNoVisit = -loadAtHorizonIfNoVisit;
                }

                if (starvationIfNoVisit > 0) {
                    //Starvation with visit within timehorizon
                    //Starvation before visit
                    double loadRightBeforeVisit = stationInitialLoad + demand/60*visitTime;
                    if (loadRightBeforeVisit < 0) {
                        starvationBeforeVisit = -loadRightBeforeVisit;
                        loadRightBeforeVisit = 0;
                    }

                    //Starvations after visit
                    double loadAtHorizon = loadRightBeforeVisit+loadingQuantity+demand/60*(timeHorizon-visitTime);
                    if (loadAtHorizon < 0) {
                        starvationAfterVisit = -loadAtHorizon;
                    }
                }

                starvationsPrevented += starvationIfNoVisit - (starvationBeforeVisit + starvationAfterVisit);
            }
        }

        return starvationsPrevented;
    }


    private static void printLoadFromHeuristic(Input input, PrintWriter writer) {
        writer.println();
        writer.println("intRepUnloadingQuantity : [");


        for (Vehicle vehicle : input.getVehicles().values()) {
            for (int route = 0; route < vehicle.getInitializedRoutes().size(); route++) {

                HashMap<Integer, Double> pickUpStations = new HashMap<>();

                for (int stationVisitNr = 0; stationVisitNr < vehicle.getInitializedRoutes().get(route).size(); stationVisitNr++) {

                    int stationId = vehicle.getInitializedRoutes().get(route).get(stationVisitNr).getStation().getId();
                    double load = vehicle.getInitializedRoutes().get(route).get(stationVisitNr).getLoadingQuantity();

                    //Pick up station
                    if (load < 0) {
                        //Check if station is already in hashmap
                        if (pickUpStations.containsKey(stationId)) {
                            double currentLoad = pickUpStations.get(stationId);
                            pickUpStations.put(stationId, currentLoad - load);
                        } else {
                            pickUpStations.put(stationId, -load);
                        }
                    }
                }

                //Print
                for (int stationId : pickUpStations.keySet()) {
                    double load = pickUpStations.get(stationId);
                    if (load > 0) {
                        writer.println("( " + stationId + " " + vehicle.getId() + " " + (route + 1) + " ) " + load);
                    }
                }

            }
        }
        writer.println("]");


        writer.println();
        writer.println("intRepLoadingQuantity : [");
        for (Vehicle vehicle : input.getVehicles().values()) {
            for (int route = 0; route < vehicle.getInitializedRoutes().size(); route++) {

                HashMap<Integer, Double> deliveryStations = new HashMap<>();

                for (int stationVisitNr = 0; stationVisitNr < vehicle.getInitializedRoutes().get(route).size(); stationVisitNr++) {

                    int stationId = vehicle.getInitializedRoutes().get(route).get(stationVisitNr).getStation().getId();
                    double load = vehicle.getInitializedRoutes().get(route).get(stationVisitNr).getLoadingQuantity();

                    //Delivery station
                    if (load >= 0) {
                        //Check if station is already in hashmap
                        if (deliveryStations.containsKey(stationId)) {
                            double currentLoad = deliveryStations.get(stationId);
                            deliveryStations.put(stationId, currentLoad + load);
                        } else {
                            deliveryStations.put(stationId, load);
                        }
                    }
                }
                //Print
                for (int stationId : deliveryStations.keySet()) {
                    double load = deliveryStations.get(stationId);
                    if (load > 0) {
                        writer.println("( " + stationId + " " + vehicle.getId() + " " + (route + 1) + " ) " + load);
                    }
                }

            }
        }
        writer.println("]");
    }

    private static void printArrivalTimeFromHeuristic(Input input, PrintWriter writer) {
        writer.println();
        writer.println("intRepArrivalTime : [");


        for (Vehicle vehicle : input.getVehicles().values()) {
            for (int route = 0; route < vehicle.getInitializedRoutes().size(); route++) {

                HashMap<Integer, Double> stations = new HashMap<>();

                for (int stationVisitNr = 0; stationVisitNr < vehicle.getInitializedRoutes().get(route).size(); stationVisitNr++) {

                    int stationId = vehicle.getInitializedRoutes().get(route).get(stationVisitNr).getStation().getId();
                    double arrivalTime = vehicle.getInitializedRoutes().get(route).get(stationVisitNr).getVisitTime();

                    //Check if station is already in hashmap
                    if (stations.containsKey(stationId)) {
                        double currentTime = stations.get(stationId);
                        stations.put(stationId, currentTime + arrivalTime);
                    } else {
                        stations.put(stationId, arrivalTime);
                    }
                }

                //Print
                for (int stationId : stations.keySet()) {
                    double arrivalTime = stations.get(stationId);
                    if (arrivalTime > 0) {
                        writer.println("( " + stationId + " " + vehicle.getId() + " " + (route + 1) + " ) " + arrivalTime);
                    }
                }
            }
        }
        writer.println("]");

    }

    private static void printStationInitialInformation(Collection<Station> values, PrintWriter writer, double currentMinute) {
        //stationsInitialLoad
        writer.println();
        writer.println("stationsInitialLoad : [");
        for (Station station : values) {
            writer.println(station.getLoad());
        }
        writer.println("0");
        writer.println("]");

        //optimalState
        writer.println();
        writer.println("optimalState : [");
        for (Station station : values) {
            writer.println(station.getOptimalState(TimeConverter.convertSecondsToHourRounded(currentMinute*60)));
        }
        writer.println("0");
        writer.println("]");

        //stationDemand - net demand per minute
        writer.println();
        writer.println("stationDemand : [");
        for (Station station : values) {
            double bikeWanted = station.getBikeWantedMedian(TimeConverter.convertSecondsToHourRounded(currentMinute*60));
            double bikeReturned = station.getBikeReturnedMedian(TimeConverter.convertSecondsToHourRounded(currentMinute*60));
            writer.println((bikeReturned-bikeWanted)/60);
        }
        writer.println("0");
        writer.println("]");

        //StarvationStations - station with negative net demand
        writer.println();
        writer.println("StarvationStations : [");
        for (Station station : values) {
            double bikeWanted = station.getBikeWantedMedian(TimeConverter.convertSecondsToHourRounded(currentMinute*60));
            double bikeReturned = station.getBikeReturnedMedian(TimeConverter.convertSecondsToHourRounded(currentMinute*60));
            if (bikeReturned-bikeWanted <= 0 ) {
                writer.println(station.getId());
            }
        }
        writer.println("]");

        //CongestionStations - station with positive net demand
        writer.println();
        writer.println("CongestionStations : [");
        for (Station station : values) {
            double bikeWanted = station.getBikeWantedMedian(TimeConverter.convertSecondsToHourRounded(currentMinute*60));
            double bikeReturned = station.getBikeReturnedMedian(TimeConverter.convertSecondsToHourRounded(currentMinute*60));
            if (bikeReturned-bikeWanted >= 0 ) {
                writer.println(station.getId());
            }
        }
        writer.println("]");
    }

    private static void printVehicleInitialInformation(Collection<Vehicle> vehicles, Input input, PrintWriter writer) {

        //vehicleInitialStation
        writer.println("vehicleInitialStation : [");
        for (Vehicle vehicle : vehicles) {
            writer.println(vehicle.getNextStation());
        }
        writer.println("]");

        //vehicleRemainingTimeToInitialStation
        writer.println();
        writer.println("vehicleRemainingTimeToInitialStation : [");
        for (Vehicle vehicle : vehicles) {
            writer.println(vehicle.getTimeToNextStation());
        }
        writer.println("]");

        //vehicleInitialLoad
        writer.println();
        writer.println("vehicleInitialLoad : [");
        for (Vehicle vehicle : vehicles) {
            writer.println(vehicle.getLoad());
        }
        writer.println("]");
    }

    private static void printMaxRoute(Collection<Vehicle> values, PrintWriter writer) {
        //maxRoute
        writer.print("maxRoute : ");
        int maxRoute = 0;
        for(Vehicle vehicle : values) {
            if (vehicle.getInitializedRoutes().size() > maxRoute) {
                maxRoute = vehicle.getInitializedRoutes().size();
            }
        }
        writer.println(maxRoute+1);
        writer.println();
    }

    private static void printMaxVisit(int maxVisit, PrintWriter writer) {
        //maxVisit
        writer.print("maxVisits : ");
        writer.println(maxVisit);
        writer.println();
    }

    private static void printLengthOfPlanningHorizon(double timeHorizon, PrintWriter writer) {
        //lengthOfPlanningHorizon
        writer.print("lengthOfPlanningHorizon : ");
        writer.println(timeHorizon);
    }

    private static void printWeights(Input input, PrintWriter writer) {
        //weightViolation
        writer.print("weightViolation : ");
        writer.println(input.getWeightViolation());

        //weightDeviation
        writer.print("weightDeviation : ");
        writer.println(input.getWeightDeviation());

        //weightReward
        writer.print("weightReward : ");
        writer.println(input.getWeightReward());

        //weightDeviation
        writer.print("weightDeviationReward : ");
        writer.println(input.getWeightDeviationReward());

        //weightReward
        writer.print("weightDrivingTimePenalty : ");
        writer.println(input.getWeightDrivingTimePenalty());
        writer.println();
    }

    private static void printInteriorRepresentationOnlyOrigin(Input input, PrintWriter writer) {
        writer.println();
        writer.println("intRep : [");

        for (Vehicle vehicle : input.getVehicles().values()) {

            for (int route = 0; route < vehicle.getInitializedRoutes().size(); route++) {

                for (int stationVisitNr = 0; stationVisitNr < vehicle.getInitializedRoutes().get(route).size(); stationVisitNr++) {

                    //From station
                    int stationOriginID = vehicle.getInitializedRoutes().get(route).get(stationVisitNr).getStation().getId();

                    writer.println("( " + stationOriginID + " " + vehicle.getId() + " " + (route + 1) + " ) " + 1);

                }

                writer.println();
            }
        }
        writer.println("]");

    }

    private static void printInteriorRepresentation(Input input, PrintWriter writer) {
        writer.println();
        writer.println("intRep : [");
        for (Vehicle vehicle : input.getVehicles().values()) {
            for (int route = 0; route < vehicle.getInitializedRoutes().size(); route++) {

                ArrayList<ArrayList<Integer>> allStationPairs = new ArrayList<>();

                for (int stationVisitNr = 0; stationVisitNr < vehicle.getInitializedRoutes().get(route).size(); stationVisitNr++) {

                    ArrayList<Integer> stationPair = new ArrayList<>();

                    //From station
                    stationPair.add(vehicle.getInitializedRoutes().get(route).get(stationVisitNr).getStation().getId());

                    //To station (or artificial station)
                    if (stationVisitNr == vehicle.getInitializedRoutes().get(route).size()-1) {
                        stationPair.add(0);
                    } else {
                        stationPair.add(vehicle.getInitializedRoutes().get(route).get(stationVisitNr+1).getStation().getId());
                    }

                    allStationPairs.add(stationPair);
                }

                ArrayList<ArrayList<Integer>> alreadyPrinted = new ArrayList<>();

                for (ArrayList<Integer> stationPair : allStationPairs) {

                    //Check if station pair is not already printed
                    if (!alreadyPrinted.contains(stationPair)) {
                        //Count how many identical station pairs there are
                        int count = 0;
                        for (ArrayList<Integer> stationPairToCompareAgainst : allStationPairs) {
                            if (stationPair.equals(stationPairToCompareAgainst)) {
                                count++;
                            }
                        }

                        //print
                        writer.println("( " + stationPair.get(0) + " " + stationPair.get(1) + " " + vehicle.getId() + " " + (route+1) + " ) " + count);
                        alreadyPrinted.add(stationPair);
                    }
                }
                writer.println();
            }
        }
        writer.println("]");

    }


    public static void printFixedInput (Input input)
            throws FileNotFoundException, UnsupportedEncodingException {
        String filename = input.getFixedInputFile();
        PrintWriter writer = new PrintWriter(filename, "UTF-8");


        writer.println("artificialStation: 0");
        writer.println("visitInterval: " + input.getVisitInterval());
        writer.println("loadInterval: " + input.getLoadInterval());
        if (input.isSimulation()) {
            writer.println("simulation: 1");
        } else {
            writer.println("simulation: 0");
        }
        if (input.isRunPricingProblem()) {
            writer.println("pricingProblem: 1");
        } else {
            writer.println("pricingProblem: 0");
        }

        //Station IDs
        writer.println();
        writer.println("Stations : [");
        for (Station station : input.getStations().values()) {
            writer.println(station.getId());
        }
        writer.println("0");
        writer.println("]");

        //Station capacities
        writer.println();
        writer.println("stationsCapacities : [");
        for (Station station : input.getStations().values()) {
            writer.println(station.getCapacity());
        }
        writer.println("0");
        writer.println("]");

        //VEHICLES
        writer.println("vehicleParkingTime: " + input.getVehicleParkingTime());
        writer.println("unitHandlingTime: " + input.getVehicleHandlingTime());

        //Vehicle IDs
        writer.println();
        writer.println("Vehicles : [");
        for (Vehicle vehicle : input.getVehicles().values()) {
            writer.println(vehicle.getId());
        }
        writer.println("]");

        //vehicleCapacity
        writer.println();
        writer.println("vehicleCapacity : [");
        for (Vehicle vehicle : input.getVehicles().values()) {
            writer.println(vehicle.getCapacity());
        }
        writer.println("]");

        //DrivingTime
        writer.println();
        writer.println("drivingTime : [");
        for (Station stationOrigin : input.getStations().values()) {
            for (Station stationDestination : input.getStations().values()) {
                writer.print(stationOrigin.getDrivingTimeToStation(stationDestination.getId()) + " ");
            }
            writer.println("0");
        }
        for (Station station : input.getStations().values()) {
            writer.print("0 ");
        }

        writer.println("0");
        writer.println("]");

        writer.close();
    }

    public static void writeClusterInformation(Input input) throws FileNotFoundException, UnsupportedEncodingException {

        String filename = "clusterInput.txt";
        PrintWriter writer = new PrintWriter(filename, "UTF-8");

        writer.println("weightDrivingTime: " + input.getWeightClusterDrivingTime());
        writer.println("weightNetDemand: " + input.getWeightClusterNetDemand());
        writer.println("weightEqualSize: " + input.getWeightClusterEqualSize());
        writer.println("instance: " + input.getTestInstance());
        writer.println("vehicleNr: " + input.getVehicles().size());

        //Station IDs
        writer.println();
        writer.println("Stations : [");
        for (Station station : input.getStations().values()) {
            writer.println(station.getId());
        }
        writer.println("]");

        //initialStation
        writer.println();
        writer.println("initialStation : [");
        for (Vehicle vehicle : input.getVehicles().values()) {
            writer.println(vehicle.getNextStation());
        }
        writer.println("]");

        //Vehicle IDs
        writer.println();
        writer.println("Vehicles : [");
        for (Vehicle vehicle : input.getVehicles().values()) {
            writer.println(vehicle.getId());
        }
        writer.println("]");

        //Demand
        writer.println();
        writer.println("demand : [");
        for (Station station : input.getStations().values()) {
            writer.println(station.getNetDemand(TimeConverter.convertMinutesToHourRounded(input.getCurrentMinute())));
        }
        writer.println("]");

        //ClusterNr
        writer.println();
        writer.println("clusterNr : [");
        for (Station station : input.getStations().values()) {
            double netDemand = station.getNetDemand(TimeConverter.convertMinutesToHourRounded(input.getCurrentMinute()));
            double highDemand = input.getHighDemand();
            double mediumDemand = input.getMediumDemand();
            if (netDemand >= highDemand || netDemand <= -highDemand) {
                writer.println(2);
            } else if (netDemand >= mediumDemand || netDemand <= -mediumDemand) {
                writer.println(1);
            } else {
                writer.println(0);
            }
        }
        writer.println("]");

        //DrivingTime
        writer.println();
        writer.println("drivingTime : [");
        for (Station stationOrigin : input.getStations().values()) {
            for (Station stationDestination : input.getStations().values()) {
                writer.print(stationOrigin.getDrivingTimeToStation(stationDestination.getId()) + " ");
            }
            writer.println();
        }
        writer.println("]");

        writer.close();


    }
}
