package org.gleditsch_hagen.classes;

import com.dashoptimization.XPRMCompileException;
import org.gleditsch_hagen.enums.SolutionMethod;
import org.gleditsch_hagen.functions.ReadClusterList;
import org.gleditsch_hagen.functions.TimeConverter;
import org.gleditsch_hagen.solutionMethods.*;
import org.gleditsch_hagen.enums.NextEvent;
import org.gleditsch_hagen.functions.NextSimulation;
import org.gleditsch_hagen.xpress.ReadXpressResult;
import org.gleditsch_hagen.xpress.RunXpress;
import org.gleditsch_hagen.xpress.WriteXpressFiles;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

public class Simulation {

    private double numberOfTimesVehicleRouteGenerated = 0;
    private double congestions = 0;
    private double starvations = 0;
    private int totalNumberOfCustomers = 0;
    private int happyCustomers = 0;
    private ArrayList<Double> timeToNextSimulationList = new ArrayList<>();
    private ArrayList<Double> computationalTimesXpress = new ArrayList<>();
    private ArrayList<Double> computationalTimesXpressPlusInitialization = new ArrayList<>();
    private double maxComputationalTimeIncludingInitialization = -1;
    private double minComputationalTimeIncludingInitialization = -1;
    private ArrayList<VehicleArrival> vehicleArrivals = new ArrayList<>();
    private ArrayList<Double> numberOfTimesPPImproved = new ArrayList<>();

    //Constructor
    public Simulation() {
    }

    public static FomoAction policy(ArrayList<FomoStation> stations, FomoVehicle vehicle) {
      FomoAction action = new FomoAction();

      try {
        HashMap<Integer, Station> stationMap = new HashMap<>();
        //System.out.println("Stations:");
        for(FomoStation s : stations) {
          Station station = new Station((int)s.id + 1);
          station.setLoad(s.bikes.size());
          station.setCapacity((int)s.capacity);
          station.setBikeReturnedMedian(0, s.arriveIntensity);
          station.setBikeWantedMedian(0, s.leaveIntensity);
          station.setOptimalState(0, s.idealState);
          //System.out.println("  id: " + (s.id + 1));
          //System.out.println("  load: " + s.bikes.size());
          //System.out.println("  cap: " + s.capacity);
          //System.out.println("  arrive: " + s.arriveIntensity);
          //System.out.println("  leave: " + s.leaveIntensity);
          //System.out.println("  ideal: " + s.idealState);
          //System.out.println("  distances: ");
          for(Map.Entry<Long, Double> entry : s.distances.entrySet()) {
            Long d = entry.getKey() + 1;
            double t = entry.getValue() * 15;
            station.addDistanceToStationHashmap(d.intValue(), t);
            //System.out.println("    dest: " + d + " dist: " + t);
          }
          //System.out.println();
          stationMap.put((int)s.id + 1, station);
        }

        HashMap<Integer, Vehicle> vehicleMap = new HashMap<>();
        Vehicle v = new Vehicle((int)vehicle.id + 1);
        v.setCapacity((int)vehicle.capacity);
        v.setLoad(vehicle.bikes.size());
        v.setNextStation((int)vehicle.currentStation + 1);
        v.setTimeToNextStation(0);
        ArrayList<Station> stationsList = new ArrayList<>();
        stationsList.addAll(stationMap.values());
        v.setClusterStationList(stationsList);
        vehicleMap.put((int)vehicle.id + 1, v);
        //System.out.println("\nVehicle:");
        //System.out.println("  id: " + (vehicle.id + 1));
        //System.out.println("  cap: " + vehicle.capacity);
        //System.out.println("  load: " + vehicle.bikes.size());
        //System.out.println("  location: " + (vehicle.currentStation + 1));

        Input input = new Input(stationMap, vehicleMap);
        input.setSimulationStartTime(0);

        WriteXpressFiles.printFixedInput(input);

        ArrayList<VehicleArrival> vehicleArrivals = generateVehicleRouteNoProfiling(input);

        for(VehicleArrival nextVehicleArrival : vehicleArrivals) {
          int vehicleId = nextVehicleArrival.getVehicle();

          if(vehicleId == (vehicle.id + 1)) {
            Vehicle vhcl = vehicleMap.get(vehicleId);

            int vehicleLoad = vhcl.getLoad();
            int vehicleCapacity = vhcl.getCapacity();

            int stationId = nextVehicleArrival.getStationId();
            Station station = stationMap.get(stationId);
            double stationLoad = station.getLoad();
            int stationCapacity = station.getCapacity();

            int load = nextVehicleArrival.getLoad();

            //Load to station
            if (load > 0) {
              if (stationLoad + load > stationCapacity) {
                load = (int) (stationCapacity - stationLoad);
              }
              if (load > vehicleLoad) {
                load = vehicleLoad;
              }
              for(int i = 0; i < load; i++) {
                action.deliveryScooters.add(vehicle.bikes.get(i));
              }
            }

            //Unload from station
            else {
              load = -load;
              if (stationLoad < load ) {
                load = (int) (stationLoad);
              }
              if (vehicleCapacity - vehicleLoad < load) {
                load = vehicleCapacity - vehicleLoad;
              }
              for(int i = 0; i < load; i++) {
                action.pickUps.add(stations.get(stationId - 1).bikes.get(i));
              }
            }

            action.nextLocation = nextVehicleArrival.getNextStationId();

            break;
          }
        }

      } catch (Exception e) {
        e.printStackTrace(System.out);
      }

      return action;
    }

    public void run(String simulationFile, Input input) throws IOException, XPRMCompileException, InterruptedException {

        //Demand input file
        CustomerArrival nextCustomerArrival = new CustomerArrival();
        File inputFile = new File(simulationFile);
        input.setCurrentMinute(input.getSimulationStartTime());
        Scanner in = new Scanner(inputFile);


        // 1 : SIMULATION STOP TIME
        double simulationStopTime = input.getSimulationStopTime();                              //Actual time minutes


        // 2 : TIME FOR NEW VEHICLE ROUTES
        double timeToNewVehicleRoutes = simulationStopTime + 1;

        //If vehicle routes are to be generated
        if (!input.getSolutionMethod().equals(SolutionMethod.NO_VEHICLES)) {

            //Generate routes for service vehicles
            vehicleArrivals = generateVehicleRoute(input);

            //Determine time to generate new vehicle routes
            timeToNewVehicleRoutes = NextSimulation.determineTimeToNextSimulation(vehicleArrivals, input.getTimeHorizon(), input.getReOptimizationMethod(), input.getCurrentMinute());      //Actual time minutes
            this.timeToNextSimulationList.add(timeToNewVehicleRoutes-input.getCurrentMinute());

            System.out.println();
            System.out.println("Remaining time: " + (simulationStopTime - timeToNewVehicleRoutes));

        }


        //3 : NEXT CUSTOMER ARRIVAL
        nextCustomerArrival.updateNextCustomerArrival(in, input.getCurrentMinute(), simulationStopTime);         //Actual time minutes


        // 4 : NEXT VEHICLE ARRIVAL
        int vehicleArrivalCounter = 0;
        VehicleArrival nextVehicleArrival = new VehicleArrival(simulationStopTime);
        if (!input.getSolutionMethod().equals(SolutionMethod.NO_VEHICLES)) {
            if (this.vehicleArrivals.size() > 0) {
                nextVehicleArrival = this.vehicleArrivals.get(vehicleArrivalCounter);                        //Actual time minutes
            }
        }





        //Determine next event (simulation stop, new vehicle routes, customer arrival, vehicle arrival)
        boolean stopTimeReached = false;

        while (!stopTimeReached) {

            double nextEventTime = simulationStopTime;
            NextEvent nextEvent = NextEvent.SIMULATION_STOP;

            //If vehicle routes are to be generated
            if (!input.getSolutionMethod().equals(SolutionMethod.NO_VEHICLES)) {
                if (timeToNewVehicleRoutes < nextEventTime) {
                    nextEventTime = timeToNewVehicleRoutes;
                    nextEvent = NextEvent.NEW_VEHICLE_ROUTES;
                }
            }

            if (nextCustomerArrival.getTime() < nextEventTime) {
                nextEventTime = nextCustomerArrival.getTime();
                nextEvent = NextEvent.CUSTOMER_ARRIVAL;
            }

            //If vehicle routes are to be generated
            if (!input.getSolutionMethod().equals(SolutionMethod.NO_VEHICLES)) {
                if (nextVehicleArrival.getTime() < nextEventTime) {
                    nextEventTime = nextVehicleArrival.getTime();
                    nextEvent = NextEvent.VEHICLE_ARRIVAL;
                }
            }



            switch (nextEvent) {

                case SIMULATION_STOP:
                    stopTimeReached = true;
                    break;

                case NEW_VEHICLE_ROUTES:

                    //Generate new clusters
                    if (input.isDynamicClustering()) {
                        WriteXpressFiles.writeClusterInformation(input);
                        RunXpress.runXpress("createCluster");
                        String xpressOutputFile = "clusterOutput-Instance" + input.getTestInstance() + "-V" + input.getVehicles().size()+".txt";
                        ReadClusterList.readClusterListTextFile(input, xpressOutputFile);
                    }

                    //Generate new routes
                    determineRemainingDrivingTimeAndStation(timeToNewVehicleRoutes, input.getVehicles(), vehicleArrivals, input.getStations(), input.getCurrentMinute() );
                    input.setCurrentMinute(timeToNewVehicleRoutes);

                    vehicleArrivals = generateVehicleRoute(input);


                    //Update nextVehicleArrival
                    vehicleArrivalCounter = 0;
                    nextVehicleArrival = vehicleArrivals.get(vehicleArrivalCounter);
                    timeToNewVehicleRoutes = NextSimulation.determineTimeToNextSimulation(vehicleArrivals, input.getTimeHorizon(), input.getReOptimizationMethod(), input.getCurrentMinute());      //Actual time minutes
                    this.timeToNextSimulationList.add(timeToNewVehicleRoutes-input.getCurrentMinute());

                    System.out.println();
                    System.out.println("Remaining time: " + (simulationStopTime - timeToNewVehicleRoutes));
                    break;

                case CUSTOMER_ARRIVAL:
                    upDateLoadAndViolation(nextCustomerArrival, input.getStations());
                    input.setCurrentMinute(nextCustomerArrival.getTime());
                    nextCustomerArrival.updateNextCustomerArrival(in, input.getCurrentMinute(), simulationStopTime);
                    break;

                case VEHICLE_ARRIVAL:
                    updateStationAfterVehicleArrival(nextVehicleArrival, input.getStations(), input.getVehicles());
                    vehicleArrivalCounter ++;
                    input.setCurrentMinute(nextVehicleArrival.getTime());
                    nextVehicleArrival = updateNextVehicleArrival(vehicleArrivals, vehicleArrivalCounter, simulationStopTime);
                    break;

            }
        }
    }

    //Find next vehicle arrival
    private VehicleArrival updateNextVehicleArrival(ArrayList<VehicleArrival> vehicleArrivals, int vehicleArrivalCounter, double simulationStopTime) {

        int lengthVehicleArrivals = vehicleArrivals.size();
        if (vehicleArrivalCounter < lengthVehicleArrivals) {
            return vehicleArrivals.get(vehicleArrivalCounter);
        } else {
            return new VehicleArrival(simulationStopTime);
        }
    }

    //Update load and violation when a customer arrives at a station
    private void upDateLoadAndViolation (CustomerArrival nextCustomerArrival, HashMap<Integer, Station> stations) {
        Station station = stations.get(nextCustomerArrival.getStationId());
        if (nextCustomerArrival.getLoad() > 0) {
            //Check for congestion
            if (station.getLoad() + 1 > station.getCapacity() ) {
                this.congestions++;
            } else {
                station.addBikeToStation(1);
                this.happyCustomers ++;
            }
        } else if (nextCustomerArrival.getLoad() < 0) {
            //Check for starvation
            if (station.getLoad() - 1 < 0) {
                this.starvations++;
            } else {
                station.addBikeToStation(-1);
                this.happyCustomers ++;
            }
        }
        this.totalNumberOfCustomers++;
    }

    //Update load when a vehicle arrives at a station
    private void updateStationAfterVehicleArrival (VehicleArrival nextVehicleArrival, HashMap<Integer, Station> stations, HashMap<Integer, Vehicle> vehicles) {
        int vehicleId = nextVehicleArrival.getVehicle();
        Vehicle vehicle = vehicles.get(vehicleId);
        int vehicleLoad = vehicle.getLoad();
        int vehicleCapacity = vehicle.getCapacity();

        int stationId = nextVehicleArrival.getStationId();
        Station station = stations.get(stationId);
        double stationLoad = station.getLoad();
        int stationCapacity = station.getCapacity();
        int load = nextVehicleArrival.getLoad();

        //Load to station
        if (load > 0) {
            if (stationLoad + load > stationCapacity) {
                load = (int) (stationCapacity - stationLoad);
            }
            if (load > vehicleLoad) {
                load = vehicleLoad;
            }
            station.addBikeToStation(load);
            vehicle.addLoad(-load);
        }

        //Unload from station
        else {
            load = -load;
            if (stationLoad < load ) {
                load = (int) (stationLoad);
            }
            if (vehicleCapacity - vehicleLoad < load) {
                load = vehicleCapacity - vehicleLoad;
            }
            station.addBikeToStation(-load);
            vehicle.addLoad(load);
        }
    }

    //Generate new vehicle routes
    static private ArrayList<VehicleArrival> generateVehicleRouteNoProfiling(Input input) throws IOException, XPRMCompileException {
        ArrayList<VehicleArrival> vehicleArrivals;

        switch (input.getSolutionMethod()) {
            case HEURISTIC_VERSION_1:
                HeuristicVersion1 heuristicVersion1 = new HeuristicVersion1(input);
                vehicleArrivals = ReadXpressResult.readVehicleArrivals(input.getCurrentMinute());
                break;

            case HEURISTIC_VERSION_2:
                HeuristicVersion2 heuristicVersion2 = new HeuristicVersion2(input);
                vehicleArrivals = ReadXpressResult.readVehicleArrivals(input.getCurrentMinute());
                break;

            case HEURISTIC_VERSION_3:
                HeuristicVersion3 heuristicVersion3 = new HeuristicVersion3(input);
                vehicleArrivals = ReadXpressResult.readVehicleArrivalsVersion3(input.getVehicles(), input.getCurrentMinute());
                break;

            case EXACT_METHOD:
                ExactMethod exactMethod = new ExactMethod(input);
                vehicleArrivals = ReadXpressResult.readVehicleArrivals(input.getCurrentMinute());
                break;

            case CURRENT_SOLUTION_IN_OSLO:
                CurrentSolutionInOslo currentSolutionInOslo = new CurrentSolutionInOslo(input);
                vehicleArrivals = currentSolutionInOslo.getVehicleArrivals();
                break;

            default:
            case NO_VEHICLES:
                NoVehicles noVehicles = new NoVehicles(input);
                vehicleArrivals = new ArrayList<>();
                break;

        }

        return vehicleArrivals;
    }

    private ArrayList<VehicleArrival> generateVehicleRoute(Input input) throws IOException, XPRMCompileException {
        ArrayList<VehicleArrival> vehicleArrivals;

        numberOfTimesVehicleRouteGenerated ++;

        double totalTime = 0;


        switch (input.getSolutionMethod()) {
            case HEURISTIC_VERSION_1:
                HeuristicVersion1 heuristicVersion1 = new HeuristicVersion1(input);
                computationalTimesXpress.add(heuristicVersion1.getComputationalTimeXpress());
                computationalTimesXpressPlusInitialization.add(heuristicVersion1.getComputationalTimeIncludingInitialization());
                totalTime = heuristicVersion1.getComputationalTimeIncludingInitialization();
                vehicleArrivals = ReadXpressResult.readVehicleArrivals(input.getCurrentMinute());
                break;

            case HEURISTIC_VERSION_2:
                HeuristicVersion2 heuristicVersion2 = new HeuristicVersion2(input);
                computationalTimesXpress.add(heuristicVersion2.getComputationalTimeXpress());
                computationalTimesXpressPlusInitialization.add(heuristicVersion2.getComputationalTimeIncludingInitialization());
                totalTime = heuristicVersion2.getComputationalTimeIncludingInitialization();
                vehicleArrivals = ReadXpressResult.readVehicleArrivals(input.getCurrentMinute());
                break;

            case HEURISTIC_VERSION_3:
                HeuristicVersion3 heuristicVersion3 = new HeuristicVersion3(input);
                computationalTimesXpress.add(heuristicVersion3.getComputationalTimeXpress());
                computationalTimesXpressPlusInitialization.add(heuristicVersion3.getComputationalTimeIncludingInitialization());
                totalTime = heuristicVersion3.getComputationalTimeIncludingInitialization();
                vehicleArrivals = ReadXpressResult.readVehicleArrivalsVersion3(input.getVehicles(), input.getCurrentMinute());
                numberOfTimesPPImproved.add(heuristicVersion3.getNumberOfTimesObjectiveImproved());
                break;

            case EXACT_METHOD:
                ExactMethod exactMethod = new ExactMethod(input);
                computationalTimesXpress.add(exactMethod.getComputationalTime());
                computationalTimesXpressPlusInitialization.add(exactMethod.getComputationalTime());
                totalTime = exactMethod.getComputationalTime();
                vehicleArrivals = ReadXpressResult.readVehicleArrivals(input.getCurrentMinute());
                break;

            case CURRENT_SOLUTION_IN_OSLO:
                CurrentSolutionInOslo currentSolutionInOslo = new CurrentSolutionInOslo(input);
                vehicleArrivals = currentSolutionInOslo.getVehicleArrivals();
                computationalTimesXpress.add(0.0);
                computationalTimesXpressPlusInitialization.add(0.0);
                break;

            case NO_VEHICLES:
                NoVehicles noVehicles = new NoVehicles(input);
                computationalTimesXpress.add(0.0);
                computationalTimesXpressPlusInitialization.add(0.0);
                vehicleArrivals = new ArrayList<>();
                break;
            default:
                computationalTimesXpress.add(0.0);
                computationalTimesXpressPlusInitialization.add(0.0);
                vehicleArrivals = new ArrayList<>();

        }

        if (maxComputationalTimeIncludingInitialization == -1 || totalTime > maxComputationalTimeIncludingInitialization) {
            maxComputationalTimeIncludingInitialization = totalTime;
        }
        if (minComputationalTimeIncludingInitialization == -1 || totalTime < minComputationalTimeIncludingInitialization) {
            minComputationalTimeIncludingInitialization = totalTime;
        }

        return vehicleArrivals;
    }

    //Determine time to next station, works as input to next vehicle route generation
    private void determineRemainingDrivingTimeAndStation(double timeForNewVehicleRoutes, HashMap<Integer, Vehicle> vehicles,
                                                         ArrayList<VehicleArrival> vehicleArrivals, HashMap<Integer, Station> stations, double currentMinute) {

        for (VehicleArrival vehicleArrival : vehicleArrivals) {

            boolean vehicleArrivalBeforeGeneratingNewRoutes = vehicleArrival.getTime() < timeForNewVehicleRoutes;
            boolean nextVehicleArrivalAfterOrAtTimeForGeneratingNewRoutes = vehicleArrival.getTimeNextVisit() >= timeForNewVehicleRoutes;
            boolean vehicleArrivalFirstVisit = vehicleArrival.isFirstvisit();
            boolean vehicleArrivalAfterOrAtTimeForGeneratingNewRoutes = vehicleArrival.getTime() >= timeForNewVehicleRoutes;
            boolean nextStationIsArtificialStation = vehicleArrival.getNextStationId() == 0;

            if ( vehicleArrivalBeforeGeneratingNewRoutes & nextVehicleArrivalAfterOrAtTimeForGeneratingNewRoutes & !nextStationIsArtificialStation) {
                int vehicleId = vehicleArrival.getVehicle();
                Vehicle vehicle = vehicles.get(vehicleId);
                double timeToNextStation = vehicleArrival.getTimeNextVisit()-timeForNewVehicleRoutes;
                vehicle.setTimeToNextStation(timeToNextStation);
                vehicle.setNextStation(vehicleArrival.getNextStationId());

            } else if (vehicleArrivalFirstVisit & vehicleArrivalAfterOrAtTimeForGeneratingNewRoutes){
                int vehicleId = vehicleArrival.getVehicle();
                Vehicle vehicle = vehicles.get(vehicleId);
                double timeToNextStation = vehicleArrival.getTime()-timeForNewVehicleRoutes;
                vehicle.setTimeToNextStation(timeToNextStation);
                vehicle.setNextStation(vehicleArrival.getStationId());

            }  else if (nextStationIsArtificialStation & vehicleArrivalBeforeGeneratingNewRoutes ) {
                int vehicleId = vehicleArrival.getVehicle();
                Vehicle vehicle = vehicles.get(vehicleId);
                vehicle.setTimeToNextStation(0);
                vehicle.setNextStation(vehicleArrival.getStationId());
            }
        }

        //Find out if multiple vehicles have same initial station
        ArrayList<Integer> initialStationIds = new ArrayList<>();

        for (Vehicle vehicle : vehicles.values()) {
            int initialStationId = vehicle.getNextStation();

            if (initialStationIds.contains(initialStationId)) {
                //Change initial station
                boolean visitAPickUpStation = stations.get(initialStationId).getNetDemand(TimeConverter.convertMinutesToHourRounded(currentMinute)) > 0;
                int newStation = vehicle.getNextStationInitial();
                for (Station station : stations.values()) {
                    boolean stationIsPickUpStation = station.getNetDemand(TimeConverter.convertMinutesToHourRounded(currentMinute)) > 0;
                    if (stationIsPickUpStation && visitAPickUpStation && !initialStationIds.contains(station.getId())) {
                        newStation = station.getId();
                        initialStationIds.add(newStation);
                        break;
                    } else if (!stationIsPickUpStation && !visitAPickUpStation && !initialStationIds.contains(station.getId())) {
                        newStation = station.getId();
                        initialStationIds.add(newStation);
                        break;
                    }
                }
                vehicle.setNextStation(newStation);

            } else {
                initialStationIds.add(initialStationId);
            }
        }

    }

















    //Getters and setters

    public double getNumberOfTimesVehicleRouteGenerated() {
        return numberOfTimesVehicleRouteGenerated;
    }

    public void setNumberOfTimesVehicleRouteGenerated(double numberOfTimesVehicleRouteGenerated) {
        this.numberOfTimesVehicleRouteGenerated = numberOfTimesVehicleRouteGenerated;
    }
    public double getCongestions() {
        return congestions;
    }

    public void setCongestions(double congestions) {
        this.congestions = congestions;
    }

    public double getStarvations() {
        return starvations;
    }

    public void setStarvations(double starvations) {
        this.starvations = starvations;
    }

    public int getTotalNumberOfCustomers() {
        return totalNumberOfCustomers;
    }

    public void setTotalNumberOfCustomers(int totalNumberOfCustomers) {
        this.totalNumberOfCustomers = totalNumberOfCustomers;
    }

    public int getHappyCustomers() {
        return happyCustomers;
    }

    public void setHappyCustomers(int happyCustomers) {
        this.happyCustomers = happyCustomers;
    }

    public ArrayList<Double> getTimeToNextSimulationList() {
        return timeToNextSimulationList;
    }

    public void setTimeToNextSimulationList(ArrayList<Double> timeToNextSimulationList) {
        this.timeToNextSimulationList = timeToNextSimulationList;
    }

    public ArrayList<Double> getComputationalTimesXpress() {
        return computationalTimesXpress;
    }

    public void setComputationalTimesXpress(ArrayList<Double> computationalTimesXpress) {
        this.computationalTimesXpress = computationalTimesXpress;
    }

    public ArrayList<Double> getComputationalTimesXpressPlusInitialization() {
        return computationalTimesXpressPlusInitialization;
    }

    public void setComputationalTimesXpressPlusInitialization(ArrayList<Double> computationalTimesXpressPlusInitialization) {
        this.computationalTimesXpressPlusInitialization = computationalTimesXpressPlusInitialization;
    }

    public double getMaxComputationalTimeIncludingInitialization() {
        return maxComputationalTimeIncludingInitialization;
    }

    public void setMaxComputationalTimeIncludingInitialization(double maxComputationalTimeIncludingInitialization) {
        this.maxComputationalTimeIncludingInitialization = maxComputationalTimeIncludingInitialization;
    }

    public double getMinComputationalTimeIncludingInitialization() {
        return minComputationalTimeIncludingInitialization;
    }

    public void setMinComputationalTimeIncludingInitialization(double minComputationalTimeIncludingInitialization) {
        this.minComputationalTimeIncludingInitialization = minComputationalTimeIncludingInitialization;
    }

    public ArrayList<Double> getNumberOfTimesPPImproved() {
        return numberOfTimesPPImproved;
    }

    public void setNumberOfTimesPPImproved(ArrayList<Double> numberOfTimesPPImproved) {
        this.numberOfTimesPPImproved = numberOfTimesPPImproved;
    }
}






















