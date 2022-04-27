package org.gleditsch_hagen.classes;


import org.gleditsch_hagen.enums.SolutionMethod;
import org.gleditsch_hagen.functions.TimeConverter;
import org.gleditsch_hagen.xpress.ReadXpressResult;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class PricingProblem {


    public PricingProblem(){
    }

    public void setPricingProblemScore(Input input, HashMap<Integer, Double> pricingProblemScores) throws FileNotFoundException {
        pricingProblemScores.clear();

        ArrayList<VehicleArrival> vehicleArrivals;
        if (input.getSolutionMethod().equals(SolutionMethod.HEURISTIC_VERSION_3)) {
            vehicleArrivals = ReadXpressResult.readVehicleArrivalsVersion3(input.getVehicles(), input.getCurrentMinute());
        } else if (input.getSolutionMethod().equals(SolutionMethod.HEURISTIC_VERSION_1) || input.getSolutionMethod().equals(SolutionMethod.HEURISTIC_VERSION_2) || input.getSolutionMethod().equals(SolutionMethod.EXACT_METHOD)) {
            vehicleArrivals = ReadXpressResult.readVehicleArrivals(input.getCurrentMinute());
        } else {
            vehicleArrivals = null;
        }

        for (Station station: input.getStations().values()) {

            //Checks if station is visited by any service vehicles
            boolean visited = false;
            for (VehicleArrival vehicleArrival : vehicleArrivals) {
                if (vehicleArrival.getStationId() == station.getId()) {
                    visited = true;
                }
            }

            if (!visited) {
                double ViolationsIfNoVisit = 0;
                double DeviationsIfNoVisit;
                double initialLoad = station.getLoad();
                double demandPerMinute = station.getNetDemand(TimeConverter.convertMinutesToHourRounded(input.getCurrentMinute()))/60;
                double loadAtHorizon = initialLoad + demandPerMinute*input.getTimeHorizon();
                double optimalState = station.getOptimalState(TimeConverter.convertMinutesToHourRounded(input.getCurrentMinute()));

                if (loadAtHorizon > station.getCapacity()) {
                    ViolationsIfNoVisit = loadAtHorizon-station.getCapacity();
                    loadAtHorizon = station.getCapacity();
                }
                else if (loadAtHorizon < 0) {
                    ViolationsIfNoVisit = -loadAtHorizon;
                    loadAtHorizon = 0;
                }

                DeviationsIfNoVisit = Math.abs(optimalState-loadAtHorizon);

                pricingProblemScores.put(station.getId(), ViolationsIfNoVisit + DeviationsIfNoVisit);
            }

        }
    }
}
