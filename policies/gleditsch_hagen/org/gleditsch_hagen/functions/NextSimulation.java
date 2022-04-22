package org.gleditsch_hagen.functions;

import org.gleditsch_hagen.classes.*;
import org.gleditsch_hagen.enums.ReOptimizationMethod;

import java.util.ArrayList;

public class NextSimulation {

    public static double determineTimeToNextSimulation(ArrayList<VehicleArrival> vehicleArrivals, double timeHorizon, ReOptimizationMethod reOptimizationMethod, double currentTime) {
        double firstVisit = timeHorizon + currentTime;
        double secondVisit = timeHorizon + currentTime;
        double thirdVisit = timeHorizon + currentTime;

        for (VehicleArrival vehicleArrival : vehicleArrivals) {
            double time = vehicleArrival.getTime();

            if (time > currentTime & time < firstVisit) {
                firstVisit = time;
            }
            if (time > currentTime & time < secondVisit & time > firstVisit) {
                secondVisit = time;
            }
            if (time > currentTime & time < thirdVisit & time > secondVisit) {
                thirdVisit = time;
            }

        }

        double nextSimulation = 0;

        switch (reOptimizationMethod) {
            case EVERY_VEHICLE_ARRIVAL:
                nextSimulation = firstVisit;
                break;
            case EVERY_SECOND_VEHICLE_ARRIVAL:
                nextSimulation = secondVisit;
                break;
            case EVERY_THIRD_VEHICLE_ARRIVAL:
                nextSimulation = thirdVisit;
                break;
            case TEN_MIN:
                nextSimulation = currentTime + 10;
                break;
            case TWENTY_MIN:
                nextSimulation = currentTime + 20;
                break;
            case THIRTY_MIN:
                nextSimulation = currentTime + 30;
                break;
        }

        return nextSimulation;
    }
}
