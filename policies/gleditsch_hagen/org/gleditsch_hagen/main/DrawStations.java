
package org.gleditsch_hagen.main;


import org.gleditsch_hagen.classes.GraphViewer;
import org.gleditsch_hagen.classes.Input;
import org.gleditsch_hagen.classes.Station;
import org.gleditsch_hagen.functions.TimeConverter;

import java.io.IOException;

public class DrawStations {

    public static void main(String[] args) throws IOException {

        Input input = new Input();

        //Low : 0-mediumDemand, Medium: mediumDemand-highDemand, High: highDemand++
        double highDemand = input.getHighDemand();
        double mediumDemand = input.getMediumDemand();

        GraphViewer graph = new GraphViewer();
        graph.drawStationDemand(input, mediumDemand, highDemand);


        int nrOfHighCriticalStations = 0;
        int nrOfMediumCriticalStations = 0;
        int nrOfLowCriticalStations = 0;

        for (Station station : input.getStations().values()) {
            double netDemand = station.getNetDemand(TimeConverter.convertMinutesToHourRounded(input.getCurrentMinute()));
            if (netDemand >= highDemand || netDemand <= -highDemand) {
                nrOfHighCriticalStations++;
            } else if (netDemand >= mediumDemand || netDemand <= -mediumDemand) {
                nrOfMediumCriticalStations++;
            } else {
                nrOfLowCriticalStations++;
            }
        }

        System.out.println("Number of stations with high demand: " + nrOfHighCriticalStations);
        System.out.println("Number of stations with medium demand: " + nrOfMediumCriticalStations);
        System.out.println("Number of stations with low demand: " + nrOfLowCriticalStations);
    }
}

