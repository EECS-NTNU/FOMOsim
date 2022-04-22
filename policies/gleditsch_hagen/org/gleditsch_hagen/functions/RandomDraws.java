package org.gleditsch_hagen.functions;

import java.util.Random;


public class RandomDraws {

    public static double drawNormal(double mean, double standardDeviation) {
        Random r = new Random();
        return (r.nextGaussian() * standardDeviation) + mean;
    }

    //Draw a random arrival VisitTime between startTime and startTime+1
    public static double drawArrivalTimes (double startTime){
        return TimeConverter.convertHourtoSeconds(Math.random()+startTime);
    }

}
