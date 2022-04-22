package org.gleditsch_hagen.functions;

import static java.lang.StrictMath.floor;

public class TimeConverter {

    public static double convertHourtoSeconds(double hour) {
        return hour*60*60;
    }

    public static double convertSecondsToHourRounded(double seconds){
        return floor(seconds/60/60);
    }

    public static double convertMinutesToHourRounded(double seconds){
        return floor(seconds/60);
    }

    public static double convertSecondsToHour(double seconds){
        return seconds/60/60;
    }

}
