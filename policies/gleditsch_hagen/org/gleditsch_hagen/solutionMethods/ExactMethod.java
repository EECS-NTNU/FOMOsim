package org.gleditsch_hagen.solutionMethods;

import org.gleditsch_hagen.classes.Input;
import org.gleditsch_hagen.classes.StopWatch;
import com.dashoptimization.XPRMCompileException;
import org.gleditsch_hagen.enums.SolutionMethod;
import org.gleditsch_hagen.xpress.RunXpress;
import org.gleditsch_hagen.xpress.WriteXpressFiles;

import java.io.IOException;

public class ExactMethod {

    private double computationalTime;

    //Constructor
    public ExactMethod(Input input) throws IOException, XPRMCompileException {

        //Start timer
        StopWatch stopWatch = new StopWatch();
        stopWatch.start();

        WriteXpressFiles.printTimeDependentInput(input, SolutionMethod.EXACT_METHOD);
        RunXpress.runXpress(input.getXpressFile());

        stopWatch.stop();

        this.computationalTime = stopWatch.getElapsedTimeSecs();
    }

    //Getters and setters
    public double getComputationalTime() {
        return computationalTime;
    }

    public void setComputationalTime(double computationalTime) {
        this.computationalTime = computationalTime;
    }

}
