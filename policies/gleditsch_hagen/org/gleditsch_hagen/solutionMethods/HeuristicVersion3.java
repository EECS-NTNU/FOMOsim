package org.gleditsch_hagen.solutionMethods;

import org.gleditsch_hagen.classes.Input;
import org.gleditsch_hagen.classes.PricingProblem;
import org.gleditsch_hagen.classes.StopWatch;
import org.gleditsch_hagen.classes.Vehicle;
import org.gleditsch_hagen.enums.SolutionMethod;
import com.dashoptimization.XPRMCompileException;
import org.gleditsch_hagen.xpress.ReadXpressResult;
import org.gleditsch_hagen.xpress.RunXpress;
import org.gleditsch_hagen.xpress.WriteXpressFiles;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.HashMap;

public class HeuristicVersion3 {

    private double computationalTimeXpress;
    private double computationalTimeIncludingInitialization;
    private double numberOfTimesObjectiveImproved = 0;


    //Constructor
    public HeuristicVersion3(Input input) throws IOException, XPRMCompileException {

        //Start timer
        StopWatch stopWatchIncludingInitialization = new StopWatch();
        stopWatchIncludingInitialization.start();

        HashMap<Integer, Double> pricingProblemScores = new HashMap<>();
        initiateRoutes(input, pricingProblemScores);
        WriteXpressFiles.printTimeDependentInput(input, SolutionMethod.HEURISTIC_VERSION_3);

        //Start timer and run Xpress
        StopWatch stopWatchXpress = new StopWatch();
        stopWatchXpress.start();

        RunXpress.runXpress(input.getXpressFile());

        double objectiveValue = ReadXpressResult.readObjectiveValue();


        //Run pricing problem
        if (input.isRunPricingProblem()) {
            input.setNowRunningPricingProblem(true);
            int initialBranchingConstant = input.getNrStationBranching();
            input.setNrStationBranching(input.getNrOfBranchingPricingProblem());

            for (int i = 0; i < input.getNrOfRunsPricingProblem(); i ++) {

                runPricingProblem(input, pricingProblemScores);
                initiateRoutes(input, pricingProblemScores);
                WriteXpressFiles.printTimeDependentInput(input, SolutionMethod.HEURISTIC_VERSION_3);
                RunXpress.runXpress(input.getXpressFile());

                double objectiveValueAfterPricingProblem = ReadXpressResult.readObjectiveValue();
                if (objectiveValueAfterPricingProblem <= objectiveValue) {
                    objectiveValue = objectiveValueAfterPricingProblem;
                    this.numberOfTimesObjectiveImproved ++;
                }
            }

            input.setNrStationBranching(initialBranchingConstant);
            input.setNowRunningPricingProblem(false);
        }

        stopWatchIncludingInitialization.stop();
        stopWatchXpress.stop();

        this.computationalTimeXpress = stopWatchXpress.getElapsedTimeSecs();
        this.computationalTimeIncludingInitialization = stopWatchIncludingInitialization.getElapsedTimeSecs();

    }


    private void runPricingProblem(Input input, HashMap<Integer, Double> pricingProblemScores) throws FileNotFoundException {
        PricingProblem pricingProblem = new PricingProblem();
        pricingProblem.setPricingProblemScore(input, pricingProblemScores);
    }

    private static void initiateRoutes(Input input, HashMap<Integer, Double> pricingProblemScores) throws FileNotFoundException, UnsupportedEncodingException {

        //Initialize routes for each vehicle
        for (Vehicle vehicle: input.getVehicles().values()) {
            vehicle.createRoutes(input, pricingProblemScores);
        }


    }

    public double getComputationalTimeXpress() {
        return computationalTimeXpress;
    }

    public void setComputationalTimeXpress(double computationalTimeXpress) {
        this.computationalTimeXpress = computationalTimeXpress;
    }

    public double getComputationalTimeIncludingInitialization() {
        return computationalTimeIncludingInitialization;
    }

    public void setComputationalTimeIncludingInitialization(double computationalTimeIncludingInitialization) {
        this.computationalTimeIncludingInitialization = computationalTimeIncludingInitialization;
    }

    public double getNumberOfTimesObjectiveImproved() {
        return numberOfTimesObjectiveImproved;
    }

    public void setNumberOfTimesObjectiveImproved(double numberOfTimesObjectiveImproved) {
        this.numberOfTimesObjectiveImproved = numberOfTimesObjectiveImproved;
    }
}
