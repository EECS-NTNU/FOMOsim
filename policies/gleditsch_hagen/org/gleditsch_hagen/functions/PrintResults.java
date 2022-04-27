package org.gleditsch_hagen.functions;

import org.gleditsch_hagen.classes.Input;
import org.gleditsch_hagen.enums.SolutionMethod;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.*;
import java.util.ArrayList;

public class PrintResults {

    public static void printSimulationResultsToExcelFile(double averageViolation, double averagePercentageViolation, ArrayList<Double> percentageViolationsList, double sdViolations, double sdPercentageViolation,
                                                         double averageNumberOfTimesVehicleRouteGenerated, double avergageTimeToVehicleRouteGenerated,
                                                         double averageComputationalTimeXpress, double averageComputationalTimeXpressPlusInitialization, Input input,
                                                         double averageTimePPImprovement) throws IOException {

        System.out.println("averageComputationalTimeXpress: " + averageComputationalTimeXpress);
        System.out.println("averageComputationalTimeXpressPlusInitialization" + averageComputationalTimeXpressPlusInitialization);


        //Read the spreadsheet that needs to be updated
        FileInputStream fileInputStream= new FileInputStream(new File("Results.xlsx"));
        XSSFWorkbook workbook = new XSSFWorkbook(fileInputStream);
        XSSFSheet worksheet = workbook.getSheetAt(0);

        int rowNumber = 0;
        boolean findEmptyRow = true;

        while (findEmptyRow) {
            if (worksheet.getRow(rowNumber) == null) {
                findEmptyRow = false;
            } else {
                rowNumber ++;
            }
        }

        //Print output to file
        Row rowOutput = worksheet.createRow(rowNumber);


        boolean heuristic1 = input.getSolutionMethod().equals(SolutionMethod.HEURISTIC_VERSION_1);
        boolean heuristic2 = input.getSolutionMethod().equals(SolutionMethod.HEURISTIC_VERSION_2);
        boolean heuristic3 = input.getSolutionMethod().equals(SolutionMethod.HEURISTIC_VERSION_3);
        boolean exact = input.getSolutionMethod().equals(SolutionMethod.EXACT_METHOD);
        boolean current = input.getSolutionMethod().equals(SolutionMethod.CURRENT_SOLUTION_IN_OSLO);
        boolean noVehicles = input.getSolutionMethod().equals(SolutionMethod.NO_VEHICLES);
        boolean allHeuristics = heuristic1 || heuristic2 || heuristic3;



        //Results
        if (exact || allHeuristics) {
            rowOutput.createCell(0).setCellValue(averageComputationalTimeXpress);
            rowOutput.createCell(1).setCellValue(averageComputationalTimeXpressPlusInitialization);
            rowOutput.createCell(3).setCellValue(averageNumberOfTimesVehicleRouteGenerated);
            rowOutput.createCell(4).setCellValue(avergageTimeToVehicleRouteGenerated);
        }
        rowOutput.createCell(2).setCellValue(averagePercentageViolation);

        if (allHeuristics) {
            //Weights - Criticality score
            rowOutput.createCell(5).setCellValue(input.getWeightCritScTimeToViolation());
            rowOutput.createCell(6).setCellValue(input.getWeightCritScViolationRate());
            rowOutput.createCell(7).setCellValue(input.getWeightCritScDrivingTime());
            rowOutput.createCell(8).setCellValue(input.getWeightCritScOptimalState());
            rowOutput.createCell(9).setCellValue(input.getWeightPricingProblemScore());
        }else if (current) {
            //Weights - Criticality score
            rowOutput.createCell(5).setCellValue(input.getWeightCritScTimeToViolationCurrent());
            rowOutput.createCell(6).setCellValue(input.getWeightCritScViolationRateCurrent());
            rowOutput.createCell(7).setCellValue(input.getWeightCritScDrivingTimeCurrent());
            rowOutput.createCell(8).setCellValue(input.getWeightCritScOptimalStateCurrent());
            rowOutput.createCell(9).setCellValue(0);
        }

        //Weights - Objective function
        if (!(current || noVehicles)) {
            rowOutput.createCell(10).setCellValue(input.getWeightViolation());
            rowOutput.createCell(11).setCellValue(input.getWeightDeviation());
            rowOutput.createCell(12).setCellValue(input.getWeightReward());
            rowOutput.createCell(13).setCellValue(input.getWeightDeviationReward());
            rowOutput.createCell(14).setCellValue(input.getWeightDrivingTimePenalty());
        }

        //Weights - Clustering
        if (allHeuristics && input.isClustering()) {
            rowOutput.createCell(15).setCellValue(input.getWeightClusterDrivingTime());
            rowOutput.createCell(16).setCellValue(input.getWeightClusterNetDemand());
            rowOutput.createCell(17).setCellValue(input.getWeightClusterEqualSize());
        }


        //Loading interval
        rowOutput.createCell(18).setCellValue(input.getMinLoad());
        rowOutput.createCell(19).setCellValue(input.getMaxLoad());

        //Input
        rowOutput.createCell(20).setCellValue(input.getSolutionMethod().toString());
        if (!noVehicles) {
            rowOutput.createCell(21).setCellValue(input.getReOptimizationMethod().toString());
            rowOutput.createCell(22).setCellValue(input.getMaxVisit());
            rowOutput.createCell(23).setCellValue(input.getTimeHorizon());
        }
        rowOutput.createCell(24).setCellValue(input.getSimulationStartTime()/60);
        rowOutput.createCell(25).setCellValue(input.getSimulationStopTime()/60);
        rowOutput.createCell(26).setCellValue(input.getTestInstance());
        if (!noVehicles) {
            rowOutput.createCell(27).setCellValue(input.getNumberOfVehicles());
        }
        if (allHeuristics) {
            rowOutput.createCell(28).setCellValue(input.getNrStationBranching());
            rowOutput.createCell(31).setCellValue(input.getTresholdLengthRoute());
        }
        if (heuristic2) {
            rowOutput.createCell(29).setCellValue(input.getLoadInterval());
        }
        rowOutput.createCell(30).setCellValue(input.getNumberOfRuns());

        //Clustering
        if (allHeuristics) {
            rowOutput.createCell(32).setCellValue(input.isClustering());
            if (input.isClustering()) {
                rowOutput.createCell(33).setCellValue(input.isDynamicClustering());
                rowOutput.createCell(34).setCellValue(input.getHighDemand());
                rowOutput.createCell(35).setCellValue(input.getMediumDemand());
            }
        }

        //Pricing problem
        if (allHeuristics) {
            rowOutput.createCell(36).setCellValue(input.isRunPricingProblem());
            if (input.isRunPricingProblem()) {
                rowOutput.createCell(37).setCellValue(input.getNrOfRunsPricingProblem());
                rowOutput.createCell(38).setCellValue(input.getNrOfBranchingPricingProblem());
                rowOutput.createCell(39).setCellValue(input.getProbabilityOfChoosingUnvisitedStation());
            }
        }


        //Print results for statistical t-test
        for (int i = 40; i < input.getNumberOfRuns()+ 40 ; i++) {
            rowOutput.createCell(i).setCellValue(percentageViolationsList.get(i-40));
        }

        if (heuristic3 && input.isRunPricingProblem()) {
            rowOutput.createCell(50).setCellValue(averageTimePPImprovement);
        }


        fileInputStream.close();

        FileOutputStream fileOut = new FileOutputStream("Results.xlsx");
        workbook.write(fileOut);
        fileOut.close();


        System.out.println("Printet resultater");

    }







    public static void printOneRouteResultsToExcelFile(Input input, double objectiveValue, double computationalTimeXpress, double computationalTimeIncludingInitialization) throws IOException {

        System.out.println("Objective value: " + objectiveValue);
        System.out.println("Computational time Xpress: " + computationalTimeXpress);
        System.out.println("Computational time including initialization: " + computationalTimeIncludingInitialization);
        System.out.println();


        //Read the spreadsheet that needs to be updated
        FileInputStream fileInputStream= new FileInputStream(new File("Results.xlsx"));
        XSSFWorkbook workbook = new XSSFWorkbook(fileInputStream);
        XSSFSheet worksheet = workbook.getSheetAt(1);

        int rowNumber = 0;
        boolean findEmptyRow = true;

        while (findEmptyRow) {
            if (worksheet.getRow(rowNumber) == null) {
                findEmptyRow = false;
            } else {
                rowNumber ++;
            }
        }


        //Print output to file
        Row rowOutput = worksheet.createRow(rowNumber);

        boolean heuristic1 = input.getSolutionMethod().equals(SolutionMethod.HEURISTIC_VERSION_1);
        boolean heuristic2 = input.getSolutionMethod().equals(SolutionMethod.HEURISTIC_VERSION_2);
        boolean heuristic3 = input.getSolutionMethod().equals(SolutionMethod.HEURISTIC_VERSION_3);
        boolean exact = input.getSolutionMethod().equals(SolutionMethod.EXACT_METHOD);
        boolean noVehicles = input.getSolutionMethod().equals(SolutionMethod.NO_VEHICLES);
        boolean allHeuristics = heuristic1 || heuristic2 || heuristic3;



        //Results
        if (exact || allHeuristics) {
            rowOutput.createCell(0).setCellValue(computationalTimeXpress);
            rowOutput.createCell(1).setCellValue(computationalTimeIncludingInitialization);
        }
        rowOutput.createCell(2).setCellValue(objectiveValue);

        if (allHeuristics) {
            //Weights - Criticality score
            rowOutput.createCell(3).setCellValue(input.getWeightCritScTimeToViolation());
            rowOutput.createCell(4).setCellValue(input.getWeightCritScViolationRate());
            rowOutput.createCell(5).setCellValue(input.getWeightCritScDrivingTime());
            rowOutput.createCell(6).setCellValue(input.getWeightCritScOptimalState());
            rowOutput.createCell(7).setCellValue(input.getWeightPricingProblemScore());
        }

        //Weights - Objective function
        if (!(noVehicles)) {
            rowOutput.createCell(8).setCellValue(input.getWeightViolation());
            rowOutput.createCell(9).setCellValue(input.getWeightDeviation());
            rowOutput.createCell(10).setCellValue(input.getWeightReward());
            rowOutput.createCell(11).setCellValue(input.getWeightDeviationReward());
            rowOutput.createCell(12).setCellValue(input.getWeightDrivingTimePenalty());
        }

        //Weights - Clustering
        if (allHeuristics && input.isClustering()) {
            rowOutput.createCell(13).setCellValue(input.getWeightClusterNetDemand());
            rowOutput.createCell(14).setCellValue(input.getWeightClusterDrivingTime());
            rowOutput.createCell(15).setCellValue(input.getWeightClusterEqualSize());
        }

        //Loading interval
        rowOutput.createCell(16).setCellValue(input.getMinLoad());
        rowOutput.createCell(17).setCellValue(input.getMaxLoad());

        //Input
        rowOutput.createCell(18).setCellValue(input.getSolutionMethod().toString());
        if (!noVehicles) {
            rowOutput.createCell(19).setCellValue(input.getMaxVisit());
            rowOutput.createCell(20).setCellValue(input.getTimeHorizon());
            rowOutput.createCell(22).setCellValue(input.getNumberOfVehicles());
        }
        rowOutput.createCell(21).setCellValue(input.getTestInstance());
        if (allHeuristics) {
            rowOutput.createCell(23).setCellValue(input.getNrStationBranching());
            rowOutput.createCell(25).setCellValue(input.getTresholdLengthRoute());
        }
        if (heuristic2) {
            rowOutput.createCell(24).setCellValue(input.getLoadInterval());
        }


        if (allHeuristics) {
            //Clustering
            rowOutput.createCell(26).setCellValue(input.isClustering());
            if (input.isClustering()) {
                rowOutput.createCell(27).setCellValue(input.getHighDemand());
                rowOutput.createCell(28).setCellValue(input.getMediumDemand());
            }
        }

        if (allHeuristics) {
            //Pricing problem
            rowOutput.createCell(29).setCellValue(input.isRunPricingProblem());
            if (input.isRunPricingProblem()) {
                rowOutput.createCell(30).setCellValue(input.getNrOfRunsPricingProblem());
                rowOutput.createCell(31).setCellValue(input.getNrOfBranchingPricingProblem());
                rowOutput.createCell(32).setCellValue(input.getProbabilityOfChoosingUnvisitedStation());
            }
        }

        rowOutput.createCell(33).setCellValue(input.getCurrentMinute()/60);

        fileInputStream.close();

        FileOutputStream fileOut = new FileOutputStream("Results.xlsx");
        workbook.write(fileOut);
        fileOut.close();

    }


}




















