package org.gleditsch_hagen.main;


import org.gleditsch_hagen.classes.GraphViewer;
import org.gleditsch_hagen.classes.Input;
import org.gleditsch_hagen.classes.Station;
import org.gleditsch_hagen.classes.Vehicle;
import com.dashoptimization.XPRMCompileException;
import org.gleditsch_hagen.enums.SolutionMethod;
import org.gleditsch_hagen.functions.ReadClusterList;
import org.gleditsch_hagen.xpress.RunXpress;
import org.gleditsch_hagen.xpress.WriteXpressFiles;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class GenerateCluster {


    public static void main(String[] args) throws IOException, XPRMCompileException {
        Input input = new Input();

        WriteXpressFiles.writeClusterInformation(input);
        RunXpress.runXpress("createCluster");


        readCluster(input);
        GraphViewer graph = new GraphViewer();
        graph.drawClusters(input);

    }

    private static void readCluster(Input input) throws IOException {

        if (input.getSolutionMethod() == SolutionMethod.CURRENT_SOLUTION_IN_OSLO) {
            ReadClusterList.readClusterListExcel(input, "clusterCurrentSolution.xlsx");

        } else if (input.getSolutionMethod() == SolutionMethod.HEURISTIC_VERSION_1 || input.getSolutionMethod() == SolutionMethod.HEURISTIC_VERSION_2 || input.getSolutionMethod() == SolutionMethod.HEURISTIC_VERSION_3) {

            String xpressOutputFile = "clusterOutput-Instance" + input.getTestInstance() + "-V" + input.getVehicles().size()+".txt";
            ReadClusterList.readClusterListTextFile(input, xpressOutputFile);

        }

    }

}
