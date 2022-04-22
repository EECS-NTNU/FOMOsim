package org.gleditsch_hagen.functions;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class ReadStationInitialState {

    public static ArrayList<Integer> readStationInitialState(String textfile) throws FileNotFoundException {
        ArrayList<Integer> stationIdList = new ArrayList<>();

        File inputfile = new File(textfile);
        Scanner in = new Scanner(inputfile);

        while (in.hasNextLine()) {
            String line = in.nextLine();
            Scanner element = new Scanner(line).useDelimiter("\\s*,\\s*");
            if (element.hasNextInt()) {
                int stationId = element.nextInt();
                stationIdList.add(stationId);
            }
            element.close();
        }
        in.close();
        return(stationIdList);
    }
}
