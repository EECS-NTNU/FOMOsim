package org.gleditsch_hagen.xpress;

import com.dashoptimization.*;
import java.lang.*;
import java.util.ArrayList;
import java.io.*;

public class RunXpress {

    public static void runXpress(String moselFile) throws XPRMCompileException, IOException {
        XPRM mosel;
        XPRMModel mod;

        //Initialize model
        mosel = new XPRM();

        //Compile model
        mosel.compile(moselFile+ ".mos");

        //Load bim file
        mod = mosel.loadModel(moselFile + ".bim");

        //Execute model
        //System.out.println("Run " + moselFile + ".mos");
        mod.run();

        //Stop if no solution is found
        if(mod.getProblemStatus()!=mod.PB_OPTIMAL) {
            System.out.println("NO SOLUTION FOUND");
            System.exit(1);
        }
    }

}
