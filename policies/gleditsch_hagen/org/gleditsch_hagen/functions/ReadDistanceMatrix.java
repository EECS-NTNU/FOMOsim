package org.gleditsch_hagen.functions;

import org.gleditsch_hagen.classes.Station;
//import main.DrivingTimeMatrix;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.CellType;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Scanner;

public class ReadDistanceMatrix {

    public static void lookUpDrivingTimes(HashMap<Integer, Station> stations, ArrayList<Integer> stationIDlist) throws IOException {
        FileInputStream excelFile = new FileInputStream(new File("DrivingTimeMatrix.xlsx"));
        XSSFWorkbook workbook = new XSSFWorkbook(excelFile);
        Sheet datatypeSheet = workbook.getSheetAt(0);
        Row headerRow = datatypeSheet.getRow(0);


        for (Row row: datatypeSheet) {

            int originId = (int) row.getCell(0).getNumericCellValue();
            if (stationIDlist.contains(originId)) {

                for (Cell cell: row) {
                    int destinationId = (int) headerRow.getCell(cell.getColumnIndex()).getNumericCellValue();
                    if (stationIDlist.contains(destinationId)) {
                        stations.get(originId).addDistanceToStationHashmap(destinationId, cell.getNumericCellValue());
                    }
                }
            }
        }


    }
}
