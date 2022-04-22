package org.gleditsch_hagen.main;

import org.gleditsch_hagen.classes.Station;
import java.io.*;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Scanner;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.apache.poi.ss.usermodel.Row;
import org.json.JSONArray;
import org.json.JSONObject;


public class DrivingTimeMatrix {

   static String initialFile = "stationCoordinates.txt";
   static ArrayList<Station> stations = new ArrayList<>();

    public DrivingTimeMatrix() throws FileNotFoundException {
    }


    public static void main(String[] args) throws IOException, org.json.JSONException {

        lookUpCoordinates();
        getDrivingTimes();
        System.out.println("Driving Matrix successfully created");
    }

    public static void lookUpCoordinates() throws FileNotFoundException {
        File inputFile = new File(initialFile);
        Scanner in = new Scanner(inputFile);
        while (in.hasNextLine()){
            String line = in.nextLine();
            Scanner element = new Scanner(line);
            if (element.hasNextInt()) {
                Station station = new Station(element.nextInt());
                station.setLatitude(Double.parseDouble(element.next()));
                station.setLongitude(Double.parseDouble(element.next()));
                stations.add(station);
            }
        }
        in.close();
    }

    public static void getDrivingTimes() throws IOException, org.json.JSONException {
        //Oppretter nytt Excell-ark
        XSSFWorkbook workbook = new XSSFWorkbook();
        XSSFSheet sheet = workbook.createSheet("Driving Times");
        int numberOfQueries10sek = 0;

        //Start fra input
        Row headerRow = sheet.createRow(0);
        int rowNumber = 1;
        for (Station origin: stations) {
            Row row = sheet.createRow(rowNumber);
            row.createCell(0).setCellValue(origin.getId()); //print vertikal stasjonsID
            headerRow.createCell(rowNumber).setCellValue(origin.getId()); //print horisontal stasjonsID
            int cellNumber = 1;
            for (Station destination: stations) {
                if (origin.getId() != destination.getId()) {
                    if (numberOfQueries10sek > 99) {
                        try {
                            System.out.println("Execution sleeps for 10 seconds");
                            Thread.currentThread().sleep(10*1000);
                            numberOfQueries10sek = 0;

                        }
                        catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }

                    int drivingTimeSek = getDrivingTimeBetweenCoordinates(origin, destination);
                    numberOfQueries10sek++;
                    double drivingTimeMin = ((double) drivingTimeSek)/60;
                    row.createCell(cellNumber).setCellValue(drivingTimeMin);
                    cellNumber++;

                } else {
                    row.createCell(cellNumber).setCellValue(0);
                    cellNumber++;
                }
            }
            rowNumber++;
        }

        FileOutputStream fileOut = new FileOutputStream("DrivingTimeMatrix.xlsx");
        workbook.write(fileOut);
        fileOut.close();
    }







    private static int getDrivingTimeBetweenCoordinates(Station origin, Station destination) throws IOException, org.json.JSONException {
        double originLongitude = origin.getLongitude();
        double originLatitude = origin.getLatitude();
        double destinationLongitude = destination.getLongitude();
        double destinationLatitude = destination.getLatitude();
        String urlRequestGoogleMaps = "http://maps.googleapis.com/maps/api/distancematrix/json?origins=" + originLatitude + "," + originLongitude + "&destinations=" + destinationLatitude + "," + destinationLongitude + "&mode=driving&language=en-EN&sensor=false";
        JSONObject json = readJsonFromUrl(urlRequestGoogleMaps);
        return getDrivingDurationAsInt(json);
    }

    private static JSONObject readJsonFromUrl(String urlRequestGoogleMaps) throws IOException, org.json.JSONException {
        InputStream is = new URL(urlRequestGoogleMaps).openStream();
        try {
            BufferedReader rd = new BufferedReader(new InputStreamReader(is, Charset.forName("UTF-8")));
            String jsonText = readAll(rd);
            JSONObject json = new JSONObject(jsonText);
            return json;
        } finally {
            is.close();
        }
    }

    private static String readAll(Reader rd) throws IOException {
        StringBuilder sb = new StringBuilder();
        int cp;
        while ((cp = rd.read()) != -1) {
            sb.append((char) cp);
        }
        return sb.toString();
    }

    private static int getDrivingDurationAsInt(JSONObject json) throws org.json.JSONException {
        JSONArray rows = json.getJSONArray("rows");
        JSONObject firstObjectRows = rows.getJSONObject(0);
        JSONArray elements = firstObjectRows.getJSONArray("elements");
        JSONObject firstObjectElements = elements.getJSONObject(0);
        JSONObject duration = firstObjectElements.getJSONObject("duration");
        return Integer.parseInt(duration.get("value").toString());
    }


}
