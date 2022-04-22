package org.gleditsch_hagen.classes;

import java.util.ArrayList;
import java.util.HashMap;

public class FomoStation {
  public int id;
  public ArrayList<Integer> bikes = new ArrayList<>();
  public int capacity;
  public double leaveIntensity;
  public double arriveIntensity;
  public int idealState;
  public HashMap<Long, Double> distances = new HashMap<>();

  public FomoStation(int id, int capacity, double leaveIntensity, double arriveIntensity, int idealState) {
    this.id = id;
    this.capacity = capacity;
    this.idealState = idealState;
    this.leaveIntensity = leaveIntensity;
    this.arriveIntensity = arriveIntensity;
  }
}
