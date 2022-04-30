package org.gleditsch_hagen.classes;

import java.util.ArrayList;
import java.util.HashMap;

public class FomoStation {
  public long id;
  public ArrayList<Long> bikes = new ArrayList<>();
  public long capacity;
  public double leaveIntensity;
  public double arriveIntensity;
  public long idealState;
  public HashMap<Long, Double> distances = new HashMap<>();

  public FomoStation(long id, long capacity, double leaveIntensity, double arriveIntensity, long idealState) {
    this.id = id;
    this.capacity = capacity;
    this.idealState = idealState;
    this.leaveIntensity = leaveIntensity;
    this.arriveIntensity = arriveIntensity;
  }
}
