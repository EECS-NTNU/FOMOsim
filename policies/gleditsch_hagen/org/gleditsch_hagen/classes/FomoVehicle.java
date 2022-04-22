package org.gleditsch_hagen.classes;

import java.util.ArrayList;

public class FomoVehicle {
  public int id;
  public int capacity;
  public int currentStation;
  public ArrayList<Integer> bikes = new ArrayList<>();

  public FomoVehicle(int id, int capacity, int currentStation) {
    this.id = id;
    this.capacity = capacity;
    this.currentStation = currentStation;
  }
}
