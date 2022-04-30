package org.gleditsch_hagen.classes;

import java.util.ArrayList;

public class FomoVehicle {
  public long id;
  public long capacity;
  public long currentStation;
  public ArrayList<Long> bikes = new ArrayList<>();

  public FomoVehicle(long id, long capacity, long currentStation) {
    this.id = id;
    this.capacity = capacity;
    this.currentStation = currentStation;
  }
}
