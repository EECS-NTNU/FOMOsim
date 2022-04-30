package org.gleditsch_hagen.classes;

import java.util.ArrayList;

public class FomoAction {
  public ArrayList<Long> batterySwaps = new ArrayList<>();
  public ArrayList<Long> pickUps = new ArrayList<>();
  public ArrayList<Long> deliveryScooters = new ArrayList<>();
  public long nextLocation;

  FomoAction() {
    nextLocation = 0;
  }
}
