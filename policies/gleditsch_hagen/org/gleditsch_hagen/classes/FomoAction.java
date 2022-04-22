package org.gleditsch_hagen.classes;

import java.util.ArrayList;

public class FomoAction {
  public ArrayList<Integer> batterySwaps = new ArrayList<>();
  public ArrayList<Integer> pickUps = new ArrayList<>();
  public ArrayList<Integer> deliveryScooters = new ArrayList<>();
  public int nextLocation;

  FomoAction() {
    nextLocation = 0;
  }
}
