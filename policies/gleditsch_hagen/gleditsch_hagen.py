import jpype
import jpype.imports
from jpype.types import *

from policies import Policy
import sim

# linux
# jpype.startJVM(convertStrings=False, classpath = ['jars/gs-core-1.3.jar','jars/gs-algo-1.3.jar','jars/gs-ui-1.3.jar','jars/javafx.graphics.jar','jars/poi-5.2.2.jar','jars/poi-ooxml-5.2.2.jar','jars/json-20220320.jar','/opt/xpressmp/lib/xprm.jar','policies/gleditsch_hagen/jars/gleditsch_hagen.jar'])
# Lasse PC1, PC2, PC3
jpype.startJVM(convertStrings=False, classpath = ['jars/gs-core-1.3.jar','jars/gs-algo-1.3.jar','jars/gs-ui-1.3.jar','jars/javafx.graphics.jar','jars/poi-5.2.2.jar','jars/poi-ooxml-5.2.2.jar','jars/json-20220320.jar','C:/xpressmp/lib/xprm.jar','policies/gleditsch_hagen/jars/gleditsch_hagen.jar'])

import java.util.ArrayList
from org.gleditsch_hagen.classes import Simulation,FomoAction,FomoStation,FomoVehicle

class GleditschHagenPolicy(Policy):
    def __init__(self):
        super().__init__()

    def get_best_action(self, simul, vehicle):
        stations = java.util.ArrayList()

        for s in simul.state.locations:
            station = FomoStation(s.id, s.capacity, s.get_leave_intensity(simul.day(), simul.hour()), s.get_arrive_intensity(simul.day(), simul.hour()), s.get_ideal_state(simul.day(), simul.hour()))
            for b in s.scooters:
                station.bikes.add(b.id)
            for d in simul.state.locations:
                station.distances.put(d.id, simul.state.get_distance(s.id, d.id))
            stations.add(station)

        vehicle = FomoVehicle(vehicle.id, vehicle.scooter_inventory_capacity, vehicle.current_location.id)

        fomoAction = Simulation.policy(stations, vehicle)

        battery_swaps = []
        for b in fomoAction.batterySwaps:
            battery_swaps.append(b)
        pick_ups = []
        for b in fomoAction.pickUps:
            pick_ups.append(b)
        delivery_scooters = []
        for b in fomoAction.deliveryScooters:
            delivery_scooters.append(b)
        return sim.Action(battery_swaps, pick_ups, delivery_scooters, fomoAction.nextLocation)
    
