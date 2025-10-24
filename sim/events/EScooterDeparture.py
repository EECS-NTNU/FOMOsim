import sim
from sim import Event
from settings import *
import numpy as np
import random

class EScooterDeparture(Event):
    """
    Event fired when a customer requests a trip from a given departure area. Creates a Lost Trip or Bike Arrival
    event based on the availability of the area
    """

    def __init__(self, departure_time, departure_area_id):
        super().__init__(departure_time)
        self.departure_area_id = departure_area_id

    def perform(self, simul) -> None:
        """
        :param simul: Simulation object
        """

        super().perform(simul)

        # get departure area
        departure_area = simul.state.get_location_by_id(self.departure_area_id)

        # get all available bike in the area
        available_escooters = departure_area.get_available_bikes()

        # if there are no more available bikes -> make a LostTrip event for that departure time
        if len(available_escooters) > 0:
            escooter = available_escooters.pop(0)

            if FULL_TRIP:
                if simul.state.rng.uniform(0, 1) < RANDOM_DESTINATION_PROB:
                    # Exclude the current area from the random selection
                    other_areas = [area for area in simul.state.get_areas() if area.id != self.departure_area_id]
                    arrival_area = simul.state.rng.choice(other_areas)
                    simul.state.metrics.add_aggregate_metric(simul.state, "random trips", 1)
                else:
                    # get an arrival area from the leave prob distribution
                    p=departure_area.get_move_probabilities(simul.state, simul.state.day(), simul.state.hour())
                    sum_p = sum(p.values())
                    p_normalized = []
                    for i in p.keys():
                        if sum_p > 0:
                            p_normalized.append(p[i] * (1.0/sum_p))
                        else:
                            p_normalized.append(1/len(p))
                    arrival_area = simul.state.rng.choice(simul.state.get_areas(), p = p_normalized)

                # calculate arrival time
                travel_time = simul.state.get_travel_time(
                    departure_area.id,
                    arrival_area.id,
                )

                # create an arrival event for the departed bike
                simul.add_event(
                    sim.EScooterArrival(
                        self.time,
                        travel_time,
                        escooter,
                        arrival_area.id,
                        departure_area.id,
                    )
                )
                
            # remove bike from the departure area
            departure_area.remove_bike(escooter)

            simul.state.set_bike_in_use(escooter)

            simul.state.metrics.add_aggregate_metric(simul.state, "escooter departure", 1)
            simul.state.metrics.add_aggregate_metric(simul.state, "events", 2)

        else:
            if FULL_TRIP:

                # Find closest area within allowed radius of Hexagons to search for available bikes
                closest_neighbour_with_bikes = simul.state.get_closest_available_area(departure_area)

                if closest_neighbour_with_bikes:
                    # Find distance to closest neighbor and 
                    distance = departure_area.distance_to(closest_neighbour_with_bikes.get_lat(), closest_neighbour_with_bikes.get_lon())
                    
                    # Get probability distribution for finding arrival area
                    p = departure_area.get_move_probabilities(simul.state, simul.state.day(), simul.state.hour())
                    sum_p = sum(p.values())
                    p_normalized = []
                    for i in p.keys():
                        if sum_p > 0:
                            p_normalized.append(p[i] * (1.0/sum_p))
                        else:
                            p_normalized.append(1/len(p))
                else:
                    distance = float('inf')

                # Use acceptance/rejection function to decide if roaming is accepted
                if self.acceptance_rejection(distance, simul):
                    available_escooters = closest_neighbour_with_bikes.get_available_bikes()
                    escooter=available_escooters.pop(0)
                    
                    if simul.state.rng.uniform(0, 1) < RANDOM_DESTINATION_PROB:
                        # Exclude the current area from the random selection
                        other_areas = [area for area in simul.state.get_areas() if area.id != self.departure_area_id]
                        arrival_area = simul.state.rng.choice(other_areas)
                        simul.state.metrics.add_aggregate_metric(simul.state, "random trips", 1)
                    else:
                        arrival_area = simul.state.rng.choice(simul.state.get_areas(), p = p_normalized)

                    # calculate arrival time 
                    #total travel time, roaming for bike from departure area to neighbour + cycling to arrival area
                    travel_time = simul.state.get_travel_time(
                        closest_neighbour_with_bikes.id,
                        arrival_area.id,) + simul.state.get_travel_time(departure_area.id,
                        closest_neighbour_with_bikes.id)*(ESCOOTER_SPEED/WALKING_SPEED)

                    # remove bike from the new departure area
                    closest_neighbour_with_bikes.remove_bike(escooter)

                    simul.state.set_bike_in_use(escooter)

                    # create an arrival event for the roaming user from the new departure area
                    simul.add_event(
                        sim.EScooterArrival(
                            self.time,
                            travel_time,
                            escooter,
                            arrival_area.id,
                            closest_neighbour_with_bikes.id,
                        )
                    )

                    simul.state.metrics.add_aggregate_metric(simul.state, "escooter departure", 1)
                    simul.state.metrics.add_aggregate_metric(simul.state, "events", 2)

                    simul.state.metrics.add_aggregate_metric(simul.state, "roaming for escooters", 1)
                    simul.state.metrics.add_aggregate_metric(simul.state, "roaming distance for escooters", distance)

                else:
                    if departure_area.number_of_bikes() <= 0:
                        simul.state.metrics.add_aggregate_metric(simul.state, "escooter starvations", 1)
                        departure_area.metrics.add_aggregate_metric(simul.state, "escooter starvations", 1)
                    else:
                        simul.state.metrics.add_aggregate_metric(simul.state, "battery starvations", 1)
                        departure_area.metrics.add_aggregate_metric(simul.state, "battery starvations", 1)

                    simul.state.metrics.add_aggregate_metric(simul.state, "events", 1)
                    simul.state.metrics.add_aggregate_metric(simul.state, "starvations", 1)
                    departure_area.metrics.add_aggregate_metric(simul.state, "starvations", 1)
                    simul.state.metrics.add_aggregate_metric(simul.state, "failed events", 1)

        simul.state.metrics.add_aggregate_metric(simul.state, "trips", 1)

    def __repr__(self):
        return f"<{self.__class__.__name__} at time {self.time}, departing from area {self.departure_area_id}>"
    
    def acceptance_rejection(self,distance, simul):
        # TODO denne må tilpasses e-scooter kunder
        prob_acceptance = -1.6548*distance**2-0.7036*distance+1.0133
        random_roaming_limit = simul.state.rng.uniform(0,1)
        if random_roaming_limit <= prob_acceptance:
            return True
        else:
            return False 