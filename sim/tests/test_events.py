import unittest
import random

import clustering.scripts
import decision
from sim import (
    ScooterDeparture,
    ScooterArrival,
    VehicleArrival,
    Event,
    GenerateScooterTrips,
    LostTrip,
    World,
)
from settings import ITERATION_LENGTH_MINUTES

class EventsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World(
            40,
            policy=decision.SwapAllPolicy(),
            initial_state=clustering.scripts.get_initial_state(
                100, 10, initial_location_depot=False
            ),
        )
        self.world.stack = []
        self.vehicle = self.world.state.vehicles[0]
        self.large_world = World(
            40,
            policy=decision.SwapAllPolicy(),
            initial_state=clustering.scripts.get_initial_state(
                100, 20, initial_location_depot=False
            ),
        )
        self.large_world.stack = []
        self.vehicle_large_world = self.large_world.state.vehicles[0]

        self.departure_time = 1
        self.travel_time = 5

    def test_scooter_departure(self):
        departure_event = ScooterDeparture(
            self.departure_time, self.vehicle.current_location.id
        )
        # Remove all scooters from the scooter departure location
        self.vehicle.current_location.scooters = []
        departure_event.perform(self.world)

        # test if the time of world object is set to the departure time
        self.assertEqual(departure_event.time, self.world.time)

        # Check that a lost trip event is created
        self.assertIsInstance(self.world.stack.pop(), LostTrip)

        # create new departure event with scooters in departure cluster
        new_destination = random.choice(
            [
                cluster
                for cluster in self.world.state.stations
                if len(cluster.scooters) > 0
                and cluster.id != self.vehicle.current_location.id
            ]
        )
        departure_event = ScooterDeparture(11, new_destination.id)
        departure_event.perform(self.world)

        arrival_event = self.world.stack.pop()

        # Check that a ScooterArrival event is created
        self.assertIsInstance(arrival_event, ScooterArrival)

        # test if the arrival event created in departure has the same departure cluster id
        self.assertEqual(
            new_destination.id, arrival_event.departure_cluster_id,
        )

        # scooter should have been removed from the scooters in the state
        self.assertNotIn(arrival_event.scooter, self.world.state.get_scooters())

    def test_scooter_arrival(self):
        # Take two random clusters
        departure_cluster = self.world.state.get_random_cluster()
        arrival_cluster = self.world.state.get_random_cluster(exclude=departure_cluster)

        # Find a scooter to move in the departure cluster
        scooter = random.choice(departure_cluster.scooters)
        scooter_battery = scooter.battery

        arrival_event = ScooterArrival(
            self.departure_time + self.travel_time,
            scooter,
            arrival_cluster.id,
            departure_cluster.id,
            self.world.state.get_distance(departure_cluster.id, arrival_cluster.id),
        )

        arrival_event.perform(self.world)

        # test til world time is set to arrival time after arrival is performed
        self.assertEqual(self.world.time, self.departure_time + self.travel_time)

        # test if arrival cluster contains the arrived scooter
        self.assertIn(scooter, arrival_cluster.scooters)

        # test if battery has decreased
        self.assertLess(scooter.battery, scooter_battery)

    def test_vehicle_arrival(self):
        # Clear stack to check specific vehicle arrival event
        self.world.stack = []
        # Choose a random cluster for the vehicle to be in
        arrival_cluster = self.world.state.get_random_cluster(
            exclude=self.vehicle.current_location
        )
        self.vehicle.current_location = arrival_cluster
        # Create a vehicle arrival event with a arrival time of 20 arriving at a random cluster in the world state
        vehicle_arrival_event = VehicleArrival(20, self.vehicle.id, False)

        # Perform the vehicle arrival event
        vehicle_arrival_event.perform(self.world)

        # test if the time of world object is set to the departure time
        self.assertEqual(vehicle_arrival_event.time, self.world.time)

        # New current cluster is not the arrival cluster, as the do_action takes the vehicle to a new cluster
        self.assertNotEqual(arrival_cluster.id, self.vehicle.current_location.id)

        # Vehicle arrival event created a new vehicle arrival event
        self.assertEqual(1, len(self.world.stack))

    def test_generate_scooter_trips(self):
        generate_trips_event = GenerateScooterTrips(0)

        generate_trips_event.perform(self.large_world)

        # check if any trips or GenerateScooterTrip object is created
        self.assertGreater(len(self.large_world.stack), 0)

        # check if world stack contains a GenerateScooterTrips object
        self.assertTrue(
            any(
                isinstance(event, GenerateScooterTrips)
                for event in self.large_world.stack
            )
        )

        # check if all departure times are inside the interval of iteration duration
        self.assertTrue(
            all(
                [
                    True if 0 <= event.time <= ITERATION_LENGTH_MINUTES else False
                    for event in self.large_world.stack
                ]
            )
        )

    def test_lost_trip(self):
        lost_trip = LostTrip(2, 0)
        lost_trip.perform(self.world)

        # check if lost trip gives negative reward
        self.assertLess(sum([reward for reward, _ in self.world.rewards]), 0)

    def test_try_to_move_back_in_time(self):
        # Create a nice event to make time fly
        event = Event(10)
        event.perform(self.world)
        # Create an event that tries to move back in time
        wrong_event = Event(5)
        self.assertRaises(ValueError, wrong_event.perform, self.world)


if __name__ == "__main__":
    unittest.main()
