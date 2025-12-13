# -*- coding: utf-8 -*-
"""
Simulation logging utilities for tracking hourly and daily metrics.
"""
 
import os
import csv
import sim
from pathlib import Path
from settings import MAINTENANCE_INCREASE_PER_MINUTE
 
 
# Determine output directory relative to this file's location
_LOGGING_FILE_DIR = Path(__file__).parent
RESULTS_DIR = _LOGGING_FILE_DIR / 'simulation_results'
 
 
class LoggingSimulator(sim.Simulator):
    """
    Extended Simulator class that tracks hourly and daily metrics.
    """
   
    def __init__(self, *args, **kwargs):
        # Initialize tracking lists BEFORE calling super().__init__
        self.bike_movements = []  # List of all bike movements
        self.trip_requests = []  # List of all trip requests
        self.hourly_metrics = []  # List of hourly metrics
        self.hourly_station_metrics = []  # List of per-station hourly data
       
        self.last_logged_day = -1
        self.last_starvations = 0
        self.last_congestions = 0
        self.last_logged_hour = -1
        self.last_hour_starvations = 0
        self.last_hour_long_congestions = 0
        self.last_hour_short_congestions = 0
        self.last_hour_maintenance_violations = 0
        self.last_hour_maintenance_starvations = 0
        self.last_hour_bike_pickups = 0
        self.last_hour_bike_deliveries = 0
        self.last_hour_maintenance_time = 0.0
       
        # Now call parent __init__
        super().__init__(*args, **kwargs)
   
    def _estimate_post_trip_criticality(self, pre_criticality, travel_time):
        """
        Calculates the expected criticality after the trip.
        Since the bike object hasn't physically moved yet (we are logging at request time),
        we must calculate this mathematically rather than fetching the object state.
        """
        if pre_criticality is None or travel_time is None:
            return pre_criticality
           
        rate = MAINTENANCE_INCREASE_PER_MINUTE
        if hasattr(self.state, 'parameters') and 'degradation_rate' in self.state.parameters:
            rate = self.state.parameters['degradation_rate']
           
        # DEBUG PRINT
        print(f"DEBUG CRITICALITY: Pre={pre_criticality}, Time={travel_time}, Rate={rate}, Added={travel_time * rate}")
 
        estimated = pre_criticality + (travel_time * rate)
        return min(estimated, 1.0)
 
    def log_bike_movement(self, time, bike_id, departure_station_id, arrival_station_id, did_roam=False, bike_criticality=0.0):
        """Log a bike movement for later export to CSV"""
        # For bike movements (which usually happen upon arrival), we can trust the passed value
        # or fetch it if needed. Assuming passed value is correct here.
        day = int(time // (24*60))
        hour = int((time % (24*60)) // 60)
        minute = int(time % 60)
       
        self.bike_movements.append({
            'time_minutes': time,
            'day': day,
            'hour': hour,
            'minute': minute,
            'bike_id': bike_id,
            'departure_station': departure_station_id,
            'arrival_station': arrival_station_id,
            'did_roam': did_roam,
            'bike_criticality': bike_criticality
        })
        if len(self.bike_movements) % 100 == 0:
             print(f"DEBUG: Logged {len(self.bike_movements)} bike movements")
   
    def log_trip_request(self, time, station_id, success=True, failure_reason=None, did_roam=False,
                        arrival_station_id=None, travel_time=None, bike_id=None, bike_criticality=None):
        """Log a trip request (successful or failed)"""
        day = int(time // (24*60))
        hour = int((time % (24*60)) // 60)
        minute = int(time % 60)
       
        # 1. Previous Criticality
        prev_criticality = bike_criticality if bike_criticality is not None else 0.0
 
        # 2. Post-Trip Criticality (Calculated)
        # We cannot fetch this from the bike object because the bike hasn't arrived yet.
        post_criticality = prev_criticality
        if success and travel_time is not None:
             post_criticality = self._estimate_post_trip_criticality(prev_criticality, travel_time)
 
        self.trip_requests.append({
            'time_minutes': time,
            'day': day,
            'hour': hour,
            'minute': minute,
            'station_id': station_id,
            'success': success,
            'failure_reason': failure_reason if not success else None,
            'did_roam': did_roam,
            'arrival_station_id': arrival_station_id if success else None,
            'travel_time': travel_time if success else None,
            'bike_id': bike_id if success else None,
            'previous_bike_criticality': prev_criticality if success else None,
            'post_trip_bike_criticality': post_criticality if success else None
        })
        if len(self.trip_requests) % 100 == 0:
             print(f"DEBUG: Logged {len(self.trip_requests)} trip requests")
 
    def full_step(self):
        super().full_step()
       
        # Check for hourly logging
        current_time = self.state.time
        hour = int(current_time // 60)  # Hour since simulation start (0-indexed)
       
        # Log metrics when we enter a new hour (log the previous hour's data)
        if hour > self.last_logged_hour:
            # Log the hour that just completed (previous hour)
            hour_to_log = self.last_logged_hour if self.last_logged_hour >= 0 else 0
            if hour_to_log >= 0:
                self.log_hourly_metrics(hour_to_log, current_time)
            self.last_logged_hour = hour
       
        # Check for 23:00 daily logging
        day = int(current_time // (24*60))
        minute_of_day = current_time % (24*60)
       
        # We want to log once per day, when we pass 23:00 (1380 minutes)
        if day > self.last_logged_day and minute_of_day >= 1380:
            self.log_daily_metrics(day)
            self.last_logged_day = day
   
    def log_hourly_metrics(self, hour, current_time):
        """Log metrics for the hour that just completed (delta since last hour)"""
        # Get current aggregate values
        current_starvations = self.state.metrics.get_aggregate_value('starvations')
        current_bike_starvations = self.state.metrics.get_aggregate_value('bike starvations')
        current_long_congestions = self.state.metrics.get_aggregate_value('long congestions')
        current_short_congestions = self.state.metrics.get_aggregate_value('short congestions')
        current_maintenance_violations = self.state.metrics.get_aggregate_value('maintenance violations')
        current_maintenance_starvations = self.state.metrics.get_aggregate_value('maintenance_starvation')
        current_bike_pickups = self.state.metrics.get_aggregate_value('num bike pickups')
        current_bike_deliveries = self.state.metrics.get_aggregate_value('num bike deliveries')
        current_maintenance_time = self.state.metrics.get_aggregate_value('maintenance time')
       
        # Calculate average bike criticality for entire fleet
        all_bikes = self.state.get_all_bikes()
        print(f"DEBUG: Calculating average bike criticality for {len(all_bikes)} bikes")
        #print criticalities for all bikes along the bike id
        for bike in all_bikes:
            print(f"  Bike ID: {bike.bike_id}, Criticality: {bike.maintenance_criticality:.6f}")
        if all_bikes:
            avg_bike_criticality = sum(bike.maintenance_criticality for bike in all_bikes) / len(all_bikes)
            # Count bikes in different criticality ranges
            bikes_critical = sum(1 for bike in all_bikes if bike.maintenance_criticality > 0.83)
            bikes_high = sum(1 for bike in all_bikes if 0.6 < bike.maintenance_criticality <= 0.83)
            bikes_medium = sum(1 for bike in all_bikes if 0.3 < bike.maintenance_criticality <= 0.6)
            bikes_low = sum(1 for bike in all_bikes if bike.maintenance_criticality <= 0.3)
           
            # Log bike criticality distribution
            print(f"\n--- HOURLY BIKE CRITICALITY (Hour {hour}, t={current_time:.1f}) ---")
            print(f"Average Criticality: {avg_bike_criticality:.4f}")
            print(f"Distribution:")
            print(f"  Critical (>0.83):     {bikes_critical:>4} bikes ({bikes_critical/len(all_bikes)*100:.1f}%)")
            print(f"  High (0.60-0.83):     {bikes_high:>4} bikes ({bikes_high/len(all_bikes)*100:.1f}%)")
            print(f"  Medium (0.30-0.60):   {bikes_medium:>4} bikes ({bikes_medium/len(all_bikes)*100:.1f}%)")
            print(f"  Low (<=0.30):         {bikes_low:>4} bikes ({bikes_low/len(all_bikes)*100:.1f}%)")
            print(f"  Total Fleet:          {len(all_bikes):>4} bikes")
            print(f"-------------------------------------------------------\n")
        else:
            avg_bike_criticality = 0.0
            bikes_critical = bikes_high = bikes_medium = bikes_low = 0
       
        # Calculate station-level metrics
        self.log_station_metrics(hour, current_time)
       
        # Calculate deltas (events in the hour that just completed)
        hourly_starvations = current_starvations - self.last_hour_starvations
        hourly_bike_starvations = current_bike_starvations - getattr(self, 'last_hour_bike_starvations', 0)
        hourly_long_congestions = current_long_congestions - self.last_hour_long_congestions
        hourly_short_congestions = current_short_congestions - self.last_hour_short_congestions
        hourly_maintenance_violations = current_maintenance_violations - self.last_hour_maintenance_violations
        hourly_maintenance_starvations = current_maintenance_starvations - self.last_hour_maintenance_starvations
        hourly_bike_pickups = current_bike_pickups - self.last_hour_bike_pickups
        hourly_bike_deliveries = current_bike_deliveries - self.last_hour_bike_deliveries
        hourly_maintenance_time = current_maintenance_time - self.last_hour_maintenance_time
       
        # Calculate day and hour in proper format
        day = int(current_time // (24*60))
        hour_of_day = int((current_time % (24*60)) // 60)
        hour_formatted = 24 if hour_of_day == 0 else hour_of_day
       
        # Store hourly data (hour is the hour that just completed)
        self.hourly_metrics.append({
            'day': day,
            'hour': hour_formatted,
            'hour_index': hour,  # Original hour index for reference
            'time_minutes': current_time,
            'starvations': hourly_starvations,
            'bike_starvations': hourly_bike_starvations,
            'long_congestions': hourly_long_congestions,
            'short_congestions': hourly_short_congestions,
            'maintenance_violations': hourly_maintenance_violations,
            'maintenance_starvations': hourly_maintenance_starvations,
            'bike_pickups': hourly_bike_pickups,
            'bike_deliveries': hourly_bike_deliveries,
            'maintenance_time': hourly_maintenance_time,
            'avg_bike_criticality': avg_bike_criticality,
            'bikes_critical': bikes_critical,
            'bikes_high': bikes_high,
            'bikes_medium': bikes_medium,
            'bikes_low': bikes_low,
        })
       
        # Update last hour values for next calculation
        self.last_hour_starvations = current_starvations
        self.last_hour_bike_starvations = current_bike_starvations
        self.last_hour_long_congestions = current_long_congestions
        self.last_hour_short_congestions = current_short_congestions
        self.last_hour_maintenance_violations = current_maintenance_violations
        self.last_hour_maintenance_starvations = current_maintenance_starvations
        self.last_hour_bike_pickups = current_bike_pickups
        self.last_hour_bike_deliveries = current_bike_deliveries
        self.last_hour_maintenance_time = current_maintenance_time
   
    def log_station_metrics(self, hour, current_time):
        """Log per-station metrics for the current hour"""
        day = int(current_time // (24*60))
        hour_of_day = int((current_time % (24*60)) // 60)
        hour_formatted = 24 if hour_of_day == 0 else hour_of_day
       
        # Collect metrics for each station
        for station in self.state.get_stations():
            station_data = {
                'day': day,
                'hour': hour_formatted,
                'hour_index': hour,
                'time_minutes': current_time,
                'station_id': station.id,
                'num_bikes': station.number_of_bikes(),
                'num_usable_bikes': len(station.get_available_bikes()),
                'num_unusable_bikes': len(station.get_unusable_bikes()),
                'avg_criticality': station.get_average_maintenance_criticality(),
                'capacity': station.capacity,
            }
            self.hourly_station_metrics.append(station_data)
           
    def log_daily_metrics(self, day):
        starvations = self.state.metrics.get_aggregate_value('starvations')
        congestions = self.state.metrics.get_aggregate_value('long congestions')
       
        daily_starvations = starvations - self.last_starvations
        daily_congestions = congestions - self.last_congestions
       
        # Get bikes that have criticality above 0.83
        critical_bikes = [bike for bike in self.state.get_all_bikes() if bike.maintenance_criticality > 0.83]
        print(f"DAY {day} END: Bikes with criticality > 0.83: {len(critical_bikes)}")
       
        # Print details of critical bikes BEFORE resetting
        if critical_bikes:
            print(f"\n{'='*60}")
            print(f"CRITICAL BIKES BEFORE OVERNIGHT MAINTENANCE (Day {day} End)")
            print(f"{'='*60}")
            print(f"{'Bike ID':<15} {'Criticality Before':<20} {'Location':<20}")
            print("-" * 60)
           
            for bike in critical_bikes:
                location = "In Transit"
                if hasattr(bike, 'location') and bike.location:
                    location = bike.location.id
                else:
                    # Check if bike is at any station
                    for station in self.state.get_stations():
                        if bike in station.get_bikes():
                            location = station.id
                            break
               
                print(f"{bike.bike_id:<15} {bike.maintenance_criticality:<20.4f} {location:<20}")
           
            print("-" * 60)
            print(f"Total: {len(critical_bikes)} bikes will be reset to 0.0 criticality")
            print(f"{'='*60}\n")
           
            # Reset criticality to 0.0 (simulating overnight maintenance)
            for bike in critical_bikes:
                bike.maintenance_criticality = 0.0
           
            # Print confirmation AFTER resetting
            print(f"{'='*60}")
            print(f"CRITICAL BIKES AFTER OVERNIGHT MAINTENANCE (Day {day} End)")
            print(f"{'='*60}")
            print(f"All {len(critical_bikes)} bikes have been reset to 0.0 criticality")
            print(f"{'='*60}\n")
 
        print(f"\n{'='*40}")
        print(f"DAY {day} SUMMARY (23:00)")
        print(f"{'='*40}")
        print(f"Accumulated Starvations: {starvations} (+{daily_starvations} today)")
        print(f"Accumulated Congestions: {congestions} (+{daily_congestions} today)")
        print(f"{'='*40}\n")
       
        self.last_starvations = starvations
        self.last_congestions = congestions
   
    def run(self):
        """Override run to log final hour metrics"""
        super().run()
       
        # Log final hour metrics after simulation ends
        final_time = self.state.time
        final_hour = int(final_time // 60)
       
        # Log the final hour's data (the hour we're currently in when simulation ends)
        if final_hour >= 0:
            self.log_hourly_metrics(final_hour, final_time)
 
 
def write_results_to_file(filename, simulator, duration, solve_time, seed, append=False):
    """
    Write simulation results to a CSV file.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = RESULTS_DIR / filename
   
    mode = 'a' if append else 'w'
    file_exists = os.path.isfile(filepath) and append
   
    with open(filepath, mode, newline='') as f:
        writer = csv.writer(f)
       
        # Write header if new file
        if not file_exists:
            writer.writerow([
                'Seed',
                'Duration (hours)',
                'Total Runtime (s)',
                'Failed Events',
                'Starvations',
                'Bike Starvations',
                'Long Congestions',
                'Short Congestions',
                'Total Trips',
                'Bike Departures',
                'Bike Arrivals',
                'Vehicle Arrivals',
                'Bike Deliveries',
                'Bike Pickups',
                'Maintenance Time (minutes)',
                'Maintenance Violations',
                'Maintenance Starvations',
            ])
       
        # Write data row
        writer.writerow([
            seed,
            duration,
            round(solve_time, 2),
            simulator.state.metrics.get_aggregate_value('failed events'),
            simulator.state.metrics.get_aggregate_value('starvations'),
            simulator.state.metrics.get_aggregate_value('bike starvations'),
            simulator.state.metrics.get_aggregate_value('long congestions'),
            simulator.state.metrics.get_aggregate_value('short congestions'),
            simulator.state.metrics.get_aggregate_value('trips'),
            simulator.state.metrics.get_aggregate_value('bike departure'),
            simulator.state.metrics.get_aggregate_value('bike arrival'),
            simulator.state.metrics.get_aggregate_value('vehicle arrivals'),
            simulator.state.metrics.get_aggregate_value('num bike deliveries'),
            simulator.state.metrics.get_aggregate_value('num bike pickups'),
            simulator.state.metrics.get_aggregate_value('maintenance time'),
            simulator.state.metrics.get_aggregate_value('maintenance violations'),
            simulator.state.metrics.get_aggregate_value('maintenance_starvation'),
        ])
 
 
def write_simulation_summary(filename, simulator, duration, policy, seed, num_vehicles=1):
    """
    Write simulation summary including parameters, objective function, and routes.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = RESULTS_DIR / filename
 
    with open(filepath, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"SJOVIK SUND POLICY SIMULATION SUMMARY (Seed {seed})\n")
        f.write("="*80 + "\n\n")
       
        # 1. Parameters
        f.write("--- PARAMETERS ---\n")
        f.write(f"Policy Type: MILP-based (Direct Optimization)\n")
        f.write(f"Number of Vehicles: {num_vehicles}\n")
        f.write(f"Simulation Duration: {duration:.1f} hours\n")
       
        if hasattr(policy, 'time_horizon'):
            f.write(f"MILP Parameters:\n")
            f.write(f"  Time Horizon (T): {policy.time_horizon} periods\n")
            f.write(f"  Period Length (tau): {policy.tau} minutes\n")
            f.write(f"  Roaming Enabled: {policy.roaming}\n")
       
        if policy.weights:
            f.write(f"\nObjective Weights:\n")
            f.write(f"  Starvation Weight (w_S): {policy.weights[0]}\n")
            f.write(f"  Congestion Weight (w_C): {policy.weights[1]}\n")
            f.write(f"  Deviation Weight (w_D): {policy.weights[2]}\n")
            f.write(f"  Maintenance reward (r_M): {policy.weights[3]}\n")
        else:
            f.write(f"\nObjective Weights: Default (0.45, 0.45, 0.09, 0.01)\n")
       
        # 2. Accumulated Objective Function
        f.write("\n--- ACCUMULATED OBJECTIVE FUNCTION ---\n")
       
        # Get aggregate metrics
        # Regner bare med long congestions her
        starvations = simulator.state.metrics.get_aggregate_value('starvations')
        bike_starvations = simulator.state.metrics.get_aggregate_value('bike starvations')
        congestions_long = simulator.state.metrics.get_aggregate_value('long congestions') # + simulator.state.metrics.get_aggregate_value('short congestions')
        maintenance_time = simulator.state.metrics.get_aggregate_value('maintenance time')
        maintenance_violations = simulator.state.metrics.get_aggregate_value('maintenance_violations')
        maintenance_starvation = simulator.state.metrics.get_aggregate_value('maintenance_starvation')
        congestions_short = simulator.state.metrics.get_aggregate_value('short congestions')
        bike_departures = simulator.state.metrics.get_aggregate_value('bike departure')
        trips = simulator.state.metrics.get_aggregate_value('trips')
        #deviations = simulator.state.metrics.get_aggregate_value('deviation')
       
        # Calculate objective
        w_S, w_C, w_D, r_M = policy.weights if policy.weights else (0.45, 0.45, 0.09, 0.01)
       
        obj_val = (w_S * starvations) + (w_C * congestions_long) - (r_M * maintenance_time)
       
        f.write(f"Total Objective Value: {obj_val:.4f}\n")
        f.write(f"Breakdown:\n")
        f.write(f"  Starvations: {starvations} (Contribution: {w_S * starvations:.4f})\n")
        f.write(f"  Congestions: {congestions_long} (Contribution: {w_C * congestions_long:.4f})\n")
        #f.write(f"  Deviations: {deviations} (Contribution: {w_D * deviations:.4f})\n")
        f.write(f"  Maintenance Time: {maintenance_time:.2f} (Contribution: {-r_M * maintenance_time:.4f})\n")
        f.write(f"  Short Congestions: {congestions_short} (Contribution: {0})\n")
        f.write(f"  Maintenance Violation Events: {maintenance_violations}\n")
        f.write(f"  Maintenance Starvations: {maintenance_starvation}\n")
       
        # Verification of trip accounting
        f.write(f"\n--- TRIP ACCOUNTING VERIFICATION ---\n")
        f.write(f"Attempted Bike Departures (trips metric): {trips}\n")
        f.write(f"Successful Bike Departures: {bike_departures}\n")
        f.write(f"Bike Starvations: {bike_starvations}\n")
        f.write(f"Maintenance Starvations: {maintenance_starvation}\n")
        f.write(f"\nVerification: {bike_departures} + {bike_starvations} + {maintenance_starvation} = {bike_departures + bike_starvations + maintenance_starvation}\n")
        if trips == bike_departures + bike_starvations + maintenance_starvation:
            f.write(f"[OK] VERIFIED: All attempted departures accounted for\n")
        else:
            f.write(f"[ERROR] MISMATCH: Expected {trips}, got {bike_departures + bike_starvations + maintenance_starvation}\n")
       
        if hasattr(policy, 'optimality_gaps') and policy.optimality_gaps:
            avg_gap = sum(policy.optimality_gaps) / len(policy.optimality_gaps)
            max_gap = max(policy.optimality_gaps)
            f.write(f"\n--- OPTIMIZATION PERFORMANCE ---\n")
            f.write(f"Average Optimality Gap: {avg_gap:.4%}\n")
            f.write(f"Maximum Optimality Gap: {max_gap:.4%}\n")
            f.write(f"Total Optimizations: {len(policy.optimality_gaps)}\n")
 
        # 3. Routes
        f.write("\n--- ACTUAL VEHICLE ROUTES ---\n")
       
        # Extract routes from policy attached to vehicles in the simulator
        policy_with_routes = None
        for vehicle in simulator.state.vehicles.values():
            if hasattr(vehicle.policy, 'vehicle_routes'):
                policy_with_routes = vehicle.policy
                break
       
        if policy_with_routes and policy_with_routes.vehicle_routes:
            for vehicle_id, route in policy_with_routes.vehicle_routes.items():
                f.write(f"Vehicle {vehicle_id} Route:\n")
                if not route:
                    f.write("  No route recorded\n\n")
                    continue
               
                # Sort by time to ensure chronological order
                route.sort(key=lambda x: x[0])
               
                # Write route with time information
                for i, (time_val, station_id) in enumerate(route):
                    # Convert time to day/hour/minute format
                    day = int(time_val // (24*60))
                    hour = int((time_val % (24*60)) // 60)
                    minute = int(time_val % 60)
                   
                    if i == 0:
                        f.write(f"  Start: {station_id} at Day {day}, Hour {hour:02d}:{minute:02d} (t={time_val:.1f})\n")
                    else:
                        # Calculate travel time from previous station
                        prev_time = route[i-1][0]
                        travel_duration = time_val - prev_time
                        f.write(f"  Move {i}: {station_id} at Day {day}, Hour {hour:02d}:{minute:02d} (t={time_val:.1f}) [+{travel_duration:.1f} min]\n")
               
                # Summary statistics
                total_time = route[-1][0] - route[0][0] if len(route) > 1 else 0
                unique_stations = len(set(station for _, station in route))
                f.write(f"  Summary: {len(route)} stops, {unique_stations} unique stations, {total_time:.1f} min total\n\n")
        else:
             f.write("No route information available.\n")
       
 
    print(f"Simulation summary written to: {filepath}")
 
 
def write_hourly_metrics_to_file(filename, simulator, seed):
    """
    Write hourly metrics to a CSV file (one file per simulation run).
   
    Args:
        filename: Name of the CSV file to write
        simulator: LoggingSimulator instance with hourly_metrics populated
        seed: Random seed used for the simulation
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = RESULTS_DIR / filename
   
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
       
        # Write header
        writer.writerow([
            'Seed',
            'Day',
            'Hour',
            'Time (minutes)',
            'Starvations',
            'Bike Starvations',
            'Long Congestions',
            'Short Congestions',
            'Maintenance Violations',
            'Maintenance Starvations',
            'Bike Pickups',
            'Bike Deliveries',
            'Maintenance Time (minutes)',
            'Avg Bike Criticality',
            'Bikes Critical (>0.83)',
            'Bikes High (0.60-0.83)',
            'Bikes Medium (0.30-0.60)',
            'Bikes Low (<=0.30)',
        ])
       
        # Write hourly data rows
        for hour_data in simulator.hourly_metrics:
            writer.writerow([
                seed,
                hour_data['day'],
                hour_data['hour'],
                round(hour_data['time_minutes'], 2),
                hour_data['starvations'],
                hour_data.get('bike_starvations', 0),
                hour_data['long_congestions'],
                hour_data['short_congestions'],
                hour_data['maintenance_violations'],
                hour_data['maintenance_starvations'],
                hour_data.get('bike_pickups', 0),
                hour_data.get('bike_deliveries', 0),
                round(hour_data.get('maintenance_time', 0.0), 2),
                round(hour_data.get('avg_bike_criticality', 0.0), 4),
                hour_data.get('bikes_critical', 0),
                hour_data.get('bikes_high', 0),
                hour_data.get('bikes_medium', 0),
                hour_data.get('bikes_low', 0),
            ])
 
 
def write_vehicle_visits_to_file(filename, simulator, seed):
    """
    Write vehicle visit logs to a CSV file, showing where and when each vehicle visited stations.
   
    Args:
        filename: Name of the CSV file to write
        simulator: Simulator instance with vehicle policy containing route information
        seed: Random seed used for the simulation
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = RESULTS_DIR / filename
   
    # Extract routes from policy attached to vehicles
    policy_with_routes = None
    for vehicle in simulator.state.vehicles.values():
        if hasattr(vehicle.policy, 'vehicle_routes'):
            policy_with_routes = vehicle.policy
            break
   
    if not policy_with_routes or not policy_with_routes.vehicle_routes:
        # No route data available - create empty file with header
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Seed',
                'Vehicle ID',
                'Visit Number',
                'Station ID',
                'Time (minutes)',
                'Day',
                'Hour',
                'Minute',
                'Time Since Previous Visit (minutes)',
            ])
        print(f"Warning: No vehicle route data available for seed {seed}")
        return
   
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
       
        # Write header
        writer.writerow([
            'Seed',
            'Vehicle ID',
            'Visit Number',
            'Station ID',
            'Time (minutes)',
            'Day',
            'Hour',
            'Minute',
            'Time Since Previous Visit (minutes)',
        ])
       
        # Write visit data for each vehicle
        for vehicle_id, route in sorted(policy_with_routes.vehicle_routes.items()):
            if not route:
                continue
           
            # Sort by time to ensure chronological order
            route_sorted = sorted(route, key=lambda x: x[0])
           
            for visit_num, (time_val, station_id) in enumerate(route_sorted, start=1):
                # Convert time to day/hour/minute format
                day = int(time_val // (24*60))
                hour = int((time_val % (24*60)) // 60)
                minute = int(time_val % 60)
               
                # Calculate time since previous visit
                if visit_num == 1:
                    time_since_prev = 0.0
                else:
                    prev_time = route_sorted[visit_num - 2][0]
                    time_since_prev = time_val - prev_time
               
                writer.writerow([
                    seed,
                    vehicle_id,
                    visit_num,
                    station_id,
                    round(time_val, 2),
                    day,
                    hour,
                    minute,
                    round(time_since_prev, 2),
                ])
 
 
def write_station_hourly_metrics_to_file(filename, simulator, seed):
    """
    Write per-station hourly metrics to a CSV file, showing bike counts and average criticality
    for each station at each hour.
   
    Args:
        filename: Name of the CSV file to write
        simulator: LoggingSimulator instance with hourly_station_metrics populated
        seed: Random seed used for the simulation
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = RESULTS_DIR / filename
   
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
       
        # Write header
        writer.writerow([
            'Seed',
            'Day',
            'Hour',
            'Time (minutes)',
            'Station ID',
            'Total Bikes',
            'Usable Bikes',
            'Unusable Bikes',
            'Avg Maintenance Criticality',
            'Capacity',
        ])
       
        # Write station data rows
        for station_data in simulator.hourly_station_metrics:
            writer.writerow([
                seed,
                station_data['day'],
                station_data['hour'],
                round(station_data['time_minutes'], 2),
                station_data['station_id'],
                station_data['num_bikes'],
                station_data['num_usable_bikes'],
                station_data['num_unusable_bikes'],
                round(station_data['avg_criticality'], 4),
                station_data['capacity'],
            ])
 
 
def write_bike_movements_to_file(filename, simulator, seed, alpha=None):
    """
    Write all bike movements to a CSV file.
   
    Args:
        filename: Name of the CSV file to write
        simulator: LoggingSimulator instance with bike_movements populated
        seed: Random seed used for the simulation
        alpha: Alpha parameter value (optional)
    """
    print(f"DEBUG write_bike_movements: simulator_id={id(simulator)}, list_id={id(simulator.bike_movements)}, len={len(simulator.bike_movements)}")
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        filepath = RESULTS_DIR / filename
        print(f"DEBUG: Opening file for writing: {filepath}")
        print(f"DEBUG: RESULTS_DIR = {RESULTS_DIR}")
        print(f"DEBUG: First movement = {simulator.bike_movements[0] if simulator.bike_movements else 'EMPTY LIST'}")
       
        with open(filepath, 'w', newline='') as f:
            print(f"DEBUG: File opened, writing header...")
            writer = csv.writer(f)
           
            # Write header
            writer.writerow([
                'Seed',
                'Alpha',
                'Day',
                'Hour',
                'Minute',
                'Time (minutes)',
                'Bike ID',
                'Departure Station',
                'Arrival Station',
                'Did Roam',
                'Bike Criticality',
            ])
            print(f"DEBUG: Header written, writing {len(simulator.bike_movements)} rows...")
           
            # Write bike movement data rows
            count = 0
            for movement in simulator.bike_movements:
                writer.writerow([
                    seed,
                    alpha if alpha is not None else '',
                    movement['day'],
                    movement['hour'],
                    movement['minute'],
                    round(movement['time_minutes'], 2),
                    movement['bike_id'],
                    movement['departure_station'],
                    movement['arrival_station'],
                    movement['did_roam'],
                    round(movement.get('bike_criticality', 0.0), 4),
                ])
                count += 1
           
            print(f"DEBUG: Wrote {count} rows, closing file...")
       
        print(f"DEBUG: Successfully wrote {len(simulator.bike_movements)} bike movements to {filename}")
    except Exception as e:
        print(f"ERROR writing bike movements: {e}")
        import traceback
        traceback.print_exc()
 
 
def write_trip_requests_to_file(filename, simulator, seed, alpha=None):
    """
    Write all trip requests (successful and failed) to a CSV file.
   
    Args:
        filename: Name of the CSV file to write
        simulator: LoggingSimulator instance with trip_requests populated
        seed: Random seed used for the simulation
        alpha: Alpha parameter value (optional)
    """
    print(f"DEBUG: Writing {len(simulator.trip_requests)} trip requests to {filename}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = RESULTS_DIR / filename
   
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
       
        # Write header
        writer.writerow([
            'Seed',
            'Alpha',
            'Day',
            'Hour',
            'Minute',
            'Time (minutes)',
            'Station ID',
            'Success',
            'Failure Reason',
            'Did Roam',
            'Arrival Station ID',
            'Travel Time (minutes)',
            'Bike ID',
            'Previous Bike Criticality',
            'Post-Trip Bike Criticality',
        ])
       
        # Write trip request data rows
        for request in simulator.trip_requests:
            writer.writerow([
                seed,
                alpha if alpha is not None else '',
                request['day'],
                request['hour'],
                request['minute'],
                round(request['time_minutes'], 2),
                request['station_id'],
                request['success'],
                request['failure_reason'] if not request['success'] else '',
                request['did_roam'],
                request.get('arrival_station_id', ''),
                round(request['travel_time'], 2) if request.get('travel_time') is not None else '',
                request.get('bike_id', ''),
                round(request['previous_bike_criticality'], 4) if request.get('previous_bike_criticality') is not None else '',
                round(request['post_trip_bike_criticality'], 4) if request.get('post_trip_bike_criticality') is not None else '',
            ])
 