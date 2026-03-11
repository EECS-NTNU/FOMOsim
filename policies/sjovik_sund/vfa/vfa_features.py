"""
VFA Basis Functions (Features) for DSJBRMP

Implements the separable value function approximation:
    V̄(S_k^x) ≈ Σ_n v̄_n(f_k^{n,x})

Each station's value is approximated using these features:
1. Expected functional shortage (demand vs. functional bikes)
2. Expected return rejections (capacity vs. total bikes + returns)
3. Unattended severe damage (depot bikes left behind)
4. Spatial-routing synergy (vehicles en route with cargo)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

from .vfa_state import MDPState, StationInventory, VehicleStatus


@dataclass
class DemandForecast:
    """
    Demand forecast for a station at a given time.
    
    Attributes:
        station_id: Station identifier
        time: Forecast time (hour of day, day of week, etc.)
        expected_rentals: Expected number of rental requests
        expected_returns: Expected number of bike returns
        rental_variance: Variance in rental demand
        return_variance: Variance in return demand
    """
    station_id: str
    time: float
    expected_rentals: float
    expected_returns: float
    rental_variance: float = 0.0
    return_variance: float = 0.0


class DemandForecaster:
    """
    Provides demand forecasts based on historical patterns.
    
    In a real implementation, this would use:
    - Time-of-day patterns
    - Day-of-week patterns
    - Historical demand data
    - Machine learning models
    
    For now, we implement a simple moving average approach.
    """
    
    def __init__(self, historical_data: Optional[Dict] = None):
        """
        Initialize forecaster.
        
        Args:
            historical_data: Historical demand patterns (station_id -> time -> demand)
        """
        self.historical_data = historical_data or {}
        self.default_rental_rate = 1.0  # bikes/hour
        self.default_return_rate = 1.0  # bikes/hour
    
    def forecast(self, station_id: str, time: float, horizon: float = 1.0) -> DemandForecast:
        """
        Forecast demand for a station over a time horizon.
        
        Args:
            station_id: Station to forecast for
            time: Current time
            horizon: Forecast horizon (hours)
        
        Returns:
            DemandForecast with expected rentals and returns
        """
        # Get day of week and hour of day
        day_of_week = int(time // (24 * 60)) % 7
        hour_of_day = int((time // 60) % 24)
        
        # Look up historical pattern or use defaults
        if station_id in self.historical_data:
            pattern = self.historical_data[station_id].get(
                (day_of_week, hour_of_day),
                {'rentals': self.default_rental_rate, 'returns': self.default_return_rate}
            )
            rental_rate = pattern['rentals']
            return_rate = pattern['returns']
        else:
            rental_rate = self.default_rental_rate
            return_rate = self.default_return_rate
        
        # Scale by horizon
        expected_rentals = rental_rate * horizon
        expected_returns = return_rate * horizon
        
        # Estimate variance (Poisson assumption)
        rental_variance = expected_rentals
        return_variance = expected_returns
        
        return DemandForecast(
            station_id=station_id,
            time=time,
            expected_rentals=expected_rentals,
            expected_returns=expected_returns,
            rental_variance=rental_variance,
            return_variance=return_variance
        )
    
    def update_from_observation(self, station_id: str, time: float, 
                                rentals: int, returns: int):
        """
        Update forecaster with observed demand (online learning).
        
        Args:
            station_id: Station ID
            time: Time of observation
            rentals: Observed rentals
            returns: Observed returns
        """
        # Simple exponential smoothing
        alpha = 0.1  # Learning rate
        
        day_of_week = int(time // (24 * 60)) % 7
        hour_of_day = int((time // 60) % 24)
        key = (day_of_week, hour_of_day)
        
        if station_id not in self.historical_data:
            self.historical_data[station_id] = {}
        
        if key not in self.historical_data[station_id]:
            self.historical_data[station_id][key] = {
                'rentals': rentals,
                'returns': returns
            }
        else:
            old_data = self.historical_data[station_id][key]
            self.historical_data[station_id][key] = {
                'rentals': (1 - alpha) * old_data['rentals'] + alpha * rentals,
                'returns': (1 - alpha) * old_data['returns'] + alpha * returns
            }


class VFAFeatures:
    """
    Compute basis functions (features) for Value Function Approximation.
    
    Implements separable value functions: V̄(S^x) ≈ Σ_n v̄_n(f_n^x)
    """
    
    def __init__(self, forecaster: DemandForecaster, 
                 shortage_penalty: float = 10.0,
                 rejection_penalty: float = 5.0,
                 damage_penalty: float = 3.0,
                 discount_factor: float = 0.95):
        """
        Initialize feature computer.
        
        Args:
            forecaster: Demand forecasting model
            shortage_penalty: Weight for functional shortage feature
            rejection_penalty: Weight for return rejection feature
            damage_penalty: Weight for unattended damage feature
            discount_factor: Spatial discount for en-route vehicles
        """
        self.forecaster = forecaster
        self.shortage_penalty = shortage_penalty
        self.rejection_penalty = rejection_penalty
        self.damage_penalty = damage_penalty
        self.discount_factor = discount_factor
    
    def compute_station_features(self, station: StationInventory, 
                                  state: MDPState,
                                  forecast_horizon: float = 2.0) -> Dict[str, float]:
        """
        Compute basis functions for a single station.
        
        Args:
            station: Station inventory (post-decision state)
            state: Complete MDP state (for vehicle information)
            forecast_horizon: Hours to forecast ahead
        
        Returns:
            Dictionary of feature_name -> feature_value
        """
        features = {}
        
        # Get demand forecast
        forecast = self.forecaster.forecast(
            station.station_id, state.time, forecast_horizon
        )
        
        # === Feature 1: Expected Functional Shortage ===
        # Penalize (shortage)^2 to heavily discourage stockouts
        expected_shortage = max(0, forecast.expected_rentals - station.functional)
        features['functional_shortage'] = expected_shortage ** 2
        
        # === Feature 2: Expected Return Rejections ===
        # Penalize when total bikes + expected returns exceed capacity
        total_bikes = station.total_bikes()
        expected_total = total_bikes + forecast.expected_returns
        expected_overflow = max(0, expected_total - station.capacity)
        features['return_rejection'] = expected_overflow ** 2
        
        # === Feature 3: Unattended Severe Damage ===
        # Raw count of depot bikes left at station (should be removed)
        features['unattended_depot'] = float(station.depot)
        
        # === Feature 4: Unattended Minor Damage ===
        # Count of onsite bikes (can be repaired, less urgent than depot)
        features['unattended_onsite'] = float(station.onsite)
        
        # === Feature 5: Spatial-Routing Synergy ===
        # Value of vehicles en route with functional cargo
        # (Discounted by distance/time to arrival)
        inbound_functional = 0.0
        for vehicle in state.vehicles.values():
            if vehicle.current_station == station.station_id:
                # Vehicle is at or heading to this station
                time_to_arrival = max(0, vehicle.arrival_time - state.time)
                distance_discount = self.discount_factor ** (time_to_arrival / 60.0)  # Discount per hour
                inbound_functional += vehicle.functional_cargo * distance_discount
        
        features['inbound_rebalancing'] = inbound_functional
        
        # === Feature 6: Capacity Utilization ===
        # Penalize extreme utilization (both very full and very empty)
        utilization = total_bikes / station.capacity if station.capacity > 0 else 0
        # Target 50% utilization, penalize deviation
        target_utilization = 0.5
        features['utilization_deviation'] = (utilization - target_utilization) ** 2
        
        return features
    
    def compute_state_features(self, state: MDPState, 
                               forecast_horizon: float = 2.0) -> Dict[str, float]:
        """
        Compute features for entire network state (sum of station features).
        
        This implements the separable approximation:
            V̄(S^x) = Σ_n v̄_n(f_n^x) = Σ_n Σ_f θ_f φ_f(f_n^x)
        
        Args:
            state: Complete MDP state (post-decision)
            forecast_horizon: Hours to forecast ahead
        
        Returns:
            Dictionary of aggregated features
        """
        # Initialize aggregated features
        agg_features = {}
        
        # Sum features across all stations
        for station_id, station in state.stations.items():
            station_features = self.compute_station_features(
                station, state, forecast_horizon
            )
            
            for feature_name, feature_value in station_features.items():
                if feature_name not in agg_features:
                    agg_features[feature_name] = 0.0
                agg_features[feature_name] += feature_value
        
        # Add vehicle-level features
        total_vehicle_load = sum(v.total_cargo() for v in state.vehicles.values())
        agg_features['total_vehicle_load'] = float(total_vehicle_load)
        
        return agg_features
    
    def feature_vector(self, state: MDPState, 
                      forecast_horizon: float = 2.0) -> np.ndarray:
        """
        Compute feature vector φ(S^x) as numpy array.
        
        Args:
            state: MDP state (post-decision)
            forecast_horizon: Hours to forecast ahead
        
        Returns:
            Feature vector as numpy array
        """
        features = self.compute_state_features(state, forecast_horizon)
        
        # Define consistent feature order
        feature_names = [
            'functional_shortage',
            'return_rejection',
            'unattended_depot',
            'unattended_onsite',
            'inbound_rebalancing',
            'utilization_deviation',
            'total_vehicle_load'
        ]
        
        # Build vector
        vector = np.array([features.get(name, 0.0) for name in feature_names])
        
        return vector
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names."""
        return [
            'functional_shortage',
            'return_rejection',
            'unattended_depot',
            'unattended_onsite',
            'inbound_rebalancing',
            'utilization_deviation',
            'total_vehicle_load'
        ]
    
    def get_num_features(self) -> int:
        """Get number of features."""
        return len(self.get_feature_names())


class FeatureNormalizer:
    """
    Normalize features to improve learning stability.
    
    Uses running statistics (mean and std) to normalize features.
    """
    
    def __init__(self, num_features: int):
        self.num_features = num_features
        self.count = 0
        self.mean = np.zeros(num_features)
        self.m2 = np.zeros(num_features)  # For Welford's algorithm
        self.initialized = False
    
    def update(self, features: np.ndarray):
        """
        Update running statistics with new feature vector.
        
        Uses Welford's online algorithm for numerical stability.
        """
        self.count += 1
        delta = features - self.mean
        self.mean += delta / self.count
        delta2 = features - self.mean
        self.m2 += delta * delta2
        
        if self.count >= 10:
            self.initialized = True
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using current statistics.
        
        Returns:
            Normalized feature vector (z-score)
        """
        if not self.initialized or self.count < 2:
            return features
        
        std = np.sqrt(self.m2 / (self.count - 1))
        std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero
        
        return (features - self.mean) / std
    
    def get_statistics(self) -> Dict[str, np.ndarray]:
        """Get current statistics."""
        std = np.sqrt(self.m2 / max(1, self.count - 1))
        return {
            'mean': self.mean.copy(),
            'std': std,
            'count': self.count
        }
