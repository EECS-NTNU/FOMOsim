"""
 LinearVFAPolicy.py  –  Time-Indexed Linear Value Function Approximation

Implements (strictly no rollout / lookahead logic here):
  1. Feature extraction  φ(S^x)  from the **post-decision state** (numpy-vectorised)
  2. VFA scoring         V(S^x) = θᵀ φ(S^x)
  3. TD(0) weight update:
         θ ← θ + α (r + γ V(S^x_next) − V(S^x_cur)) φ(S^x_cur)

Feature definitions live in policies/sjovik_sund/vfa/vfa_features.py
(FEATURE_NAMES + extract()) and are referenced from this policy.
To add / remove features, edit vfa_features.py and retrain.
"""

from __future__ import annotations

import sys
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque

from policies import action
from policies.sjovik_sund.mdp.reward import RewardCalculator
from policies.sjovik_sund.mdp.candidate_generator import generate_candidates


WORKSPACE_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(WORKSPACE_ROOT))

from policies.policy import Policy
import sim
from policies.sjovik_sund.vfa.vfa_features import (
    get_feature_names as _get_feature_names,
    extract       as _extract_phi,
    as_dict       as _phi_as_dict,
)
from policies.sjovik_sund.mdp.mdp_config import MDPConfig
from policies.sjovik_sund.mdp.mdp_formulation import (
    MdpAction,
    extract_mdp_state,
    extract_vehicle_status,
)
from policies.sjovik_sund.mdp.action_bridge import mdp_action_to_sim_action
from settings import ENABLE_COMPONENT_FAILURES
from sim.Bike import Bike

# ─────────────────────────────────────────────────────────────────────────────
# LinearVFAPolicy
# ─────────────────────────────────────────────────────────────────────────────

class LinearVFAPolicy(Policy):
    """
    Time-indexed linear VFA policy for joint rebalancing and maintenance.

    The policy implements:
      - Lazy initialisation from sim.State on the first decision call
        (because Simulator.init() does not call policy.init_sim())
      - Post-decision feature vector extraction (numpy-vectorised)
      - TD(0) online weight updates during the learning phase
      - save / load for the trained θ vector

    Parameters that change across training episodes (learning_mode) can
    be updated externally between episodes by the training loop.
    """

    # Number of next-station candidates evaluated per decision
    N_CANDIDATES: int = 8

    def __init__(
        self,
        active_features: Optional[List[str]] = None,
        n_features: Optional[int] = None,   # defaults to len(FEATURE_NAMES); set explicitly to override
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.0,
        learning_mode: bool = True,
        config: Optional[MDPConfig] = None,
        seed: int = 42,
        maintenance_enabled: bool = ENABLE_COMPONENT_FAILURES,
        shift_timing_enabled: bool = False,
        temporal_enabled: bool = False,
        log_depot_visits: bool = False,
        depot_log_file: Optional[str] = None,
        reward_calculator: Optional[RewardCalculator] = None,
    ) -> None:
        """
        Initializes the policy, sets hyperparameters (alpha, gamma, epsilon),
        and creates the weight vector (theta) which represents the
        agent's learned knowledge.
        """

        super().__init__(maintenance_enabled=maintenance_enabled)

        self.maintenance_enabled = maintenance_enabled
        self.shift_timing_enabled = shift_timing_enabled
        self.temporal_enabled = temporal_enabled
        self.log_depot_visits = log_depot_visits
        self.depot_log_file = depot_log_file
        # 1. Get the canonical list of ALL possible features
        self.ALL_FEATURE_NAMES = _get_feature_names(True, True, True)

        # 2. Determine which features we are actually using
        if active_features is None:
            self.FEATURE_NAMES = _get_feature_names(self.maintenance_enabled, self.shift_timing_enabled, self.temporal_enabled)
        else:
            # --- FIXED: Force the user's active features into canonical mathematical order ---
            self.FEATURE_NAMES = [name for name in self.ALL_FEATURE_NAMES if name in active_features]

        # 3. Create a boolean mask to slice the numpy array extremely fast
        self._feature_mask = np.array([
            (name in self.FEATURE_NAMES) for name in self.ALL_FEATURE_NAMES
        ], dtype=bool)
        
        # --- DEBUG 1: THE SCRAMBLER ---
        actual_math_order = [name for name in self.ALL_FEATURE_NAMES if name in self.FEATURE_NAMES]
        if self.FEATURE_NAMES != actual_math_order:
            print(f"\n[DEBUG - SCRAMBLER] WARNING: FEATURE MISMATCH!")
            print(f"You requested this order : {self.FEATURE_NAMES}")
            print(f"The math outputs this order: {actual_math_order}\n")
            
        self.N_FEATURES = len(self.FEATURE_NAMES)
        self.reward_calc = reward_calculator or RewardCalculator(gamma=gamma)

        if n_features is None:
            n_features = self.N_FEATURES

        if n_features != self.N_FEATURES:
            raise ValueError(
                f"n_features={n_features} but FEATURE_NAMES has "
                f"{self.N_FEATURES} entries. "
                f"Update FEATURE_NAMES when adding or removing features."
            )

        self.n_features    = n_features
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.learning_mode = learning_mode
        default_config = MDPConfig.full_maintenance() if maintenance_enabled else MDPConfig.no_maintenance()
        self.config        = config or default_config  # defaults to full maintenance
        self._rng          = np.random.default_rng(seed)

        # ── Parameter vector θ (small random initialisation) ─────────────────
        self.theta: np.ndarray = np.zeros(n_features, dtype=np.float64)

        # weights attribute forwarded by run_simulation.py for logging
        self.weights: List[float] = list(self.theta)

        # ── Caches – filled on first call (lazy init) ─────────────────────────
        self._initialized: bool              = False
        self._station_ids: List[str]         = []
        self._sid_to_idx:  Dict[str, int]    = {}
        
        # Formally declare all lazy arrays as Optional so Pylance tracks them
        self._target_matrix:       Optional[np.ndarray] = None
        self._leave_profile:       Optional[np.ndarray] = None
        self._arrive_profile:      Optional[np.ndarray] = None
        self._travel_time_matrix:  Optional[np.ndarray] = None
        self._capacities:          Optional[np.ndarray] = None
        self._lambda_max_system:   Optional[float]      = None
        self._max_gravity:         Optional[float]      = None
        self._depot_id:            Optional[str]        = None
        self._N_stations:          Optional[int]        = None
        self._total_fleet_size:    Optional[int]        = None

        # ── Per-episode TD tracking ───────────────────────────────────────────
        self._prev_phi:          Optional[np.ndarray] = None

        # Buffer for Synchronous Batch Learning
        self.batch_buffer: List[Tuple[np.ndarray, float, np.ndarray, float]] = []
        
        # Buffer for Experience Replay (Mini-Batch SGD)
        self.use_experience_replay = False
        self.replay_buffer = deque(maxlen=2000)
        self.mini_batch_size = 64

        # logging for RL decisions (e.g., depot visits)
        self.log_rl_decisions = True  
        self.rl_logs = []
        
        # Cache for static fleet size to optimize VFA speed
        self.cached_fleet_size = None

        # Simulator reference — set by init_sim(), used to pre-warm target matrix
        self._simulator_ref = None

    def init_sim(self, simulator) -> None:
        """
        Called by Simulator.init() after the event queue is built.
        Stores the simulator reference so _lazy_init can access the
        target_state object and pre-warm the full 7×24 target matrix.
        """
        self._simulator_ref = simulator

    # ─────────────────────────────────────────────────────────────────────────
    # Lazy initialisation  (uses sim.State, not the full simulator)
    # ─────────────────────────────────────────────────────────────────────────

    def _lazy_init(self, state) -> None:
        """
        Build station metadata caches from sim.State.
        Pre-calculates static matrices and scaling bounds (Lambda_max, G_max) 
        to keep the online VFA evaluation strictly O(1) or fast O(N).
        """
        stations = sorted(state.get_stations(), key=lambda s: s.id)
        N = len(stations)

        # Add these new static constants:
        self._N_stations = N
        self._total_fleet_size = len(Bike.created_bike_ids)
        
        # (If checking the depot's queue/repair is needed in your version, add those here too)

        self._station_ids = [s.id for s in stations]
        self._sid_to_idx  = {sid: k for k, sid in enumerate(self._station_ids)}

        # Pre-warm: at _lazy_init time, the event loop has only called
        # update_target_state for the current (day, hour), leaving the other
        # 167 slots as zero in each station's target_state matrix.
        # Drive the target_state object over all 168 combinations now so that
        # the snapshot below is complete.
        if self._simulator_ref is not None and self._simulator_ref.target_state is not None:
            ts_obj = self._simulator_ref.target_state
            for d_pre in range(7):
                for h_pre in range(24):
                    ts_obj.update_target_state(state, d_pre, h_pre)

        # Target inventory matrix  (7 days × 24 hours × N stations)
        self._target_matrix = np.array(
            [[[s.get_target_state(d, h) for s in stations] for h in range(24)] for d in range(7)],
            dtype=np.float32,
        )
        self._capacities = np.array([s.capacity for s in stations], dtype=np.float32)

        # ── 2. Demand tracking — separate leave / arrive profiles ─────────────
        # Shape (2, 24, N): axis 0 = day type (0=weekday, 1=weekend),
        #                   axis 1 = hour (0-23), axis 2 = station index.
        weekday_days = [0, 1, 2, 3, 4]
        weekend_days = [5, 6]

        self._leave_profile  = np.zeros((2, 24, N), dtype=np.float32)
        self._arrive_profile = np.zeros((2, 24, N), dtype=np.float32)

        for i, s in enumerate(stations):
            for h in range(24):
                self._leave_profile[0, h, i]  = np.mean([s.get_leave_intensity(d, h)  for d in weekday_days])
                self._leave_profile[1, h, i]  = np.mean([s.get_leave_intensity(d, h)  for d in weekend_days])
                self._arrive_profile[0, h, i] = np.mean([s.get_arrive_intensity(d, h) for d in weekday_days])
                self._arrive_profile[1, h, i] = np.mean([s.get_arrive_intensity(d, h) for d in weekend_days])

        # Find the absolute peak 3-hour rolling net demand for scaling (Lambda_max)
        max_abs_3hr_activity = np.zeros(N, dtype=np.float32)
        for day_type in range(2):
            for h in range(24):
                three_hr_demand = (
                    np.abs(self._leave_profile[day_type, h]        - self._arrive_profile[day_type, h]) +
                    np.abs(self._leave_profile[day_type, (h+1)%24] - self._arrive_profile[day_type, (h+1)%24]) +
                    np.abs(self._leave_profile[day_type, (h+2)%24] - self._arrive_profile[day_type, (h+2)%24])
                )
                max_abs_3hr_activity = np.maximum(max_abs_3hr_activity, three_hr_demand)
                
        self._lambda_max_system = float(np.sum(max_abs_3hr_activity))
        
        # ── 3. Travel Time Matrix (N x N) ──────────────────────────────────────
        self._travel_time_matrix = np.zeros((N, N), dtype=np.float32)
        for i, s_from in enumerate(stations):
            for j, s_to in enumerate(stations):
                self._travel_time_matrix[i, j] = state.get_travel_time(s_from.id, s_to.id)

        # ── 4. G_max (Maximum Theoretical Gravity) ─────────────────────────────
        best_gravity = 0.0
        for j in range(N):
            # Calculate gravity using the absolute maximum 3-hour demand
            current_gravity = np.sum(max_abs_3hr_activity / (self._travel_time_matrix[j] + 1.0))
            if current_gravity > best_gravity:
                best_gravity = current_gravity
        
        self._max_gravity = float(best_gravity)

        # Closest-depot ID
        vehicles = state.get_vehicles()
        self._depot_id = state.get_closest_depot(vehicles[0]) if vehicles else None
        
        self.cached_fleet_size = float(len(state.get_all_bikes()))

        self._initialized = True

    # ─────────────────────────────────────────────────────────────────────────
    # Inventory extraction
    # ─────────────────────────────────────────────────────────────────────────
   
    def _extract_inventories(
        self, state, vehicle
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Takes a snapshot of the city directly from the true simulator state,
        bypassing the MDP extraction layer to ensure broken bikes are 
        accurately mapped.
        """
        N = len(self._station_ids)
        func = np.zeros(N, dtype=np.int32)
        onsite = np.zeros(N, dtype=np.int32)
        depot = np.zeros(N, dtype=np.int32)

        for sid, station in state.stations.items():
            if sid not in self._sid_to_idx:
                continue

            # --- INJECT DEBUG BLOCK 1: TYPE CHECK ---
            if not hasattr(self, '_debug_type_check'):
                print(f"\n[DEBUG 1 - DATA TYPE] Investigating station.bikes structure...")
                print(f"Type of station.bikes: {type(station.bikes)}")
                if isinstance(station.bikes, dict):
                    print(f"It is a dict! Example keys: {list(station.bikes.keys())[:3]}")
                elif isinstance(station.bikes, list):
                    print(f"WARNING: It is a list (length {len(station.bikes)})!")
                    print(f"Your code `station.bikes if isinstance(station.bikes, dict) else` will wipe this to 0!")
                self._debug_type_check = True
            # ----------------------------------------

            k = self._sid_to_idx[sid]
            
            # Extract true physical inventory directly from the station objects
            station_bikes = station.bikes.values() if isinstance(station.bikes, dict) else station.bikes
            for bike in station_bikes:
                status = getattr(bike, 'damage_status', None)
                if status == 'onsite':
                    onsite[k] += 1
                elif status == 'depot':
                    depot[k] += 1
                else:
                    func[k] += 1

        # --- FIXED DEBUG PRINT ---
        # Print it once and then set a flag so it doesn't spam you forever
        if not hasattr(self, '_has_printed_vision'):
            print(f"\n[DEBUG - VFA VISION] VFA sees {sum(func)} functional, {sum(onsite)} onsite, {sum(depot)} depot bikes in the city.")
            self._has_printed_vision = True

        return func, onsite, depot

    # ─────────────────────────────────────────────────────────────────────────
    # Feature vector  φ(S^x)  – delegated to vfa_features.py
    # ─────────────────────────────────────────────────────────────────────────

    def extract_features(
        self,
        state,
        vehicle,
        func: np.ndarray,      
        onsite: np.ndarray,    
        depot: np.ndarray,     
        delta_func: int = 0,
        delta_depot_cargo: int = 0,
        delta_onsite_repairs: int = 0,
        time_remaining: Optional[float] = None,
        shift_length: float = 1440.0,
        next_station_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute φ(S^x) for the post-decision state.
        """
        # Explicitly tell Pylance the lazy caches are populated
        assert self._target_matrix is not None
        assert self._leave_profile is not None
        assert self._arrive_profile is not None
        assert self._travel_time_matrix is not None
        assert self._capacities is not None
        assert self._lambda_max_system is not None
        assert self._max_gravity is not None
        assert self._N_stations is not None
        assert self.cached_fleet_size is not None

        func_post = func.copy()
        onsite_post = onsite.copy()

        #TODO: Fix so that the values on car is always integer
        
        # ── Apply post-decision delta at the vehicle's current station ─────
        if vehicle.location.id in self._sid_to_idx:
            cur_idx = self._sid_to_idx[vehicle.location.id]
            func_post[cur_idx] = max(0, func_post[cur_idx] + delta_func)
            if delta_onsite_repairs:
                repairs = max(0, int(delta_onsite_repairs))
                applied_repairs = min(repairs, int(onsite_post[cur_idx]))
                onsite_post[cur_idx] = max(0, onsite_post[cur_idx] - applied_repairs)
                func_post[cur_idx] += applied_repairs

        # ── Vehicle cargo in post-decision state ───────────────────────────
        vehicle_status = extract_vehicle_status(vehicle, state.time, self.config)
        # Note: delta_func is (deliveries - pickups), so it represents the change to the STATION's inventory.
        # Therefore, we MUST SUBTRACT it to get the change to the VEHICLE's inventory.
        func_cargo_veh = vehicle_status.functional_cargo - delta_func
        
        depot_cargo_veh = vehicle_status.depot_cargo + delta_depot_cargo
        K = max(int(vehicle_status.capacity), 1)

        # ── Anticipate the inventory change at the DESTINATION ─────────────
        if next_station_id and next_station_id in self._sid_to_idx:
            nxt_idx = self._sid_to_idx[next_station_id]
            d, h = state.day() % 7, state.hour() % 24
            target_nxt = self._target_matrix[d, h, nxt_idx]
            cur_nxt = func_post[nxt_idx]
            delta_nxt = target_nxt - cur_nxt
            
            if delta_nxt > 0: # Starving
                delivery = min(func_cargo_veh, delta_nxt)
                func_post[nxt_idx] += delivery
                func_cargo_veh -= delivery
            elif delta_nxt < 0: # Congested
                free_cap = max(0, K - (func_cargo_veh + depot_cargo_veh))
                pickup = min(-delta_nxt, free_cap, cur_nxt)
                func_post[nxt_idx] -= pickup
                func_cargo_veh += pickup
        
        # ── Rolling 3-Hour Demand Anticipation ─────────────────────────────
        d = state.day() % 7
        day_type = 1 if d in [5, 6] else 0  # 0 = weekday, 1 = weekend

        current_minute = int(state.time % 60)
        hours   = [(int((state.time // 60) % 24) + k) % 24 for k in range(4)]
        weights = np.array([
            (60 - current_minute) / 60.0,  # remaining fraction of current hour
            1.0,                            # full next hour
            1.0,                            # full hour after that
            current_minute / 60.0,          # overlap into fourth hour
        ])

        dynamic_leave  = np.array([self._leave_profile[day_type, h]  * w for h, w in zip(hours, weights)])
        dynamic_arrive = np.array([self._arrive_profile[day_type, h] * w for h, w in zip(hours, weights)])

        # ── Time-indexed target inventory (N,) ─────────────────────────────
        d, h   = state.day() % 7, state.hour() % 24
        target = self._target_matrix[d, h]              

        # ── Anticipated Distances (from Destination) ────────────────
        # Evaluate spatial gravity from where the vehicle is GOING, not where it is!
        routing_target = next_station_id if next_station_id else vehicle.location.id
        
        dist_to_depot = (
            state.get_travel_time(routing_target, self._depot_id)
            if self._depot_id and self._depot_id != routing_target
            else 0.0
        )
        
        # Fast spatial slice from cached matrix using the routing target
        if routing_target in self._sid_to_idx:
            v_idx = self._sid_to_idx[routing_target]
            dist_to_stations = self._travel_time_matrix[v_idx]
        else:
            dist_to_stations = np.array([
                state.get_travel_time(routing_target, sid) 
                for sid in self._station_ids
            ], dtype=np.float32)

        # ── Delegate to canonical feature extractor ────────────────────────
        phi_full = _extract_phi(
            func=func_post.astype(np.float64),
            onsite=onsite_post.astype(np.float64),
            depot=depot.astype(np.float64),
            target=target.astype(np.float64),
            capacities=self._capacities,
            leave_activity=dynamic_leave.astype(np.float64),
            arrive_activity=dynamic_arrive.astype(np.float64),
            dist_to_stations=dist_to_stations,
            func_cargo_veh=float(func_cargo_veh),
            depot_cargo_veh=float(depot_cargo_veh),
            vehicle_capacity=K,
            dist_to_depot=dist_to_depot,
            lambda_max_system=self._lambda_max_system,
            max_gravity=self._max_gravity,
            fleet_size=self.cached_fleet_size,
            maintenance_enabled=True,
            shift_timing_enabled=True,
            time_remaining=self._get_time_remaining(state, vehicle),
            shift_length=self._get_shift_length(state, vehicle),
            temporal_enabled=True,
            current_time_minutes=float(state.time),
            current_day_of_week=int(state.day() % 7),
            target_matrix=self._target_matrix,
            total_stations=self._N_stations
        )
    
        # 4. Slice the full feature vector to only include active features
        phi = phi_full[self._feature_mask]

        return phi

    # ─────────────────────────────────────────────────────────────────────────
    # Value function
    # ─────────────────────────────────────────────────────────────────────────

    def features_as_dict(
        self,
        state,
        vehicle,
        delta_func: int = 0,
        delta_depot_cargo: int = 0,
    ) -> dict:
        """
        Return the feature vector as a labelled dict — useful for debugging
        and logging individual feature values.

        Example::

            print(policy.features_as_dict(state, vehicle))
            # {'rebalancing_imbalance': 12.0, 'trailer_cannibalization': 0.4, ...}
        """
        # Extract once just for this debug call
        base_func, base_onsite, base_depot = self._extract_inventories(state, vehicle)

        phi = self.extract_features(state, vehicle, base_func, base_onsite, base_depot, delta_func, delta_depot_cargo)
        return dict(zip(self.FEATURE_NAMES, phi.tolist()))

    def value(self, phi: np.ndarray) -> float:
        """V(S^x) = θᵀ φ(S^x)."""
        return float(np.dot(self.theta, phi))

    # ─────────────────────────────────────────────────────────────────────────
    # Shift timing helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_time_remaining(self, state, vehicle) -> Optional[float]:
        """
        Get time remaining in vehicle's shift (minutes).
        
        Returns:
            time_remaining : float (minutes) or None if shift_timing not enabled
        """
        if not self.shift_timing_enabled:
            return None
        
        if hasattr(vehicle, "shift_end_time") and vehicle.shift_end_time is not None:
            return max(0.0, vehicle.shift_end_time - state.time)
            
        if self._simulator_ref is not None and hasattr(self._simulator_ref, "duration"):
            return max(0.0, self._simulator_ref.duration - state.time)
        
        return None
    
    def _get_shift_length(self, state, vehicle) -> float:
        """
        Get reference shift length for normalization (minutes).
        
        Default: 1440 minutes (24 hours).
        Can be customized per vehicle if needed.
        """
        if hasattr(vehicle, "shift_length") and vehicle.shift_length is not None:
            return float(vehicle.shift_length)
        if self._simulator_ref is not None and hasattr(self._simulator_ref, "duration"):
            return float(self._simulator_ref.duration)
            
        # Default 12-hour shift
        return 1220.0

    # ─────────────────────────────────────────────────────────────────────────
    # TD(0) update
    # ─────────────────────────────────────────────────────────────────────────
        
    def td_update(self, reward: float, phi_next: np.ndarray, elapsed_minutes: float) -> float:
        if self._prev_phi is None:
            return 0.0

        # Compute pre-update values for logging
        v_cur = self.value(self._prev_phi)
        v_next = self.value(phi_next)
        
        # Apply continuous time discounting
        discount = self.gamma ** (elapsed_minutes / 60.0)
        td_error = reward + (discount * v_next) - v_cur

        # --- SMART LOGGING ---
        if reward < -0.01 or abs(td_error) > 1.0:
            target_value = reward + discount * v_next
            print(f"    [TD Alert] Reward: {reward:6.3f} | V(S): {v_cur:6.3f} | Target: {target_value:6.3f} | TD Err: {td_error:6.3f}")

        if getattr(self, 'use_experience_replay', False):
            # 1. Experience Replay Logging
            self.replay_buffer.append((self._prev_phi.copy(), reward, phi_next.copy(), elapsed_minutes))
            
            # 2. Trigger mini batch update if buffer has enough experiences (Warmup phase)
            # Typically wait until we have a decent number of samples to break initial correlation
            if len(self.replay_buffer) >= max(1000, self.mini_batch_size):
                self.apply_mini_batch_update()
        else:
            # 1. Store transition for synchronous batch update
            self.batch_buffer.append((self._prev_phi.copy(), reward, phi_next.copy(), elapsed_minutes))

        return td_error

    def apply_mini_batch_update(self) -> None:
        """
        Randomly samples exactly 'mini_batch_size' transitions from the replay buffer
        and updates the weights using plain mini-batch SGD.
        THIS IS ONLY USED WHEN EXPERIENCE REPLAY IS ENABLED. In the standard synchronous batch learning setup,
        this method is not called and the policy relies on apply_batch_update() instead.
        """
        if len(self.replay_buffer) < self.mini_batch_size:
            return

        # Randomly sample a mini-batch
        import random
        batch = random.sample(list(self.replay_buffer), self.mini_batch_size)

        total_gradient = np.zeros_like(self.theta)
        total_td = 0.0

        # Compute gradient over the sampled mini-batch
        for phi_cur, reward, phi_next, elapsed_minutes in batch:
            v_cur = self.value(phi_cur)
            v_next = self.value(phi_next)
            
            # Apply continuous time discounting based on an hourly rate
            discount = self.gamma ** (elapsed_minutes / 60.0)
            td_error = np.clip(reward + (discount * v_next) - v_cur, -5.0, 5.0)

            total_td += td_error
            total_gradient += td_error * phi_cur

        avg_gradient = total_gradient / self.mini_batch_size

        # Plain SGD: no momentum, no weight sign constraint
        self.theta += self.alpha * avg_gradient
        self.weights = list(self.theta)

        # Note: We omit logging here to avoid spamming the console
        # since this triggers at EVERY decision step.

    def apply_batch_update(self) -> None:
        """
        Aggregate accumulated TD errors over frozen trajectories
        and update the value function weights.
        """
        if not self.batch_buffer:
            return

        n_transitions = len(self.batch_buffer)
        total_gradient = np.zeros_like(self.theta)
        total_td = 0.0

       # Compute gradient over the entire buffered batch
        for phi_cur, reward, phi_next, elapsed_minutes in self.batch_buffer:
            v_cur = self.value(phi_cur)
            v_next = self.value(phi_next)
            
            # Apply continuous time discounting based on an hourly rate
            discount = self.gamma ** (elapsed_minutes / 60.0)
            
            # REMOVED CLIPPING: Let the raw reward signal flow through!
            td_error = reward + (discount * v_next) - v_cur

            total_td += td_error
            total_gradient += td_error * phi_cur

        mean_td = total_td / n_transitions

        old_theta = self.theta.copy()

        # FIXED: Accumulate the trajectory's gradient (Sum, not Average)
        sum_gradient = total_gradient
        step = self.alpha * sum_gradient
        self.theta += step

        # Save the updated weights
        self.weights = list(self.theta)

        # --- CONCISE BATCH LOGGING ---
        grad_norm = np.linalg.norm(sum_gradient)
        weight_diff = np.linalg.norm(self.theta - old_theta)
        step_norm = np.linalg.norm(step)

        print(f"    [BATCH UPDATE] Processed {n_transitions} transitions | Mean TD Error: {mean_td:.4f} | Grad Norm: {grad_norm:.4f} | Theta Diff: {weight_diff:.4f} | Step Norm: {step_norm:.4f}")

        # Dump buffer for the next trajectory sync phase
        self.batch_buffer.clear()
    # ─────────────────────────────────────────────────────────────────────────
    # Action generation  (action-space splitting)
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_candidates(self, state, vehicle) -> List[sim.Action]:
        """
        DECOUPLED CANDIDATE GENERATION:
        Delegated to candidate_generator.py to create combination of operations and routing.
        """
        return generate_candidates(state, vehicle, self.maintenance_enabled)

    # ─────────────────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────────────────

    def _log_depot_decision(self, state, vehicle, selected_action, phi, value) -> None:
        """
        Log when vehicle decides to go to the depot.
        
        Tracks:
          - Current location, inventory, time
          - Destination (depot or station)
          - Value function and key feature values
          - Time remaining in shift (if applicable)
        
        Outputs to:
          1. Console (always, if log_depot_visits=True)
          2. File (if depot_log_file is set)
        """
        if not self.log_depot_visits or self._depot_id is None:
            return
        
        # Check if selected action is going to depot
        destination_id = getattr(selected_action, "destination_station", None) \
                      or getattr(selected_action, "next_station", None)
        
        if destination_id != self._depot_id:
            return  # Not going to depot, no log needed
            
        if vehicle.location.id == self._depot_id:
            return  # Already at depot (avoid spamming logs overnight while waiting for the next shift)
        
        # Construct log entry
        time_rem = self._get_time_remaining(state, vehicle)
        time_frac = time_rem / 1440.0 if time_rem is not None else None
        
        log_entry = (
            f"[DEPOT] t={state.time:7.1f}min | "
            f"vehicle={vehicle.id} | "
            f"from={vehicle.location.id} → to={destination_id} | "
            f"cargo_func={len(vehicle.get_bike_inventory())} | "
            f"V(S^x)={value:8.4f} | "
        )
        
        if self.shift_timing_enabled:
            log_entry += f"t_rem={time_rem:.1f}min ({time_frac:.2%}) | "
        
        log_entry += f"φ_1={phi[0]:.4f} "  # rebalancing_imbalance
        
        if self.maintenance_enabled:
            log_entry += f"φ_4={phi[3]:.4f} "  # trailer_cannibalization
            idx_time = 7  # after maintenance features
        else:
            idx_time = 3

        if self.shift_timing_enabled and len(phi) > idx_time:
            log_entry += f"φ_time={phi[idx_time]:.4f} φ_penalty={phi[idx_time+1]:.4f}"
        
        # Print to console
        print(log_entry)
        
        # Write to file if specified
        if self.depot_log_file:
            try:
                with open(self.depot_log_file, "a") as f:
                    f.write(log_entry + "\n")
            except IOError as e:
                print(f"Warning: could not write to depot log file {self.depot_log_file}: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Main decision entry point
    # ─────────────────────────────────────────────────────────────────────────

    def get_best_action(self, state, vehicle) -> sim.Action:
    #def get_best_action(self, state, vehicle) -> sim.Action:
        """
        Called by VehicleArrival event at each decision epoch.

        Decision flow
        ─────────────
        1. Lazy-initialise caches from sim.State (first call only).
        2. Generate N_CANDIDATES candidate actions.
        3. Compute φ(S^x_a) and V(S^x_a) = θᵀφ for each candidate.
        4. (If learning) run TD(0) update using reward since last decision
           and the greedy next post-decision state as the bootstrap target.
        5. Select action greedy (exploitation).
        6. Cache the selected action's post-decision features for step 4
           of the next call.
        """
        # ── Step 1: lazy init ─────────────────────────────────────────────
        if not self._initialized:
            self._lazy_init(state)

        # ── Step 2: generate candidate actions ────────────────────────────
        candidates = self._generate_candidates(state, vehicle)

        base_func, base_onsite, base_depot = self._extract_inventories(state, vehicle)
        
        # ── Step 3: score each candidate ──────────────────────────────────
        phis: List[np.ndarray] = []
        values = np.empty(len(candidates), dtype=np.float64)
        
        for k, action in enumerate(candidates):
            # --- FIXED: Safely map bike IDs to objects for fast lookup ---
            raw_bikes = getattr(vehicle.location, "bikes", [])
            if isinstance(raw_bikes, dict):
                station_bikes = raw_bikes
            else:
                station_bikes = {getattr(b, 'bike_id', getattr(b, 'id')): b for b in raw_bikes}
                
            functional_pickups = 0
            depot_pickups = 0
            
            for b_id in action.pick_ups:
                b = station_bikes.get(b_id)
                if b and getattr(b, 'damage_status', None) == 'depot':
                    depot_pickups += 1
                elif b:
                    functional_pickups += 1
            
            # --- DEBUG PRINT ---
            # Print only for the very first candidate of the decision epoch so it doesn't flood the console
            if k == 0 and len(action.pick_ups) > 0:
                 print(f"[DEBUG - CARGO] Action wanted {len(action.pick_ups)} pickups. VFA correctly identified {functional_pickups} functional and {depot_pickups} depot bikes.")

            # Net change in functional bikes and vehicle depot cargo
            delta_func = len(action.delivery_bikes) - functional_pickups
            delta_depot_cargo = depot_pickups
            delta_onsite_repairs = len(getattr(action, "onsite_repairs", []))
            
            # If vehicle is at depot, ALL depot cargo is unloaded
            if vehicle.is_at_depot():
                vehicle_depot_cargo = sum(1 for b in vehicle.get_bike_inventory() if getattr(b, 'damage_status', None) == 'depot')
                delta_depot_cargo = -vehicle_depot_cargo
            
            dest_id = getattr(action, "next_location", getattr(action, "next_station", None))

            time_rem = self._get_time_remaining(state, vehicle)
            shift_len = self._get_shift_length(state, vehicle)

            phi = self.extract_features(
                state, vehicle, 
                base_func, base_onsite, base_depot, 
                delta_func, delta_depot_cargo,
                delta_onsite_repairs,
                time_remaining=time_rem,
                shift_length=shift_len,
                next_station_id=dest_id
            )
            phis.append(phi)
            values[k] = self.value(phi)    
                    
        # --- DEBUG 2: SPATIAL PARALYSIS ---
        if not getattr(self, "_has_printed_spatial", False) and "proximity_to_demand_gravity" in self.FEATURE_NAMES:
            idx = self.FEATURE_NAMES.index("proximity_to_demand_gravity")
            print(f"\n[DEBUG - SPATIAL] Gravity values for 8 candidates from {vehicle.location.id}:")
            for k, action in enumerate(candidates):
                dest = getattr(action, "next_location", getattr(action, "next_station", None))
                print(f"  -> Going to {dest} | Gravity Feature: {phis[k][idx]:.6f}")
            self._has_printed_spatial = True

        # ── Step 4: TD(0) update ──────────────────────────────────────────
        td_err = 0.0
        if self.learning_mode and self._prev_phi is not None:
            # Consume the reward signal safely
            reward = self.reward_calc.compute_step_reward(state.metrics)
            
            elapsed_minutes = state.time - self._prev_time
            
            # FIXED: We expect to take the BEST action next, so use argmax!
            phi_next = phis[int(np.argmax(values))]
            td_err = self.td_update(reward, phi_next, elapsed_minutes)
            
        elif self.learning_mode and self._prev_phi is None:
            _ = self.reward_calc.compute_step_reward(state.metrics)
            # Freeze the exact warmup totals here!
            self.warmup_starvations_snapshot = self.reward_calc._prev_starvations
            self.warmup_congestions_snapshot = self.reward_calc._prev_congestions
            self.warmup_trips_snapshot = self.reward_calc._prev_trips

        # ── Step 5: select best action (epsilon-greedy) ──────────────────
        if self.learning_mode and self.epsilon > 0.0 and self._rng.random() < self.epsilon:
            # Exploration: pick a random action from the candidate pool
            sel_idx = self._rng.integers(len(candidates))
            print(f"4. Exploring: randomly selected action index {sel_idx} with value {values[sel_idx]:.4f}")
        else:
            # Exploitation: pick the action with maximum value
            sel_idx = int(np.argmax(values))
            print(f"4. Exploiting: selected best action index {sel_idx} with value {values[sel_idx]:.4f}")

        selected = candidates[sel_idx]
        
        print(f"4. Selected action: next station: {getattr(selected, 'next_location', getattr(selected, 'next_station', None))}, pickups: {len(getattr(selected, 'pick_ups', []))}, deliveries: {len(getattr(selected, 'delivery_bikes', []))}")

        # ── Step 6: Log depot decisions (optional) ────────────────────────
        self._log_depot_decision(state, vehicle, selected, phis[sel_idx], values[sel_idx])

        # ── Step 7: cache post-decision features for next TD update ───────
        self._prev_phi = phis[sel_idx]
        self._prev_time = state.time

        # ── Step 8: Log the Brain's Decision (NEW) ────────────────────────
        if getattr(self, 'log_rl_decisions', False):
            # Get the human-readable features for the winning action
            phi_dict = dict(zip(self.FEATURE_NAMES, phis[sel_idx].tolist()))
            
            log_entry = {
                'time': state.time,
                'vehicle_id': vehicle.id,
                'station_id': vehicle.location.id,
                'action_next_station': getattr(selected, 'next_location', 'N/A'),
                'action_pickups': len(getattr(selected, 'pick_ups', [])),
                'action_deliveries': len(getattr(selected, 'delivery_bikes', [])),
                'expected_value_V': values[sel_idx],
                'td_error': float(td_err),  # We just use the safe variable from Step 4!
            }
            # Merge the feature dictionary into the log entry
            log_entry.update(phi_dict)
            self.rl_logs.append(log_entry)

        return selected

    # ─────────────────────────────────────────────────────────────────────────
    # Episode boundary reset  (called by EpisodeTrainingPolicy.__init__)
    # ─────────────────────────────────────────────────────────────────────────

    def reset_episode(self) -> None:
        self._prev_phi = None
        self._prev_time = None  # New line
        self.reward_calc.reset_episode()
    # ─────────────────────────────────────────────────────────────────────────
    # Serialisation
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist θ and hyper-parameters to disk."""
        payload = {
            "theta":       self.theta,
            "n_features":  self.n_features,
            "alpha":       self.alpha,
            "gamma":       self.gamma,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"  [VFA] theta saved -> {path}")

    @classmethod
    def load(cls, path: Path, **kwargs) -> "LinearVFAPolicy":
        #Load a trained model from disk (learning_mode=False by default).
        with open(path, "rb") as f:
            payload = pickle.load(f)
        policy         = cls(
            n_features   = payload["n_features"],
            alpha        = payload["alpha"],
            gamma        = payload["gamma"],
            learning_mode= False,
            **kwargs,
        )
        policy.theta   = payload["theta"]
        policy.weights = list(policy.theta)
        return policy


# ─────────────────────────────────────────────────────────────────────────────
# EpisodeTrainingPolicy  –  warm-up / learning phase router
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeTrainingPolicy(Policy):
    """
    Episodic wrapper that routes vehicle decisions to:

      - Warmup Policy       during the warm-up phase  (no TD updates)
      - LinearVFAPolicy     during the learning phase (TD(0))

    The phase transition is time-based: once state.time >= warmup_end_time
    (an absolute simulation-minutes value), the VFA policy takes control.

    A new instance of this class is created for every episode, but the
    underlying LinearVFAPolicy (and its θ) is shared and persists.
    """

    def __init__(
        self,
        vfa_policy:     LinearVFAPolicy,
        warmup_policy,
        warmup_end_time: float,
    ) -> None:
        super().__init__(maintenance_enabled=True)

        self.vfa_policy      = vfa_policy
        self.warmup_policy   = warmup_policy
        self.warmup_end_time = warmup_end_time

        # Mirror the VFA weights for run_simulation.py logging
        self.weights = vfa_policy.weights

        # Reset per-episode TD state on the shared VFA policy
        vfa_policy.reset_episode()
    
    def init_sim(self, simulator) -> None:
        """Forward simulator reference to both inner policies."""
        self.vfa_policy.init_sim(simulator)
        self.warmup_policy.init_sim(simulator)

    def get_best_action(self, state, vehicle) -> sim.Action:
        if state.time < self.warmup_end_time:
            # Warm-up: pure exploitation policy, θ left unchanged
            return self.warmup_policy.get_best_action(state, vehicle)
        else:
            # Learning: VFA + TD(0) update
            return self.vfa_policy.get_best_action(state, vehicle)

    def __repr__(self) -> str:
        return f"EpisodeTrainingPolicy()"
