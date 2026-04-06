"""
LinearVFAPolicy.py  –  Time-Indexed Linear Value Function Approximation

Implements (strictly no rollout / lookahead logic here):
  1. Feature extraction  φ(S^x)  from the **post-decision state** (numpy-vectorised)
  2. VFA scoring         V(S^x) = θᵀ φ(S^x)
  3. Boltzmann (softmax) action selection  P(a) ∝ exp(−V(S^x_a) / τ)
  4. TD(0) weight update:
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

from policies import action
from policies import action
from policies.sjovik_sund.mdp.reward import RewardCalculator


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
      - Boltzmann action selection during training
      - TD(0) online weight updates during the learning phase
      - save / load for the trained θ vector

    Parameters that change across training episodes (τ, learning_mode) can
    be updated externally between episodes by the training loop.
    """

    # Number of next-station candidates evaluated per decision
    N_CANDIDATES: int = 8

    def __init__(
        self,
        active_features: List[str] = None,
        n_features: int = None,   # defaults to len(FEATURE_NAMES); set explicitly to override
        alpha: float = 0.01,
        gamma: float = 0.99,
        tau: float = 5.0,
        learning_mode: bool = True,
        config: Optional[MDPConfig] = None,
        seed: int = 42,
        maintenance_enabled: bool = ENABLE_COMPONENT_FAILURES,
        shift_timing_enabled: bool = False,
        log_depot_visits: bool = False,
        depot_log_file: Optional[str] = None,
        reward_calculator: Optional[RewardCalculator] = None,
        
    ) -> None:
        
        """
        Initializes the policy, sets hyperparameters (alpha, gamma, tau), 
        and creates the weight vector (theta) which represents the 
        agent's learned knowledge.
        """

        super().__init__(maintenance_enabled=maintenance_enabled)
        
        self.maintenance_enabled = maintenance_enabled
        self.shift_timing_enabled = shift_timing_enabled
        self.log_depot_visits = log_depot_visits
        self.depot_log_file = depot_log_file
        # 1. Get the canonical list of ALL possible features
        self.ALL_FEATURE_NAMES = _get_feature_names(True, True)

        # 2. Determine which features we are actually using
        if active_features is None:
            self.FEATURE_NAMES = _get_feature_names(self.maintenance_enabled, self.shift_timing_enabled)
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
        self.tau           = tau            # Boltzmann temperature – set by training loop
        self.learning_mode = learning_mode
        default_config = MDPConfig.full_maintenance() if maintenance_enabled else MDPConfig.no_maintenance()
        self.config        = config or default_config  # defaults to full maintenance
        self._rng          = np.random.default_rng(seed)

        # ── Parameter vector θ (small random initialisation) ─────────────────
        #self.theta: np.ndarray = self._rng.standard_normal(n_features) * 0.01
        # ── Parameter vector θ (start completely blind for ablation) ─────────
        self.theta: np.ndarray = np.zeros(n_features, dtype=np.float64)

        # weights attribute forwarded by run_simulation.py for logging
        self.weights: List[float] = list(self.theta)

        # ── Caches – filled on first call (lazy init) ─────────────────────────
        self._initialized: bool              = False
        self._station_ids: List[str]         = []
        self._sid_to_idx:  Dict[str, int]    = {}
        self._target_matrix: Optional[np.ndarray] = None  # shape (7, 24, N)
        self._activity:      Optional[np.ndarray] = None  # shape (N,)  avg λ_i
        self._depot_id:      Optional[str]        = None

        # ── Per-episode TD tracking ───────────────────────────────────────────
        self._prev_phi:          Optional[np.ndarray] = None
        self._prev_starvations:  int   = 0
        self._prev_congestions:  int   = 0

        # logging for RL decisions (e.g., depot visits)
        self.log_rl_decisions = True  
        self.rl_logs = []

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

        self._station_ids = [s.id for s in stations]
        self._sid_to_idx  = {sid: k for k, sid in enumerate(self._station_ids)}

        # Target inventory matrix  (7 days × 24 hours × N stations)
        self._target_matrix = np.array(
            [[[s.get_target_state(d, h) for s in stations] for h in range(24)] for d in range(7)],
            dtype=np.float32,
        )
        self._capacities = np.array([s.capacity for s in stations], dtype=np.float32)

        # ── 2. Demand tracking (Weekday/Weekend Profiles) ──────────────────────
        # Create a (2, 24, N) profile: Index 0 = Weekday, Index 1 = Weekend
        self._activity_profile = np.zeros((2, 24, N), dtype=np.float32)
        
        # Note: Standard datetime uses 0=Mon, 4=Fri, 5=Sat, 6=Sun. 
        # Adjust these arrays if your simulator uses a different day index.
        weekday_days = [0, 1, 2, 3, 4] 
        weekend_days = [5, 6]          

        for i, s in enumerate(stations):
            for h in range(24):
                #wd_rates = [s.get_arrive_intensity(d, h) for d in weekday_days]
                #we_rates = [s.get_arrive_intensity(d, h) for d in weekend_days]
                wd_rates = [
                    s.get_leave_intensity(d, h) - s.get_arrive_intensity(d, h)
                    for d in weekday_days
                ]
                we_rates = [
                    s.get_leave_intensity(d, h) - s.get_arrive_intensity(d, h)
                    for d in weekend_days
        ]
                self._activity_profile[0, h, i] = np.mean(wd_rates)
                self._activity_profile[1, h, i] = np.mean(we_rates)
        
        # Find the absolute peak 2-hour rolling demand for scaling (Lambda_max)
        max_abs_2hr_activity = np.zeros(N, dtype=np.float32)
        for day_type in range(2):
            for h in range(24):
                # Max possible demand over any 2 consecutive hours
                two_hr_demand = np.abs(self._activity_profile[day_type, h]) + \
                                np.abs(self._activity_profile[day_type, (h+1)%24])
                max_abs_2hr_activity = np.maximum(max_abs_2hr_activity, two_hr_demand)
                
        self._lambda_max_system = float(np.sum(max_abs_2hr_activity))
        
        # ── 3. Travel Time Matrix (N x N) ──────────────────────────────────────
        self._travel_time_matrix = np.zeros((N, N), dtype=np.float32)
        for i, s_from in enumerate(stations):
            for j, s_to in enumerate(stations):
                self._travel_time_matrix[i, j] = state.get_travel_time(s_from.id, s_to.id)

        # ── 4. G_max (Maximum Theoretical Gravity) ─────────────────────────────
        best_gravity = 0.0
        for j in range(N):
            # Calculate gravity using the absolute maximum 2-hour demand
            current_gravity = np.sum(max_abs_2hr_activity / (self._travel_time_matrix[j] + 1.0))
            if current_gravity > best_gravity:
                best_gravity = current_gravity
        
        self._max_gravity = float(best_gravity)

        # Closest-depot ID
        vehicles = state.get_vehicles()
        self._depot_id = state.get_closest_depot(vehicles[0]) if vehicles else None

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
            k = self._sid_to_idx[sid]
            
            # Extract true physical inventory directly from the station objects
            station_bikes = station.bikes if isinstance(station.bikes, dict) else {}
            for bike in station_bikes.values():
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
        next_station_id: str = None,
    ) -> np.ndarray:
        """
        Compute φ(S^x) for the post-decision state.
        """
        func_post = func.copy()
        onsite_post = onsite.copy()

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
        func_cargo_veh = vehicle_status.functional_cargo + delta_func
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
        
        # ── Rolling 2-Hour Demand Anticipation ─────────────────────────────
        # 1. Determine time indices
        d = state.day() % 7
        day_type = 1 if d in [5, 6] else 0  # 1 if Weekend, 0 if Weekday
        
        current_minute = int(state.time % 60)
        h0 = int((state.time // 60) % 24)
        h1 = (h0 + 1) % 24
        h2 = (h0 + 2) % 24
        
        # 2. Calculate rolling weights for a 120-minute horizon
        weight_h0 = (60 - current_minute) / 60.0  # Remaining fraction of current hour
        weight_h1 = 1.0                           # All of the next hour
        weight_h2 = current_minute / 60.0         # Overlap into the third hour
        
        # 3. Extract the dynamic anticipated net demand array (N,)
        dynamic_activity = (
            self._activity_profile[day_type, h0] * weight_h0 +
            self._activity_profile[day_type, h1] * weight_h1 +
            self._activity_profile[day_type, h2] * weight_h2
        )

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
            capacities=self._capacities,             # New scaler array
            activity=dynamic_activity.astype(np.float64),
            dist_to_stations=dist_to_stations,       # New spatial array
            func_cargo_veh=float(func_cargo_veh),
            depot_cargo_veh=float(depot_cargo_veh),
            vehicle_capacity=K,
            dist_to_depot=dist_to_depot,
            lambda_max_system=self._lambda_max_system, # New Lambda_max scaler
            max_gravity=self._max_gravity,                 # New G_max scaler
            #maintenance_enabled=self.maintenance_enabled, # This should not be true un we are doing ablation study
            maintenance_enabled=True,
            #shift_timing_enabled=self.shift_timing_enabled, # This should not be true unless we are doing ablation study
            shift_timing_enabled=True,
            time_remaining=self._get_time_remaining(state, vehicle),
            shift_length=self._get_shift_length(state, vehicle)
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
        
        return None
    
    def _get_shift_length(self, state, vehicle) -> float:
        """
        Get reference shift length for normalization (minutes).
        
        Default: 1440 minutes (24 hours).
        Can be customized per vehicle if needed.
        """
        # Default 24-hour shift
        return 1440.0

    # ─────────────────────────────────────────────────────────────────────────
    # Reward signal
    # ─────────────────────────────────────────────────────────────────────────

    def _get_reward(self, state) -> float:
        """
        Compute the reward accumulated since the previous decision epoch.

        Reward = –(c_s · Δstarvations + c_c · Δcongestions)

        Negative because starvations and congestions are costs to minimise.
        Updates the internal counters for the next call.
        """
        C_STARV = 1.0   # penalty per starvation event
        C_CONG  = 0.5   # penalty per long-congestion event

        cur_s = state.metrics.get_aggregate_value("starvations")      or 0
        cur_c = state.metrics.get_aggregate_value("long congestions") or 0

        reward = -(
            C_STARV * (cur_s - self._prev_starvations) +
            C_CONG  * (cur_c - self._prev_congestions)
        )

        self._prev_starvations = cur_s
        self._prev_congestions = cur_c
        print(f"[REWARD] Starvations: {cur_s} (Δ={cur_s - self._prev_starvations}) | "
              f"Congestions: {cur_c} (Δ={cur_c - self._prev_congestions})")
        return reward

    # ─────────────────────────────────────────────────────────────────────────
    # TD(0) update
    # ─────────────────────────────────────────────────────────────────────────

    def td_update(self, reward: float, phi_next: np.ndarray) -> None:
        """
        TD(0) semi-gradient update for linear VFA:

            θ ← θ + α ( r + γ V(S^x_next) − V(S^x_cur) ) φ(S^x_cur)

        where V(S^x) = θᵀφ(S^x) and φ(S^x_cur) is cached from the
        previous call to get_best_action().

        No-op if there is no previous post-decision state stored yet.
        """
        if self._prev_phi is None:
            return 0.0

        # Compute pre-update values for logging
        v_cur = self.value(self._prev_phi)
        v_next = self.value(phi_next)
        td_error = reward + self.gamma * v_next - v_cur

        # --- SMART LOGGING ---
        # Only print if a physical penalty occurred OR if the VFA was highly surprised
        if reward < -0.01 or abs(td_error) > 1.0:
            target_value = reward + self.gamma * v_next
            print(f"    [TD Alert] Reward: {reward:6.3f} | V(S): {v_cur:6.3f} | Target: {target_value:6.3f} | TD Err: {td_error:6.3f}")

        self.theta   += self.alpha * td_error * self._prev_phi
        self.weights  = list(self.theta)   # keep the logging attribute in sync

        return td_error
    # ─────────────────────────────────────────────────────────────────────────
    # Action generation  (action-space splitting)
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_candidates(self, state, vehicle) -> List[sim.Action]:
        #NOTE: If if your first few training runs prove that the agent is getting stuck, update candidates to for example 5 nearest stations and 3 critical stations or something
        #NOTE: Currently uses a tabu list generation for multi-vehicle coordination. Can consider adding other vehcile decisions and effective inventory to mdp state if we want a more mathematically profound coordination mechanism, but this is a simple and effective first step to prevent multiple vehicles from being dispatched to the same starving/congested station.
        """
        Generate a tractable set of candidate actions using action-space splitting:

          Micro (inventory) - push current station toward its target state.
                              This is fixed greedily; only the routing varies.
          Macro (routing)   - enumerate the N_CANDIDATES nearest next stations
                              sorted by travel time from the current location.

        This version generates multiple candidates for different numbers of onsite repairs (0 to all).
        
        NEW: Multi-vehicle coordination via tabu list. Stations claimed by en-route vehicles
        are excluded from the candidate pool to prevent multiple vehicles from being 
        dispatched to the same starving/congested location.
        """
        load_from_queue = 0  # default – overridden when at depot

        if vehicle.is_at_depot():
            rebalancing = 0
            depot_removals = 0
            # Pick up all repaired bikes that fit in free capacity
            n_vehicle = len(vehicle.get_bike_inventory())
            vehicle_capacity = int(
                getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", n_vehicle))
            )
            free_cap = max(0, vehicle_capacity - n_vehicle)
            repaired_available = len(getattr(vehicle.location, "fixed_queue", {}))
            load_from_queue = min(repaired_available, free_cap)
            onsite_repairs_options = [0]  # No onsite repairs at depot
        else:
            target = round(vehicle.location.get_target_state(state.day(), state.hour()))
            # Only count functional bikes as valid inventory for regular rebalancing
            functional_bikes = [b for b in vehicle.location.get_bikes() if getattr(b, 'is_available', True)]
            n_station = len(functional_bikes)
            
            n_vehicle = len(vehicle.get_bike_inventory())
            vehicle_capacity = int(
                getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", n_vehicle))
            )
            free_cap = max(0, vehicle_capacity - n_vehicle)
            #########
            inv = vehicle.get_bike_inventory()
            n_vehicle_total = len(inv)
            
            # Count only functional bikes (not depot or onsite)
            n_vehicle_func = sum(1 for b in inv if getattr(b, 'damage_status', None) not in ['depot', 'onsite'])
            
            vehicle_capacity = int(getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", n_vehicle_total)))
            
            # Free capacity MUST use total bikes, because broken bikes take up physical space!
            free_cap = max(0, vehicle_capacity - n_vehicle_total)
            #########

            # --- Greedily pick up any depot-damaged bikes ---
            depot_removals = 0
            onsite_bikes = []
            if self.maintenance_enabled:
                broken_bikes = [b for b in vehicle.location.bikes.values() if getattr(b, 'damage_status', None) == 'depot']
                depot_removals = min(len(broken_bikes), free_cap)
                onsite_bikes = [b for b in vehicle.location.bikes.values() if getattr(b, 'damage_status', None) == 'onsite']
            num_onsite = len(onsite_bikes)
            onsite_repairs_options = list(range(0, num_onsite + 1))  # 0 to all

            # Normal Rebalancing Logic 
            delta = target - n_station       # >0 → deliver,  <0 → pickup
            if delta > 0:
                # NEW: Cap deliveries at the number of functional bikes we actually have!
                rebalancing = min(n_vehicle_func, delta)
            elif delta < 0:
                n = min(n_station, -delta, max(free_cap - depot_removals, 0))
                rebalancing = -n
            else:
                rebalancing = 0
        
        #TODO: Handle maintenance actions here as well when we add maintenance features and train the VFA with maintenance-enabled.

        # ── TABU LIST: Identify stations already claimed by other en-route vehicles ────
        # Uses MDP formulation notation: destination_station from VehicleStatus
        claimed_stations = set()
        for v in state.get_vehicles():
            if v.id != vehicle.id:
                # Check if vehicle is en-route (eta > 0 means traveling, not idle at a location)
                # This semantics matches sim.Vehicle.eta behavior in the simulator
                v_eta = getattr(v, 'eta', 0)
                if v_eta > state.time:
                    # Extract destination using MDP-canonical notation
                    # Fallback chain: destination_station (MDP) → next_location (sim.Action) → current location
                    dest = (
                        getattr(v, 'destination_station', None) or
                        getattr(v, 'next_location', None) or
                        (v.location.id if v.location else None)
                    )
                    if dest:
                        claimed_stations.add(dest)
        
        if self.log_rl_decisions:
            print(f"[TABU] Vehicle {vehicle.id} | Claimed stations: {claimed_stations}")
            
        # ── Macro: nearest N_CANDIDATES next stations (by travel time) ────
        cur_id = vehicle.location.id
        
        # Build candidate pool, excluding currently claimed stations
        pool = [s for s in state.get_stations() 
                if s.id != cur_id and s.id not in claimed_stations]
        
        depot_stations = state.get_depots()
        if depot_stations:
            depot = depot_stations[0]  # use first/closest depot
            # Always include depot (not subject to tabu, as repairs are handled outside the rebalancing network)
            if depot.id != cur_id:
                pool.append(depot)
        
        # --- Myopic Blindspot (5 Nearest + 3 Most Critical) ---
        # 1. Sort by travel time to find the nearest
        pool.sort(key=lambda s: state.get_travel_time(cur_id, s.id))
        nearest_stations = pool[:5] # Take the 5 closest
        
        # 2. Find the 3 most critical stations (highest deviation from target)
        remaining_pool = pool[5:]
        remaining_pool.sort(key=lambda s: abs(s.get_target_state(state.day(), state.hour()) - len(s.get_bikes())), reverse=True)
        critical_stations = remaining_pool[:3]
        
        # 3. Combine them into our final candidate list
        pool = nearest_stations + critical_stations

        # Fallback: if tabu filtering removed all stations, allow any non-claimed station
        if not pool:
            pool = [s for s in state.get_stations() if s.id != cur_id]
            depot_stations = state.get_depots()
            if depot_stations:
                depot = depot_stations[0]
                if depot.id != cur_id:
                    pool.append(depot)
            pool.sort(key=lambda s: state.get_travel_time(cur_id, s.id))
            pool = pool[: self.N_CANDIDATES]

        candidates = []
        for s in pool:
            for onsite_repairs in onsite_repairs_options:
                # --- NEW: Dynamic Rebalancing Math ---
                # Adjust pickups/drop-offs based on the repairs we are doing right now!
                if vehicle.is_at_depot():
                    current_rebalancing = 0
                else:
                    new_n_station = n_station + onsite_repairs
                    delta = target - new_n_station
                    
                    if delta > 0:
                        current_rebalancing = min(n_vehicle_func, delta)
                    elif delta < 0:
                        n = min(new_n_station, -delta, max(free_cap - depot_removals, 0))
                        current_rebalancing = -n
                    else:
                        current_rebalancing = 0
                # -------------------------------------

                mdp_action = MdpAction(
                    current_station=cur_id,
                    rebalancing=int(current_rebalancing),
                    onsite_repairs=int(onsite_repairs),
                    depot_removals=int(depot_removals),
                    load_from_queue=int(load_from_queue),
                    next_station=s.id,
                )
                candidates.append(mdp_action_to_sim_action(mdp_action, state, vehicle))

        if not candidates:
            depot_id   = state.get_closest_depot(vehicle)
            for onsite_repairs in onsite_repairs_options:

                # --- Fallback Dynamic Rebalancing ---
                if vehicle.is_at_depot():
                    current_rebalancing = 0
                else:
                    new_n_station = n_station + onsite_repairs
                    delta = target - new_n_station
                    if delta > 0:
                        current_rebalancing = min(n_vehicle_func, delta)
                    elif delta < 0:
                        n = min(new_n_station, -delta, max(free_cap - depot_removals, 0))
                        current_rebalancing = -n
                    else:
                        current_rebalancing = 0

                mdp_action = MdpAction(
                    current_station=cur_id,
                    rebalancing=int(current_rebalancing),
                    onsite_repairs=int(onsite_repairs),
                    depot_removals=int(depot_removals),
                    load_from_queue=int(load_from_queue),
                    next_station=depot_id,
                )
                candidates.append(mdp_action_to_sim_action(mdp_action, state, vehicle))
             
        # --- DEBUG 3: MYOPIC BLINDSPOT ---
        if not getattr(self, "_has_printed_blindspot", False):
            cand_ids = [getattr(c, "next_location", getattr(c, "next_station", None)) for c in candidates]
            
            # Find the actual most critical station in the whole city
            all_stats = [s for s in state.get_stations() if s.id != vehicle.location.id]
            all_stats.sort(key=lambda s: abs(s.get_target_state(state.day(), state.hour()) - len(s.get_bikes())), reverse=True)
            most_critical = all_stats[0].id if all_stats else "None"
            
            print(f"\n[DEBUG - BLINDSPOT] Vehicle at {vehicle.location.id}")
            print(f"  -> Candidate options: {cand_ids}")
            print(f"  -> Most critical station in network: {most_critical}")
            if most_critical not in cand_ids:
                print(f"  ->  The most critical station is NOT in the candidate list!")
            self._has_printed_blindspot = True

        # --- DEBUG 3: MYOPIC BLINDSPOT ---
        if not getattr(self, "_has_printed_blindspot", False):
            cand_ids = [getattr(c, "next_location", getattr(c, "next_station", None)) for c in candidates]
            
            # Find the actual most critical station in the whole city
            all_stats = [s for s in state.get_stations() if s.id != vehicle.location.id]
            all_stats.sort(key=lambda s: abs(s.get_target_state(state.day(), state.hour()) - len(s.get_bikes())), reverse=True)
            most_critical = all_stats[0].id if all_stats else "None"
            
            print(f"\n[DEBUG - BLINDSPOT] Vehicle at {vehicle.location.id}")
            print(f"  -> Candidate options: {cand_ids}")
            print(f"  -> Most critical station in network: {most_critical}")
            if most_critical not in cand_ids:
                print(f"  -> The most critical station is NOT in the candidate list!")
            self._has_printed_blindspot = True

        # --- DEBUG 3: MYOPIC BLINDSPOT ---
        if not getattr(self, "_has_printed_blindspot", False):
            cand_ids = [getattr(c, "next_location", getattr(c, "next_station", None)) for c in candidates]
            
            # Find the actual most critical station in the whole city
            all_stats = [s for s in state.get_stations() if s.id != vehicle.location.id]
            all_stats.sort(key=lambda s: abs(s.get_target_state(state.day(), state.hour()) - len(s.get_bikes())), reverse=True)
            most_critical = all_stats[0].id if all_stats else "None"
            
            print(f"\n[DEBUG - BLINDSPOT] Vehicle at {vehicle.location.id}")
            print(f"  -> Candidate options: {cand_ids}")
            print(f"  -> Most critical station in network: {most_critical}")
            if most_critical not in cand_ids:
                print(f"  -> The most critical station is NOT in the candidate list!")
            self._has_printed_blindspot = True

        return candidates

    # ─────────────────────────────────────────────────────────────────────────
    # Boltzmann (softmax) selection
    # ─────────────────────────────────────────────────────────────────────────

    def _boltzmann_select(
        self,
        actions: List[sim.Action],
        values: np.ndarray,
    ) -> Tuple[sim.Action, int]:
        """
        Select an action via the Boltzmann distribution.

            P(a_i) ∝ exp( −V(S^x_i) / τ )

        Lower V is better, so we negate before the softmax.
        Subtracts the max for numerical stability.

        Returns (selected_action, selected_index).
        """
        neg_v  = -values
        neg_v -= neg_v.max()                              # numerical stability
        exp_v  = np.exp(neg_v / max(self.tau, 1e-8))
        probs  = exp_v / exp_v.sum()

        # NEW: We want to favor higher values, not lower ones!
        v = values - values.max()  # shift for numerical stability (all <= 0)
        exp_v = np.exp(v / max(self.tau, 1e-8))
        probs = exp_v / exp_v.sum()

        idx = int(self._rng.choice(len(actions), p=probs))
        return actions[idx], idx

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
        """
        Called by VehicleArrival event at each decision epoch.

        Decision flow
        ─────────────
        1. Lazy-initialise caches from sim.State (first call only).
        2. Generate N_CANDIDATES candidate actions.
        3. Compute φ(S^x_a) and V(S^x_a) = θᵀφ for each candidate.
        4. (If learning) run TD(0) update using reward since last decision
           and the greedy next post-decision state as the bootstrap target.
        5. Select action via Boltzmann (learning) or greedy (exploitation).
        6. Cache the selected action's post-decision features for step 4
           of the next call.
        """
        # ── Step 1: lazy init ─────────────────────────────────────────────
        if not self._initialized:
            self._lazy_init(state)

        # ── Step 2: generate candidate actions ────────────────────────────
        candidates = self._generate_candidates(state, vehicle)

        base_func, base_onsite, base_depot = self._extract_inventories(state, vehicle)

        """# ── NEW: Pre-compute 3-hour expectations for feature extraction ───
        curr_d = state.day() % 7
        curr_h = state.hour() % 24
        net_flow_3hr = np.zeros(len(self._station_ids))
        expected_rent_3hr = np.zeros(len(self._station_ids))
        expected_return_3hr = np.zeros(len(self._station_ids))
        
        for i, s_id in enumerate(self._station_ids):
            station = state.stations[s_id]
            flow, rent, ret = 0.0, 0.0, 0.0
            for offset in range(3):
                h = (curr_h + offset) % 24
                d = (curr_d + (curr_h + offset) // 24) % 7
                arr = station.arrive_intensities[d][h] if getattr(station, 'arrive_intensities', None) else 0
                lev = station.leave_intensities[d][h] if getattr(station, 'leave_intensities', None) else 0
                flow += (arr - lev)
                rent += lev
                ret += arr
            net_flow_3hr[i] = flow
            expected_rent_3hr[i] = rent
            expected_return_3hr[i] = ret
        # ──────────────────────────────────────────────────────────────────"""


        '''# ── Step 3: score each candidate ──────────────────────────────────
        phis: List[np.ndarray] = []
        values = np.empty(len(candidates), dtype=np.float64)
        
        for k, action in enumerate(candidates):
            # Differentiate functional vs depot pickups
            # vehicle.location.bikes is already a dictionary of {bike_id: bike}, so we use it directly:
            station_bikes = vehicle.location.bikes if isinstance(vehicle.location.bikes, dict) else {}
            functional_pickups = 0
            depot_pickups = 0
            
            for b_id in action.pick_ups:
                b = station_bikes.get(b_id)
                if b and getattr(b, 'damage_status', None) == 'depot':
                    depot_pickups += 1
                elif b:
                    functional_pickups += 1
            
            # Net change in functional bikes and vehicle depot cargo
            delta_func = len(action.delivery_bikes) - functional_pickups
            delta_depot_cargo = depot_pickups'''
        
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

            #phi = self.extract_features(state, vehicle, delta_func, delta_depot_cargo)
            phi = self.extract_features(
                state, vehicle, 
                base_func, base_onsite, base_depot, 
                delta_func, delta_depot_cargo,
                delta_onsite_repairs,
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

        # --- DEBUG 2: SPATIAL PARALYSIS ---
        if "proximity_to_demand_gravity" in self.FEATURE_NAMES:
            idx = self.FEATURE_NAMES.index("proximity_to_demand_gravity")
            print(f"\n[DEBUG - SPATIAL] Gravity values for 8 candidates from {vehicle.location.id}:")
            for k, action in enumerate(candidates):
                dest = getattr(action, "next_location", getattr(action, "next_station", None))
                print(f"  -> Going to {dest} | Gravity Feature: {phis[k][idx]:.6f}")
            

        # --- DEBUG 2: SPATIAL PARALYSIS ---
        if "proximity_to_demand_gravity" in self.FEATURE_NAMES:
            idx = self.FEATURE_NAMES.index("proximity_to_demand_gravity")
            print(f"\n[DEBUG - SPATIAL] Gravity values for 8 candidates from {vehicle.location.id}:")
            for k, action in enumerate(candidates):
                dest = getattr(action, "next_location", getattr(action, "next_station", None))
                print(f"  -> Going to {dest} | Gravity Feature: {phis[k][idx]:.6f}")
            

        # ── Step 4: TD(0) update ──────────────────────────────────────────
        """# Bootstrap with the greedy (min-value) next post-decision state,
        # consistent with the off-policy evaluation target.
        if self.learning_mode and self._prev_phi is not None:
            reward   = self._get_reward(state)
            phi_next = phis[int(np.argmin(values))]
            self.td_update(reward, phi_next)
        elif self.learning_mode and self._prev_phi is None:
            # First VFA call in the learning phase (warm-up just ended).
            # Sync metric baseline so that costs accumulated during warm-up
            # are NOT counted as part of the first reward signal.
            self._prev_starvations = state.metrics.get_aggregate_value("starvation")      or 0
            self._prev_congestions = state.metrics.get_aggregate_value("long_congestion") or 0"""
        """# ── Step 4: TD(0) update ──────────────────────────────────────────
        td_err = 0.0  # Initialize it here safely!
        
        if self.learning_mode and self._prev_phi is not None:
            reward   = self._get_reward(state)
            phi_next = phis[int(np.argmin(values))]
            # Capture the returned td_error from the function we modified earlier
            td_err = self.td_update(reward, phi_next) 
            
        elif self.learning_mode and self._prev_phi is None:
            # First VFA call in the learning phase...
            self._prev_starvations = state.metrics.get_aggregate_value("starvation")      or 0
            self._prev_congestions = state.metrics.get_aggregate_value("long_congestion") or 0"""
        
        """# ── Step 4: TD(0) update ──────────────────────────────────────────
        td_err = 0.0
        if self.learning_mode and self._prev_phi is not None:
            # The policy simply consumes the reward signal!
            reward = self.reward_calc.compute_step_reward(state.metrics)
            
            phi_next = phis[int(np.argmin(values))]
            td_err = self.td_update(reward, phi_next)"""
        
        """# ── Step 4: TD(0) update ──────────────────────────────────────────
        td_err = 0.0
        if self.learning_mode and self._prev_phi is not None:
            # Consume the reward signal safely
            reward = self.reward_calc.compute_step_reward(state.metrics)
            phi_next = phis[int(np.argmin(values))]
            td_err = self.td_update(reward, phi_next)
            
        elif self.learning_mode and self._prev_phi is None:
            # Now we must silently call compute_step_reward() once to sync the as we are no longer in warm-up, 
            # but we don't want to use this reward for the TD update (since it includes the warm-up costs). 
            _ = self.reward_calc.compute_step_reward(state.metrics)"""
        
        # ── Step 4: TD(0) update ──────────────────────────────────────────
        td_err = 0.0
        if self.learning_mode and self._prev_phi is not None:
            # Consume the reward signal safely
            reward = self.reward_calc.compute_step_reward(state.metrics)
            
            # FIXED: We expect to take the BEST action next, so use argmax!
            phi_next = phis[int(np.argmax(values))]
            td_err = self.td_update(reward, phi_next)
            
        elif self.learning_mode and self._prev_phi is None:
            _ = self.reward_calc.compute_step_reward(state.metrics)
            # Freeze the exact warmup totals here!
            self.warmup_starvations_snapshot = self.reward_calc._prev_starvations
            self.warmup_congestions_snapshot = self.reward_calc._prev_congestions
            self.warmup_trips_snapshot = self.reward_calc._prev_trips

        """ # ── Step 5: select action ─────────────────────────────────────────
        if self.learning_mode:
            selected, sel_idx = self._boltzmann_select(candidates, values)
        else:
            sel_idx  = int(np.argmin(values))
            selected = candidates[sel_idx]"""

        # ── Step 5: select action ─────────────────────────────────────────
        if self.learning_mode:
            selected, sel_idx = self._boltzmann_select(candidates, values)
        else:
            # FIXED: Pick the action with the highest expected reward!
            sel_idx  = int(np.argmax(values))
            selected = candidates[sel_idx]

        # ── Step 6: Log depot decisions (optional) ────────────────────────
        self._log_depot_decision(state, vehicle, selected, phis[sel_idx], values[sel_idx])

        # ── Step 7: cache post-decision features for next TD update ───────
        self._prev_phi = phis[sel_idx]

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
    # Temperature control  (called by training loop, once per episode)
    # ─────────────────────────────────────────────────────────────────────────

    def set_temperature(self, tau: float) -> None:
        """Set Boltzmann temperature τ for the next episode."""
        self.tau = max(tau, 1e-8)

    # ─────────────────────────────────────────────────────────────────────────
    # Episode boundary reset  (called by EpisodeTrainingPolicy.__init__)
    # ─────────────────────────────────────────────────────────────────────────

    '''def reset_episode(self) -> None:
        
        Clear per-episode TD tracking state.

        Does NOT reset θ or the station caches — those persist across episodes.
        
        self._prev_phi         = None
        self._prev_starvations = 0
        self._prev_congestions = 0
        
        # RL decision making logging
        self.log_rl_decisions = True  
        self.rl_logs = []'''
    
    def reset_episode(self) -> None:
        self._prev_phi = None
        # Delegate reset to the calculator
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

      - GreedyPolicy        during the warm-up phase  (no TD updates)
      - LinearVFAPolicy     during the learning phase (Boltzmann + TD(0))

    The phase transition is time-based: once state.time >= warmup_end_time
    (an absolute simulation-minutes value), the VFA policy takes control.

    A new instance of this class is created for every episode, but the
    underlying LinearVFAPolicy (and its θ) is shared and persists.
    """

    def __init__(
        self,
        vfa_policy:     LinearVFAPolicy,
        greedy_policy,
        warmup_end_time: float,
    ) -> None:
        super().__init__(maintenance_enabled=True)

        self.vfa_policy      = vfa_policy
        self.greedy_policy   = greedy_policy
        self.warmup_end_time = warmup_end_time

        # Mirror the VFA weights for run_simulation.py logging
        self.weights = vfa_policy.weights

        # Reset per-episode TD state on the shared VFA policy
        vfa_policy.reset_episode()

    def get_best_action(self, state, vehicle) -> sim.Action:
        if state.time < self.warmup_end_time:
            # Warm-up: purely greedy, θ left unchanged
            return self.greedy_policy.get_best_action(state, vehicle)
        else:
            # Learning: VFA + Boltzmann exploration + TD(0) update
            return self.vfa_policy.get_best_action(state, vehicle)

    def __repr__(self) -> str:
        return f"EpisodeTrainingPolicy(tau={self.vfa_policy.tau:.4f})"
