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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

if TYPE_CHECKING:
    from policies.sjovik_sund.simulation_logging import SimulationRunLogger as RunLogger
from collections import deque

from policies import action
from policies.sjovik_sund.mdp.reward import RewardCalculator
from policies.sjovik_sund.mdp.candidate_generator import generate_candidates
from helpers import format_sim_time


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
from policies.sjovik_sund.mdp.action_bridge import (
    mdp_action_to_sim_action,
    record_depot_action_stats,
)
from settings import ENABLE_COMPONENT_FAILURES, SERVICE_TIME_TO
from sim.Bike import Bike
from sim.bike_degradation_modeling.bike_component_degradation_model import ComponentFailureModel

# ── Debug flags — flip individual checks on/off from train_vfa.py ─────────────
VFA_DEBUG_FLAGS = {
    "check2_td_updates":       True,   # first 5 TD updates per episode
    "check3_mean_vs_sum":      True,   # mean vs sum gradient verification
    "check4_traces":           True,   # eligibility trace norms per episode
    "check5_rewards":          True,   # reward stats per episode
    "check6_targets":          True,   # target / prediction / td_error stats
    "check7_feature_scale":    False,  # per-feature scale (heavy — off by default)
    "check8_update_direction": False,  # per-feature update direction (heavy — off by default)
    "check10_warmup":          True,   # warmup-end state distribution
    "check11_candidates":      True,   # candidate type logging (first 20 decisions/ep)
    "every_n_episodes":        10,     # heavy checks run every N batch updates
}

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
        gamma: float = 0.97,
        epsilon: float = 0.0,
        learning_mode: bool = True,
        config: Optional[MDPConfig] = None,
        seed: int = 42,
        maintenance_enabled: bool = ENABLE_COMPONENT_FAILURES,
        logistics_enabled: bool = False,
        demand_horizon_enabled: bool = True,
        log_depot_visits: bool = False,
        depot_log_file: Optional[str] = None,
        reward_calculator: Optional[RewardCalculator] = None,
        use_bias_feature: bool = False,
        use_terminal_update: bool = False,
        use_batch_td_clip: bool = False,
        batch_td_clip_value: float = 10.0,
        use_online_td_updates: bool = False,
        transition_update_interval: int = 0,
        initial_bias: float | None = None,
        use_feature_centering: bool = False,
        feature_centering_beta: float = 0.01,
    ) -> None:
        """
        Initializes the policy, sets hyperparameters (alpha, gamma, epsilon),
        and creates the weight vector (theta) which represents the
        agent's learned knowledge.
        """

        print(f"Initializing LinearVFAPolicy with maintenance_enabled={maintenance_enabled}, ")

        super().__init__(maintenance_enabled=maintenance_enabled)

        self.maintenance_enabled = maintenance_enabled
        self.logistics_enabled = logistics_enabled
        self.demand_horizon_enabled = demand_horizon_enabled
        self.log_depot_visits = log_depot_visits
        self.depot_log_file = depot_log_file
        self.use_bias_feature = use_bias_feature or bool(active_features and "bias" in active_features)
        self.use_terminal_update = use_terminal_update
        self.use_batch_td_clip = use_batch_td_clip
        self.batch_td_clip_value = float(batch_td_clip_value)
        self.use_online_td_updates = bool(use_online_td_updates)
        self.transition_update_interval = max(0, int(transition_update_interval))
        if self.use_online_td_updates and self.transition_update_interval > 0:
            raise ValueError("use_online_td_updates and transition_update_interval are mutually exclusive")
        # 1. Get the canonical list of ALL possible features

        self.ALL_FEATURE_NAMES = _get_feature_names(True, True, True, True, include_bias=self.use_bias_feature)

        # 2. Determine which features we are actually using
        if active_features is None:
            self.FEATURE_NAMES = _get_feature_names(
                self.maintenance_enabled,
                self.logistics_enabled,
                self.demand_horizon_enabled,
                include_bias=self.use_bias_feature,
            )
        else:
            # --- FIXED: Force the user's active features into canonical mathematical order ---
            requested_features = set(active_features)
            if self.use_bias_feature:
                requested_features.add("bias")
            unknown_features = requested_features - set(self.ALL_FEATURE_NAMES)
            if unknown_features:
                raise ValueError(
                    f"Unknown active feature(s): {sorted(unknown_features)}. "
                    f"Valid features include: {self.ALL_FEATURE_NAMES}"
                )
            self.FEATURE_NAMES = [name for name in self.ALL_FEATURE_NAMES if name in requested_features]

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
        _health_names = {"fleet_failure_risk", "fleet_health_deficit", "fleet_low_health_fraction",
                         "depot_bound_health_deficit", "onsite_health_deficit", "maintenance_restoration_value"}
        self._use_health_features = bool(_health_names & set(self.FEATURE_NAMES))
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
        self._depot_repair_discount = gamma ** 24.0  # 1440 min depot repair at gamma-per-hour
        self.epsilon       = epsilon
        self.learning_mode = learning_mode
        default_config = MDPConfig.full_maintenance() if maintenance_enabled else MDPConfig.no_maintenance()
        self.config        = config or default_config  # defaults to full maintenance
        self._rng          = np.random.default_rng(seed)

        # ── Parameter vector θ (small random initialisation) ─────────────────
        self.theta: np.ndarray = np.zeros(n_features, dtype=np.float64)
        self.initial_bias = initial_bias
        if initial_bias is not None and "bias" in self.FEATURE_NAMES:
            self.theta[self.FEATURE_NAMES.index("bias")] = float(initial_bias)
        elif initial_bias not in (None, 0.0):
            raise ValueError("initial_bias requires the bias feature to be enabled")

        self.use_feature_centering = bool(use_feature_centering)
        self.feature_centering_beta = float(np.clip(feature_centering_beta, 0.0, 1.0))
        self._feature_center = np.zeros(n_features, dtype=np.float64)
        self._feature_center_initialized = False
        self._bias_feature_idx = self.FEATURE_NAMES.index("bias") if "bias" in self.FEATURE_NAMES else None

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
        self._max_travel_time:     float                = 60.0
        self._depot_id:            Optional[str]        = None
        self._N_stations:          Optional[int]        = None
        self._total_fleet_size:    Optional[int]        = None

        # ── Per-episode TD tracking ───────────────────────────────────────────
        self._prev_phi:          Optional[np.ndarray] = None
        self._prev_time:         Optional[float]      = None

        # ── Potential-based reward shaping (off by default) ──────────────────
        self.use_reward_shaping:  bool = False
        self.shaping_feature_idx: int  = 0

        # ── TD(λ) with eligibility traces (online, replaces batch buffer) ─────
        self.use_td_lambda:       bool  = True
        self.td_lambda:           float = 0.8
        self._eligibility_trace: np.ndarray = np.zeros(n_features, dtype=np.float64)
        self._td_lambda_step_count: int = 0
        self._ep_update_count:   int  = 0   # Check 2: counts TD transitions per episode
        self._ep_decision_count: int  = 0   # Check 11: counts decisions per episode

        # Buffer for Synchronous Batch Learning
        self.batch_buffer: List[Tuple[np.ndarray, float, np.ndarray, float]] = []
        self._online_update_count: int = 0
        self._transition_batch_update_count: int = 0
        self._all_phis_for_corr: Optional[List[np.ndarray]] = None
        self._online_diag_phis: List[np.ndarray] = []
        self._online_diag_tderrs: List[float] = []
        self._online_diag_raw_tderrs: List[float] = []
        
        # Buffer for Experience Replay (Mini-Batch SGD)
        self.use_experience_replay = False
        self.replay_buffer = deque(maxlen=2000)
        self.mini_batch_size = 64

        # logging for RL decisions (e.g., depot visits)
        self.log_rl_decisions = True
        self.rl_logs: List[Dict[str, object]] = []
        self.log_candidate_diagnostics: bool = False
        self._candidate_diag_rows: List[Dict[str, object]] = []
        self.log_greedy_comparison: bool = False
        self._comparison_episode: int = -1
        self._comparison_rows: List[Dict[str, object]] = []
        self._pending_comparison_row: Optional[Dict[str, object]] = None
        self._comparison_greedy: Optional[Any] = None
        self._collect_phis_inside_batch_update: bool = False
        self._warmup_depot_snapshot: Dict[str, int] = {}
        self.warmup_starvations_snapshot: int = 0
        self.warmup_congestions_snapshot: int = 0

        # Optional RunLogger — attach externally for structured per-run output
        self.logger: Optional[RunLogger] = None
        
        # Cache for static fleet size to optimize VFA speed
        self.cached_fleet_size: Optional[float] = None

        # Simulator reference — set by init_sim(), used to pre-warm target matrix
        self._simulator_ref: Optional[Any] = None

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

        leave_profile = np.zeros((2, 24, N), dtype=np.float32)
        arrive_profile = np.zeros((2, 24, N), dtype=np.float32)

        for i, s in enumerate(stations):
            for h in range(24):
                leave_profile[0, h, i]  = np.mean([s.get_leave_intensity(d, h)  for d in weekday_days])
                leave_profile[1, h, i]  = np.mean([s.get_leave_intensity(d, h)  for d in weekend_days])
                arrive_profile[0, h, i] = np.mean([s.get_arrive_intensity(d, h) for d in weekday_days])
                arrive_profile[1, h, i] = np.mean([s.get_arrive_intensity(d, h) for d in weekend_days])

        # Find the absolute peak 3-hour rolling net demand for scaling (Lambda_max)
        max_abs_3hr_activity = np.zeros(N, dtype=np.float32)
        for day_type in range(2):
            for h in range(24):
                three_hr_demand = (
                    np.abs(leave_profile[day_type, h]        - arrive_profile[day_type, h]) +
                    np.abs(leave_profile[day_type, (h+1)%24] - arrive_profile[day_type, (h+1)%24]) +
                    np.abs(leave_profile[day_type, (h+2)%24] - arrive_profile[day_type, (h+2)%24])
                )
                max_abs_3hr_activity = np.maximum(max_abs_3hr_activity, three_hr_demand)
                
        self._leave_profile = leave_profile
        self._arrive_profile = arrive_profile
        self._lambda_max_system = float(np.sum(max_abs_3hr_activity))
        
        # ── 3. Travel Time Matrix (N x N) ──────────────────────────────────────
        travel_time_matrix = np.zeros((N, N), dtype=np.float32)
        for i, s_from in enumerate(stations):
            for j, s_to in enumerate(stations):
                travel_time_matrix[i, j] = state.get_travel_time(s_from.id, s_to.id)

        # ── 4. G_max (Maximum Theoretical Gravity) ─────────────────────────────
        best_gravity = 0.0
        for j in range(N):
            # Calculate gravity using the absolute maximum 3-hour demand
            current_gravity = np.sum(max_abs_3hr_activity / (travel_time_matrix[j] + 1.0))
            if current_gravity > best_gravity:
                best_gravity = current_gravity
        
        self._travel_time_matrix = travel_time_matrix
        self._max_gravity = float(best_gravity)

        # Closest-depot ID
        vehicles = state.get_vehicles()
        self._depot_id = state.get_closest_depot(vehicles[0]) if vehicles else None
        
        self.cached_fleet_size = float(len(state.get_all_bikes()))
        self._max_travel_time = float(np.max(travel_time_matrix)) if travel_time_matrix.size > 0 else 60.0

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

        if not hasattr(self, '_has_printed_vision'):
            self._has_printed_vision = True
            on_vehicles   = sum(len(v.get_bike_inventory()) for v in state.get_vehicles())
            in_transit    = len(state.bikes_in_use)
            at_stations   = sum(func) + sum(onsite) + sum(depot)
            depot_available = sum(len(d.bikes) for d in state.get_depots())
            depot_fixed_q   = sum(len(d.fixed_queue) for d in state.get_depots())
            depot_in_repair = sum(len(bikes) for d in state.get_depots() for _, bikes in d.in_repair)
            total_tracked   = at_stations + on_vehicles + in_transit + depot_available + depot_fixed_q + depot_in_repair
            fleet = self.cached_fleet_size or 768
            print(
                f"\n[DEBUG - VFA VISION] Stations: {sum(func)} func, {sum(onsite)} onsite, {sum(depot)} depot-bound"
                f" | Vehicles: {on_vehicles}"
                f" | In-transit (customers): {in_transit}"
                f" | Depot: {depot_available} available, {depot_fixed_q} fixed-queue, {depot_in_repair} in-repair"
                f" | Total: {total_tracked} / {fleet}"
            )
    
        return func, onsite, depot

    @staticmethod
    def _bike_id(bike) -> int:
        return getattr(bike, "bike_id", getattr(bike, "id", -1))

    _failure_exposure_trip_km: float = 2.5
    _low_failure_risk_threshold: float = 0.01

    @classmethod
    def _bike_health_and_risk(cls, bike) -> Tuple[float, float, float]:
        """
        Return (health, failure_risk, deficit) for one bike.

        health is the weakest component reliability, so 1.0 is fresh/healthy and
        0.0 is failed/high-risk. failure_risk is the probability that at least one
        component fails on a representative future trip, conditional on the
        current component odometers.
        """
        odometers = getattr(bike, "component_odometers", {}) or {}
        failures = getattr(bike, "component_failures", {}) or {}
        reliabilities = []
        trip_survival_probs = []

        for category, odometer in odometers.items():
            data = failures.get(category, {})
            scale = float(data.get("scale", 0.0) or 0.0)
            shape = float(data.get("shape", 0.0) or 0.0)
            if scale <= 0.0 or shape <= 0.0:
                continue
            odometer_km = max(float(odometer), 0.0)
            reliability = float(ComponentFailureModel.calculate_reliability(odometer_km, scale, shape))
            trip_failure = float(ComponentFailureModel.calculate_trip_failure_probability(
                odometer_km,
                cls._failure_exposure_trip_km,
                scale,
                shape,
            ))
            reliabilities.append(float(np.clip(reliability, 0.0, 1.0)))
            trip_survival_probs.append(float(np.clip(1.0 - trip_failure, 0.0, 1.0)))

        if not reliabilities:
            return 1.0, 0.0, 0.0

        rel = np.array(reliabilities, dtype=np.float64)
        trip_survival = np.array(trip_survival_probs, dtype=np.float64)
        health = float(np.min(rel))
        failure_risk = float(np.clip(1.0 - np.prod(trip_survival), 0.0, 1.0))
        return health, failure_risk, 1.0 - health

    @staticmethod
    def _is_functional_bike(bike) -> bool:
        return getattr(bike, "damage_status", None) is None

    def _all_bikes_including_depot_queues(self, state) -> list:
        """Return unique live bikes, including depot repair/fixed queues."""
        bikes_by_id = {}
        for bike in state.get_all_bikes():
            bikes_by_id[self._bike_id(bike)] = bike
        for depot in state.get_depots():
            for bike in getattr(depot, "fixed_queue", {}).values():
                bikes_by_id[self._bike_id(bike)] = bike
            for _, bikes in getattr(depot, "in_repair", []):
                for bike in bikes:
                    bikes_by_id[self._bike_id(bike)] = bike
        return list(bikes_by_id.values())

    def _compute_health_base(self, state, vehicle) -> dict:
        """Iterate all bikes once per state. Expensive — call once, not per candidate."""
        all_bikes = self._all_bikes_including_depot_queues(state)
        total = max(float(len(all_bikes)), 1.0)
        functional_total = 0.0
        health_sum = risk_sum = low_count = 0.0
        onsite_def_sum = depot_def_sum = 0.0
        by_id = {}
        health_cache = {}
        for bike in all_bikes:
            bid = self._bike_id(bike)
            by_id[bid] = bike
            health, risk, deficit = self._bike_health_and_risk(bike)
            health_cache[bid] = (health, risk, deficit)
            if self._is_functional_bike(bike):
                functional_total += 1.0
                health_sum += health
                risk_sum += risk
                low_count += 1.0 if risk > self._low_failure_risk_threshold else 0.0
            status = getattr(bike, "damage_status", None)
            if status == "onsite":
                onsite_def_sum += deficit
            elif status == "depot":
                depot_def_sum += deficit
        return {
            "total": total,
            "status_denom": max(total * 0.20, 1.0),
            "functional_total": max(functional_total, 1.0),
            "health_sum": health_sum,
            "risk_sum": risk_sum,
            "low_count": low_count,
            "onsite_def_sum": onsite_def_sum,
            "depot_def_sum": depot_def_sum,
            "by_id": by_id,
            "health_cache": health_cache,
        }

    def _health_features_from_base(self, base: dict, candidate_action, vehicle, vehicle_capacity: int = 1) -> dict:
        """Apply candidate-action delta to cached base. Cheap — call per candidate."""
        total        = base["total"]
        status_denom = base["status_denom"]
        functional_total = base["functional_total"]
        health_sum   = base["health_sum"]
        risk_sum     = base["risk_sum"]
        low_count    = base["low_count"]
        onsite_def_sum = base["onsite_def_sum"]
        depot_def_sum  = base["depot_def_sum"]
        by_id        = base["by_id"]
        health_cache = base["health_cache"]

        onsite_restored_def = onsite_restored_count = 0.0
        depot_dropoff_def = 0.0

        if candidate_action is not None:
            for bike_id in getattr(candidate_action, "onsite_repairs", []) or []:
                bike = by_id.get(bike_id)
                if bike is None:
                    continue
                health, risk, deficit = health_cache.get(bike_id, self._bike_health_and_risk(bike))
                onsite_restored_def += deficit
                onsite_restored_count += 1.0

            vehicle_bikes = {self._bike_id(b): b for b in vehicle.get_bike_inventory()}
            for bike_id in getattr(candidate_action, "delivery_bikes", []) or []:
                bike = vehicle_bikes.get(bike_id)
                if bike is not None and getattr(bike, "damage_status", None) == "depot":
                    _, _, deficit = self._bike_health_and_risk(bike)
                    depot_dropoff_def += deficit

        K = max(float(vehicle_capacity), 1.0)
        functional_after = max(functional_total + onsite_restored_count, 1.0)
        return {
            "fleet_failure_risk":         max(0.0, risk_sum) / functional_after,
            "fleet_health_deficit":       max(0.0, functional_total - health_sum) / functional_after,
            "fleet_low_health_fraction":  max(0.0, low_count) / functional_after,
            "depot_bound_health_deficit": depot_def_sum / status_denom,
            "onsite_health_deficit":      max(0.0, onsite_def_sum - onsite_restored_def) / status_denom,
            "maintenance_restoration_value": (onsite_restored_def + self._depot_repair_discount * depot_dropoff_def) / K,
        }

    def _health_feature_inputs(self, state, vehicle, candidate_action=None, vehicle_capacity: int = 1) -> dict:
        """Convenience wrapper — computes base + delta in one call (used outside the candidate loop)."""
        base = self._compute_health_base(state, vehicle)
        return self._health_features_from_base(base, candidate_action, vehicle, vehicle_capacity)

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
        delta_depot_fixed_queue: int = 0,
        delta_depot_in_repair: int = 0,
        time_remaining: Optional[float] = None,
        shift_length: float = 1440.0,
        next_station_id: Optional[str] = None,
        eval_time: Optional[float] = None,
        projected_time: Optional[float] = None,
        dist_to_next: float = 0.0,
        explicit_vehicle_loc_id: Optional[str] = None,
        explicit_functional_cargo: Optional[int] = None,
        explicit_depot_cargo: Optional[int] = None,
        explicit_capacity: Optional[int] = None,
        explicit_dist_to_depot: Optional[float] = None,
        explicit_dist_to_stations: Optional[np.ndarray] = None,
        candidate_action=None,
        health_base: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Compute φ(S^x) for the post-decision state.
        """
        t         = eval_time if eval_time is not None else (float(state.time) if state else 0.0)
        eval_day  = int(t // (24 * 60)) % 7
        eval_hour = int((t // 60) % 24)
        target_matrix = self._target_matrix
        leave_profile = self._leave_profile
        arrive_profile = self._arrive_profile
        travel_time_matrix = self._travel_time_matrix
        capacities = self._capacities
        lambda_max_system = self._lambda_max_system
        max_gravity = self._max_gravity
        n_stations = self._N_stations
        cached_fleet_size = self.cached_fleet_size
        assert target_matrix is not None
        assert leave_profile is not None
        assert arrive_profile is not None
        assert travel_time_matrix is not None
        assert capacities is not None
        assert lambda_max_system is not None
        assert max_gravity is not None
        assert n_stations is not None
        assert cached_fleet_size is not None

        # Use the projected arrival time for all time-indexed features so that
        # candidates with long travel times are evaluated at t_{k+1}, not t_k.
        _time = projected_time if projected_time is not None else state.time

        func_post = func.copy()
        onsite_post = onsite.copy()
        depot_post = depot.copy()

        #TODO: Fix so that the values on car is always integer
        cur_loc_id = explicit_vehicle_loc_id if explicit_vehicle_loc_id is not None else vehicle.location.id
        
        # ── Apply post-decision delta at the vehicle's current station ─────
        if cur_loc_id in self._sid_to_idx:
            cur_idx = self._sid_to_idx[cur_loc_id]
            func_post[cur_idx] = max(0, func_post[cur_idx] + delta_func)
            if delta_onsite_repairs:
                repairs = max(0, int(delta_onsite_repairs))
                applied_repairs = min(repairs, int(onsite_post[cur_idx]))
                onsite_post[cur_idx] = max(0, onsite_post[cur_idx] - applied_repairs)
                func_post[cur_idx] += applied_repairs
            if delta_depot_cargo >0:
                depot_post[cur_idx] = max(0, depot_post[cur_idx] - delta_depot_cargo)

        # ── Vehicle cargo in post-decision state ───────────────────────────
        if explicit_functional_cargo is not None and explicit_depot_cargo is not None and explicit_capacity is not None:
            func_cargo_veh = explicit_functional_cargo - delta_func
            depot_cargo_veh = explicit_depot_cargo + delta_depot_cargo
            K = max(int(explicit_capacity), 1)
        else:
            vehicle_status = extract_vehicle_status(vehicle, state.time, self.config)
            # Note: delta_func is (deliveries - pickups), so it represents the change to the STATION's inventory.
            # Therefore, we MUST SUBTRACT it to get the change to the VEHICLE's inventory.
            func_cargo_veh = vehicle_status.functional_cargo - delta_func
            depot_cargo_veh = vehicle_status.depot_cargo + delta_depot_cargo
            K = max(int(vehicle_status.capacity), 1)

        # ── Anticipate the inventory change at the DESTINATION ─────────────
        if next_station_id and next_station_id in self._sid_to_idx:
            nxt_idx = self._sid_to_idx[next_station_id]
            d, h = eval_day % 7, eval_hour % 24
            target_nxt = target_matrix[d, h, nxt_idx]
            cur_nxt = func_post[nxt_idx]
            delta_nxt = target_nxt - cur_nxt
            
            if delta_nxt > 0: # Starving
                free_docks_nxt = max(
                    0.0,
                    float(capacities[nxt_idx])
                    - float(func_post[nxt_idx])
                    - float(onsite_post[nxt_idx])
                    - float(depot_post[nxt_idx])
                )
                delivery = min(func_cargo_veh, delta_nxt, free_docks_nxt)
                func_post[nxt_idx] += delivery
                func_cargo_veh -= delivery
            elif delta_nxt < 0: # Congested
                free_cap = max(0, K - (func_cargo_veh + depot_cargo_veh))
                pickup = min(-delta_nxt, free_cap, cur_nxt)
                func_post[nxt_idx] -= pickup
                func_cargo_veh += pickup
        
        # ── Rolling 3-Hour Demand Anticipation ─────────────────────────────
        d = eval_day % 7
        day_type = 1 if d in [5, 6] else 0  # 0 = weekday, 1 = weekend

        current_minute = int(t % 60)
        hours   = [(eval_hour + k) % 24 for k in range(4)]
        weights = np.array([
            (60 - current_minute) / 60.0,  # remaining fraction of current hour
            1.0,                            # full next hour
            1.0,                            # full hour after that
            current_minute / 60.0,          # overlap into fourth hour
        ])

        dynamic_leave  = np.array([leave_profile[day_type, h]  * w for h, w in zip(hours, weights)])
        dynamic_arrive = np.array([arrive_profile[day_type, h] * w for h, w in zip(hours, weights)])

        # ── Time-indexed target inventory (N,) ─────────────────────────────
        d, h   = eval_day % 7, eval_hour % 24
        target = target_matrix[d, h]

        # ── Anticipated Distances (from Destination) ────────────────
        # Evaluate spatial gravity from where the vehicle is GOING, not where it is!
        routing_target = next_station_id if next_station_id else cur_loc_id
        
        if explicit_dist_to_depot is not None:
            dist_to_depot = explicit_dist_to_depot
        else:
            dist_to_depot = (
                state.get_travel_time(routing_target, self._depot_id)
                if state and self._depot_id and self._depot_id != routing_target
                else 0.0
            )
        
        if explicit_dist_to_stations is not None:
            dist_to_stations = explicit_dist_to_stations
        else:
            # Fast spatial slice from cached matrix using the routing target
            if routing_target in self._sid_to_idx:
                v_idx = self._sid_to_idx[routing_target]
                dist_to_stations = travel_time_matrix[v_idx]
            else:
                dist_to_stations = np.array([
                    state.get_travel_time(routing_target, sid) if state else 0.0
                    for sid in self._station_ids
                ], dtype=np.float32)

        # ── Depot queue state ──────────────────────────────────────────────
        # Depot inventory is not part of the station arrays, so depot actions
        # need explicit queue/pipeline deltas to match PostDecisionState.
        depot_in_repair = max(
            0.0,
            float(sum(len(bikes) for d in state.get_depots() for _, bikes in d.in_repair))
            + float(delta_depot_in_repair),
        )
        depot_fixed_queue = max(
            0.0,
            float(sum(len(d.fixed_queue) for d in state.get_depots()))
            + float(delta_depot_fixed_queue),
        )
        if not self._use_health_features:
            health_features = {"fleet_failure_risk": 0.0, "fleet_health_deficit": 0.0,
                               "fleet_low_health_fraction": 0.0, "depot_bound_health_deficit": 0.0,
                               "onsite_health_deficit": 0.0, "maintenance_restoration_value": 0.0}
        elif health_base is not None:
            health_features = self._health_features_from_base(health_base, candidate_action, vehicle, K)
        else:
            health_features = self._health_feature_inputs(state, vehicle, candidate_action=candidate_action, vehicle_capacity=K)

        # Current-station local features should use the same service location
        # that received the post-decision deltas above. In normal VFA scoring
        # this is vehicle.location.id; in rollout base steps it may be an
        # explicit synthetic destination.
        cur_station_idx = self._sid_to_idx.get(cur_loc_id, -1)

        # ── Delegate to canonical feature extractor ────────────────────────
        phi_full = _extract_phi(
            func=func_post.astype(np.float64),
            onsite=onsite_post.astype(np.float64),
            depot=depot_post.astype(np.float64),
            target=target.astype(np.float64),
            capacities=capacities,
            leave_activity=dynamic_leave.astype(np.float64),
            arrive_activity=dynamic_arrive.astype(np.float64),
            dist_to_stations=dist_to_stations,
            func_cargo_veh=float(func_cargo_veh),
            depot_cargo_veh=float(depot_cargo_veh),
            vehicle_capacity=K,
            dist_to_depot=dist_to_depot,
            lambda_max_system=lambda_max_system,
            max_gravity=max_gravity,
            fleet_size=cached_fleet_size,
            depot_in_repair=depot_in_repair,
            depot_fixed_queue=depot_fixed_queue,
            fleet_failure_risk=health_features["fleet_failure_risk"],
            fleet_health_deficit=health_features["fleet_health_deficit"],
            fleet_low_health_fraction=health_features["fleet_low_health_fraction"],
            depot_bound_health_deficit=health_features["depot_bound_health_deficit"],
            onsite_health_deficit=health_features["onsite_health_deficit"],
            maintenance_restoration_value=health_features["maintenance_restoration_value"],
            maintenance_enabled=True,
            logistics_enabled=False,
            time_remaining=time_remaining,
            shift_length=shift_length,
            demand_horizon_enabled=True,
            current_time_minutes=t,
            current_day_of_week=eval_day,
            target_matrix=target_matrix,
            total_stations=n_stations,
            next_station_idx=self._sid_to_idx.get(next_station_id, -1) if next_station_id else -1,
            cur_station_idx=cur_station_idx,
            dist_to_next=dist_to_next,
            max_travel_time=self._max_travel_time,
            next_is_depot=bool(next_station_id and next_station_id == self._depot_id),
            destination_features_enabled=True,
            include_bias=self.use_bias_feature,
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

    def get_feature_importance(self) -> list:
        """Return [(feature_name, weight)] sorted by |weight| descending."""
        return sorted(zip(self.FEATURE_NAMES, self.theta.tolist()), key=lambda x: -abs(x[1]))

    def log_weight_diagnostics(self) -> None:
        """Print full feature weight table sorted by magnitude. Call after training."""
        print(f"\n{'─'*54}")
        print(f"VFA WEIGHTS  |θ|={np.linalg.norm(self.theta):.4f}  n={self.N_FEATURES}")
        print(f"  {'Feature':<38} {'θ':>8}  bar")
        print(f"  {'─'*52}")
        for name, w in self.get_feature_importance():
            bar = ('+'if w >= 0 else '-') + '█' * min(int(abs(w) * 15), 20)
            print(f"  {name:<38} {w:>+8.4f}  {bar}")
        print(f"{'─'*54}\n")

    def value(self, phi: np.ndarray) -> float:
        """V(S^x) = θᵀ φ(S^x)."""
        return float(np.dot(self.theta, phi))

    def _update_feature_center(self, phis: list[np.ndarray]) -> None:
        """Update the running feature center from a candidate set."""
        if not self.use_feature_centering or not phis:
            return

        batch_mean = np.mean(np.vstack(phis), axis=0)
        if self._bias_feature_idx is not None:
            batch_mean[self._bias_feature_idx] = 0.0

        if not self._feature_center_initialized:
            self._feature_center = batch_mean.astype(np.float64)
            self._feature_center_initialized = True
        else:
            beta = self.feature_centering_beta
            self._feature_center = ((1.0 - beta) * self._feature_center) + (beta * batch_mean)

        if self._bias_feature_idx is not None:
            self._feature_center[self._bias_feature_idx] = 0.0

    def _apply_feature_centering(self, phi: np.ndarray) -> np.ndarray:
        if not self.use_feature_centering or not self._feature_center_initialized:
            return phi

        centered = phi.astype(np.float64, copy=True)
        centered -= self._feature_center
        if self._bias_feature_idx is not None:
            centered[self._bias_feature_idx] = phi[self._bias_feature_idx]
        return centered

    def _prepare_candidate_features(self, phis: list[np.ndarray]) -> list[np.ndarray]:
        if not self.use_feature_centering:
            return phis
        if self.learning_mode:
            self._update_feature_center(phis)
        return [self._apply_feature_centering(phi) for phi in phis]

    # ─────────────────────────────────────────────────────────────────────────
    # Shift timing helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_time_remaining(self, state, vehicle) -> Optional[float]:
        """
        Get time remaining in vehicle's shift (minutes).
        
        Returns:
            time_remaining : float (minutes) or None if shift_timing not enabled
        """
        if not self.logistics_enabled:
            return None
        
        if hasattr(vehicle, "shift_end_time") and vehicle.shift_end_time is not None:
            return max(0.0, vehicle.shift_end_time - state.time)

        close_min = SERVICE_TIME_TO * 60.0
        clock_min = float(state.time % 1440.0)
        return max(0.0, close_min - clock_min)
    
    def _get_shift_length(self, state, vehicle) -> float:
        """
        Get reference shift length for normalization (minutes).
        
        Default: 1440 minutes (24 hours).
        Can be customized per vehicle if needed.
        """
        if hasattr(vehicle, "shift_length") and vehicle.shift_length is not None:
            return float(vehicle.shift_length)
        # TODO(logistics): This fallback uses the full simulator duration, which
        # can be multi-day and is not a real shift length. It is currently
        # dormant because logistics/SLC features are inactive, but replace with
        # a daily service-window reference before re-enabling logistics features.
        if self._simulator_ref is not None and hasattr(self._simulator_ref, "duration"):
            return float(self._simulator_ref.duration)
            
        # Fall back to the same 24-hour reference used by MDPState.
        return 1440

    # ─────────────────────────────────────────────────────────────────────────
    # TD update
    # ─────────────────────────────────────────────────────────────────────────

    
    def _buffer_transition_for_batch_update(
        self,
        phi_cur: np.ndarray,
        reward: float,
        phi_next: np.ndarray,
        elapsed_minutes: float,
    ) -> None:
        """Buffer one TD sample and optionally apply a periodic mean-gradient update."""
        self.batch_buffer.append((phi_cur.copy(), reward, phi_next.copy(), elapsed_minutes))
        if self.transition_update_interval > 0 and len(self.batch_buffer) >= self.transition_update_interval:
            self.apply_batch_update()
            self._transition_batch_update_count += 1

    def _uses_online_td_lambda(self) -> bool:
        """Use persistent online eligibility traces when lambda is genuinely active."""
        return (
            getattr(self, "use_td_lambda", False)
            and float(getattr(self, "td_lambda", 0.0)) > 0.0
            and not getattr(self, "use_experience_replay", False)
        )

    def _apply_online_td_lambda_update(
        self,
        reward: float,
        phi_next: np.ndarray,
        elapsed_minutes: float,
    ) -> float:
        """
        Semi-gradient online TD(lambda) with an accumulating eligibility trace.

        Unlike the batch path, self._eligibility_trace persists across decisions
        and is reset only at episode boundaries.
        """
        if self._prev_phi is None:
            return 0.0

        v_cur = self.value(self._prev_phi)
        v_next = self.value(phi_next)
        discount = self.gamma ** (elapsed_minutes / 60.0)
        midpoint_discount = self.gamma ** ((elapsed_minutes / 2.0) / 60.0)
        td_error = (reward * midpoint_discount) + (discount * v_next) - v_cur

        self._eligibility_trace = (
            discount * self.td_lambda * self._eligibility_trace
            + self._prev_phi
        )
        step = self.alpha * td_error * self._eligibility_trace
        self.theta += step
        self.weights = list(self.theta)
        self._td_lambda_step_count += 1

        if self._all_phis_for_corr is not None:
            self._all_phis_for_corr.append(self._prev_phi.copy())

        if VFA_DEBUG_FLAGS.get("check2_td_updates") and self._ep_update_count < 5:
            print(
                f"  [CHK2 td#{self._ep_update_count+1} ONLINE λ] "
                f"r={reward:.4f} V_cur={v_cur:.4f} V_nxt={v_next:.4f} "
                f"δ={td_error:.4f} α={self.alpha:.5f} "
                f"‖φ‖={np.linalg.norm(self._prev_phi):.4f} "
                f"‖e‖={np.linalg.norm(self._eligibility_trace):.4f} "
                f"‖θ‖={np.linalg.norm(self.theta):.4f} "
                f"‖step‖={np.linalg.norm(step):.6f}"
            )

        return float(td_error)

    def td_update(self, reward: float, phi_next: np.ndarray, elapsed_minutes: float) -> float:
        if self._prev_phi is None:
            return 0.0

        if self._uses_online_td_lambda():
            td_error = self._apply_online_td_lambda_update(
                reward=reward,
                phi_next=phi_next,
                elapsed_minutes=elapsed_minutes,
            )
            self._ep_update_count += 1
            return td_error

        # Compute pre-update values for logging
        v_cur = self.value(self._prev_phi)
        v_next = self.value(phi_next)
        
        # Apply continuous time discounting.
        # The aggregate reward is treated as occurring at the interval midpoint,
        # while the continuation value is discounted over the full interval.
        discount = self.gamma ** (elapsed_minutes / 60.0)
        midpoint_discount = self.gamma ** ((elapsed_minutes / 2.0) / 60.0)

        td_error = (reward * midpoint_discount) + (discount * v_next) - v_cur
        raw_td_error = td_error
        if self.use_batch_td_clip:
            td_error = float(np.clip(td_error, -self.batch_td_clip_value, self.batch_td_clip_value))

        # [Check 2] First 5 TD transitions per episode
        if VFA_DEBUG_FLAGS.get("check2_td_updates") and self._ep_update_count < 5:
            upd_contrib = abs(self.alpha * td_error * np.linalg.norm(self._prev_phi))
            print(f"  [CHK2 td#{self._ep_update_count+1}] r={reward:.4f} V_cur={v_cur:.4f} V_nxt={v_next:.4f} "
                  f"δ={td_error:.4f} α={self.alpha:.5f} "
                  f"‖φ‖={np.linalg.norm(self._prev_phi):.4f} ‖θ‖={np.linalg.norm(self.theta):.4f} "
                  f"‖upd_contrib‖={upd_contrib:.6f}")
        self._ep_update_count += 1

        # --- SMART LOGGING ---
        if reward < -0.01 or abs(td_error) > 1.0:
            target_value = (reward * midpoint_discount) + discount * v_next
            # print(f"[TD] r={reward:.3f} vs={v_cur:.3f} tgt={target_value:.3f} err={td_error:.3f}")

        if self.use_online_td_updates:
            if getattr(self, 'use_td_lambda', False):
                self._eligibility_trace = discount * self.td_lambda * self._eligibility_trace + self._prev_phi
                gradient_vector = self._eligibility_trace
            else:
                gradient_vector = self._prev_phi

            self.theta += self.alpha * td_error * gradient_vector
            self.weights = list(self.theta)
            self._online_update_count += 1

            self._online_diag_phis.append(self._prev_phi.copy())
            self._online_diag_tderrs.append(float(td_error))
            self._online_diag_raw_tderrs.append(float(raw_td_error))
        elif getattr(self, 'use_experience_replay', False):
            self.replay_buffer.append((self._prev_phi.copy(), reward, phi_next.copy(), elapsed_minutes))
            if len(self.replay_buffer) >= max(1000, self.mini_batch_size):
                self.apply_mini_batch_update()
        else:
            # ALWAYS buffer the transition for batch processing!
            # The TD(lambda) trace logic is now handled inside apply_batch_update
            self._buffer_transition_for_batch_update(self._prev_phi, reward, phi_next, elapsed_minutes)

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
            
            # Midpoint discount for aggregate interval reward; full interval
            # discount for continuation value.
            discount = self.gamma ** (elapsed_minutes / 60.0)
            midpoint_discount = self.gamma ** ((elapsed_minutes / 2.0) / 60.0)
            td_error = np.clip((reward * midpoint_discount) + (discount * v_next) - v_cur, -5.0, 5.0)

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
        Aggregate accumulated TD errors over frozen trajectories,
        compute Eligibility Traces if enabled, and apply a MEAN gradient update.
        """
        if not self.batch_buffer:
            return

        n_transitions = len(self.batch_buffer)
        total_gradient = np.zeros_like(self.theta)
        total_td = 0.0
        eligibility_trace = np.zeros_like(self.theta)

        # Debug collectors (populated in the loop below)
        _dbg_rewards, _dbg_targets, _dbg_preds, _dbg_tderrs, _dbg_raw_tderrs = [], [], [], [], []
        _dbg_phis, _dbg_trace_norms = [], []

        for phi_cur, reward, phi_next, elapsed_minutes in self.batch_buffer:
            v_cur = self.value(phi_cur)
            v_next = self.value(phi_next)
            discount = self.gamma ** (elapsed_minutes / 60.0)
            midpoint_discount = self.gamma ** ((elapsed_minutes / 2.0) / 60.0)
            td_error = (reward * midpoint_discount) + (discount * v_next) - v_cur
            raw_td_error = td_error
            if self.use_batch_td_clip:
                td_error = float(np.clip(td_error, -self.batch_td_clip_value, self.batch_td_clip_value))
            total_td += td_error

            _dbg_rewards.append(reward)
            _dbg_targets.append((reward * midpoint_discount) + discount * v_next)
            _dbg_preds.append(v_cur)
            _dbg_tderrs.append(td_error)
            _dbg_raw_tderrs.append(raw_td_error)
            _dbg_phis.append(phi_cur)

            if getattr(self, 'use_td_lambda', False):
                eligibility_trace = discount * self.td_lambda * eligibility_trace + phi_cur
                gradient_vector = eligibility_trace
                _dbg_trace_norms.append(np.linalg.norm(eligibility_trace))
            else:
                gradient_vector = phi_cur

            total_gradient += td_error * gradient_vector

        mean_td = total_td / n_transitions
        old_theta = self.theta.copy()
        mean_gradient = total_gradient / n_transitions
        step = self.alpha * mean_gradient
        self.theta += step
        self.weights = list(self.theta)

        if self._collect_phis_inside_batch_update:
            all_phis_for_corr = self._all_phis_for_corr
            if all_phis_for_corr is None:
                all_phis_for_corr = []
                self._all_phis_for_corr = all_phis_for_corr
            all_phis_for_corr.extend(phi.copy() for phi in _dbg_phis)

        # --- EXISTING BATCH LOGGING ---
        grad_norm = np.linalg.norm(mean_gradient)
        weight_diff = np.linalg.norm(self.theta - old_theta)
        step_norm = np.linalg.norm(step)
        mode_str = f"TD(λ={self.td_lambda})" if getattr(self, 'use_td_lambda', False) else "TD(0)"
        clip_msg = ""
        if self.use_batch_td_clip and _dbg_raw_tderrs:
            raw_td = np.array(_dbg_raw_tderrs)
            used_td = np.array(_dbg_tderrs)
            clip_frac = float(np.mean(np.abs(raw_td) > self.batch_td_clip_value))
            clip_msg = (
                f" | clip={self.batch_td_clip_value:g}"
                f" frac={clip_frac:.3f}"
                f" raw_mean={raw_td.mean():.4f}"
                f" used_mean={used_td.mean():.4f}"
            )
        update_label = "TRANSITION-BATCH" if self.transition_update_interval > 0 else "BATCH"
        print(f"    [{mode_str} {update_label} UPDATE] n={n_transitions} | Mean TD: {mean_td:.4f} | Grad Norm: {grad_norm:.4f} | Step Norm: {step_norm:.4f}{clip_msg}")

        self._batch_update_count = getattr(self, '_batch_update_count', 0) + 1
        if self._batch_update_count % 5 == 0:
            self.log_weight_diagnostics()

        # --- [Check 3] Mean vs Sum gradient verification ---
        if VFA_DEBUG_FLAGS.get("check3_mean_vs_sum"):
            sum_gn  = np.linalg.norm(total_gradient)
            mean_gn = np.linalg.norm(mean_gradient)
            ratio   = sum_gn / (mean_gn + 1e-12)
            print(f"  [CHK3] n={n_transitions} ‖Σg‖={sum_gn:.4f} ‖μg‖={mean_gn:.4f} "
                  f"ratio={ratio:.2f} (expect≈{n_transitions}) ‖upd‖={step_norm:.6f}")

        # --- [Check 4] Eligibility trace diagnostics ---
        if getattr(self, 'use_td_lambda', False) and VFA_DEBUG_FLAGS.get("check4_traces") and _dbg_trace_norms:
            tn = np.array(_dbg_trace_norms)
            nz = int(np.sum(np.abs(eligibility_trace) > 1e-9))
            print(f"  [CHK4] Trace ‖e‖: mean={tn.mean():.4f} max={tn.max():.4f} "
                  f"min={tn.min():.6f} nonzero_final={nz}/{self.N_FEATURES}")

        # --- [Check 5] Reward sanity ---
        if VFA_DEBUG_FLAGS.get("check5_rewards"):
            rw = np.array(_dbg_rewards)
            print(f"  [CHK5] Reward: mean={rw.mean():.4f} std={rw.std():.4f} "
                  f"min={rw.min():.4f} max={rw.max():.4f} "
                  f"frac_neg={np.mean(rw < 0):.3f} frac_zero={np.mean(rw == 0):.3f}")

        # --- [Check 6] Target / prediction / td_error sanity ---
        if VFA_DEBUG_FLAGS.get("check6_targets"):
            tg = np.array(_dbg_targets)
            pr = np.array(_dbg_preds)
            td = np.array(_dbg_tderrs)
            print(f"  [CHK6] Target: mean={tg.mean():.4f} std={tg.std():.4f} | "
                  f"Pred: mean={pr.mean():.4f} std={pr.std():.4f} | "
                  f"TD_err: mean={td.mean():.4f} std={td.std():.4f}")

        every_n = VFA_DEBUG_FLAGS.get("every_n_episodes", 10)
        ep_cnt  = self._batch_update_count

        # --- [Check 7] Feature scale report (every N episodes) ---
        if VFA_DEBUG_FLAGS.get("check7_feature_scale") and ep_cnt % every_n == 0 and _dbg_phis:
            phis_arr = np.array(_dbg_phis)
            print(f"  [CHK7] Feature scale (batch_update #{ep_cnt}):")
            for i, fname in enumerate(self.FEATURE_NAMES):
                col  = phis_arr[:, i]
                p95  = float(np.percentile(np.abs(col), 95))
                fz   = float(np.mean(col == 0))
                print(f"    {fname:<40} mean={col.mean():+.4f} std={col.std():.4f} "
                      f"min={col.min():.4f} max={col.max():.4f} p95={p95:.4f} frac0={fz:.3f}")

        # --- [Check 8] Per-feature update direction (every N episodes) ---
        if VFA_DEBUG_FLAGS.get("check8_update_direction") and ep_cnt % every_n == 0 and _dbg_phis:
            phis_arr = np.array(_dbg_phis)
            td_arr   = np.array(_dbg_tderrs)
            print(f"  [CHK8] Per-feature update direction (batch_update #{ep_cnt}):")
            for i, fname in enumerate(self.FEATURE_NAMES):
                mean_contrib = float(np.mean(td_arr * phis_arr[:, i]))
                mean_upd     = self.alpha * mean_contrib
                print(f"    {fname:<40} mean(δ·φ)={mean_contrib:+.6f} "
                      f"mean_upd={mean_upd:+.8f} "
                      f"θ_before={old_theta[i]:+.4f} θ_after={self.theta[i]:+.4f}")

        self.batch_buffer.clear()

    def flush_online_update_diagnostics(self, episode_number: int) -> None:
        """Print compact diagnostics for online TD updates and clear per-episode buffers."""
        phis = self._online_diag_phis
        if not phis:
            return

        td = np.array(self._online_diag_tderrs, dtype=np.float64)
        raw_td = np.array(self._online_diag_raw_tderrs, dtype=np.float64)
        phis_arr = np.array(phis)

        print(
            f"    [ONLINE TD UPDATE] n={len(phis)} | "
            f"Mean TD: {td.mean():.4f} | TD std: {td.std():.4f}"
        )

        if self.use_batch_td_clip and len(raw_td) > 0:
            clip_frac = float(np.mean(np.abs(raw_td) > self.batch_td_clip_value))
            print(
                f"  [ONLINE CLIP] clip={self.batch_td_clip_value:g} "
                f"frac={clip_frac:.3f} raw_mean={raw_td.mean():.4f} "
                f"used_mean={td.mean():.4f}"
            )

        every_n = VFA_DEBUG_FLAGS.get("every_n_episodes", 10)
        if VFA_DEBUG_FLAGS.get("check7_feature_scale") and episode_number % every_n == 0:
            print(f"  [CHK7] Feature scale (online episode #{episode_number}):")
            for i, fname in enumerate(self.FEATURE_NAMES):
                col = phis_arr[:, i]
                p95 = float(np.percentile(np.abs(col), 95))
                fz = float(np.mean(col == 0))
                print(
                    f"    {fname:<40} mean={col.mean():+.4f} std={col.std():.4f} "
                    f"min={col.min():.4f} max={col.max():.4f} p95={p95:.4f} frac0={fz:.3f}"
                )

        self._online_diag_phis = []
        self._online_diag_tderrs = []
        self._online_diag_raw_tderrs = []

    def add_terminal_update(self, state) -> None:
        """Append final transition with zero bootstrap so the last post-decision state is trained."""
        if not self.use_terminal_update or self._prev_phi is None:
            return

        reward = self.reward_calc.compute_step_reward(state.metrics)
        reward += self.reward_calc.compute_fleet_penalty(state)
        reward = self.reward_calc.center_reward(reward)

        prev_time = self._prev_time if self._prev_time is not None else state.time
        elapsed_minutes = max(0.0, float(state.time - prev_time))
        phi_terminal = np.zeros_like(self._prev_phi)
        v_cur = self.value(self._prev_phi)
        td_error = reward - v_cur
        if self.use_batch_td_clip:
            td_error_used = float(np.clip(td_error, -self.batch_td_clip_value, self.batch_td_clip_value))
        else:
            td_error_used = td_error

        if self.use_online_td_updates:
            self.td_update(reward, phi_terminal, elapsed_minutes)
        else:
            self._buffer_transition_for_batch_update(self._prev_phi, reward, phi_terminal, elapsed_minutes)
        self._last_terminal_update = {
            "reward": float(reward),
            "elapsed_minutes": elapsed_minutes,
            "v_cur": float(v_cur),
            "td_error_raw": float(td_error),
            "td_error_used": float(td_error_used),
        }
        print(
            "  [TERMINAL UPDATE] "
            f"r={reward:.4f} elapsed={elapsed_minutes:.1f} "
            f"V_cur={v_cur:.4f} td_raw={td_error:.4f} td_used={td_error_used:.4f}"
        )
    # ─────────────────────────────────────────────────────────────────────────
    # Action generation  (action-space splitting)
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_candidates(self, state, vehicle) -> List[sim.Action]:
        """
        DECOUPLED CANDIDATE GENERATION:
        Delegated to candidate_generator.py to create combination of operations and routing.
        """
        return cast(List[sim.Action], generate_candidates(state, vehicle, self.maintenance_enabled))

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
            f"[DEPOT] {format_sim_time(state.time)} | "
            f"vehicle={vehicle.id} | "
            f"from={vehicle.location.id} → to={destination_id} | "
            f"cargo_func={len(vehicle.get_bike_inventory())} | "
            f"V(S^x)={value:8.4f} | "
        )
        
        if self.logistics_enabled:
            log_entry += f"t_rem={time_rem:.1f}min ({time_frac:.2%}) | "
        
        log_entry += f"φ_1={phi[0]:.4f} "  # rebalancing_imbalance
        
        if self.maintenance_enabled and "trailer_cannibalization" in self.FEATURE_NAMES:
            idx = self.FEATURE_NAMES.index("trailer_cannibalization")
            log_entry += f"φ_trailer={phi[idx]:.4f} "
        idx_time = self.N_FEATURES  # logistics features; disabled so this is unused

        if self.logistics_enabled and len(phi) > idx_time:
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

    def _candidate_action_type(self, action, at_depot: bool) -> str:
        if len(getattr(action, "onsite_repairs", [])) > 0:
            return "ONSITE_REP"
        if at_depot and len(getattr(action, "pick_ups", [])) > 0:
            return "DEPOT_PICK"
        if at_depot and len(getattr(action, "delivery_bikes", [])) > 0:
            return "DEPOT_DROP"
        if len(getattr(action, "pick_ups", [])) > 0:
            return "PICKUP"
        if len(getattr(action, "delivery_bikes", [])) > 0:
            return "DELIVER"
        return "IDLE"

    def _log_candidate_diagnostics(self, state, vehicle, candidates, values, sel_idx: int, mode: str, td_err: float) -> None:
        """Collect compact per-decision candidate value spread diagnostics."""
        if not self.log_candidate_diagnostics:
            return

        ranked_idx = np.argsort(values)[::-1]
        best_idx = int(ranked_idx[0])
        second_idx = int(ranked_idx[1]) if len(ranked_idx) > 1 else best_idx
        worst_idx = int(ranked_idx[-1])
        selected = candidates[sel_idx]
        best = candidates[best_idx]
        at_depot = vehicle.is_at_depot()
        tie_tol = 1e-10
        selected_is_greedy = bool(values[sel_idx] >= values[best_idx] - tie_tol)
        selected_rank = int(np.sum(values > values[sel_idx] + tie_tol)) + 1

        _t = state.time
        _day = int(_t // 1440)
        _clock = _t % 1440
        _hh = int(_clock // 60)
        _mm = int(_clock % 60)

        self._candidate_diag_rows.append({
            "episode": self._comparison_episode,
            "decision_id": self._ep_decision_count,
            "day": _day,
            "time_hhmm": f"{_hh:02d}:{_mm:02d}",
            "time_min": round(_t, 1),
            "current_station": vehicle.location.id,
            "n_candidates": len(candidates),
            "value_best": float(values[best_idx]),
            "value_second": float(values[second_idx]),
            "value_worst": float(values[worst_idx]),
            "value_mean": float(np.mean(values)),
            "value_std": float(np.std(values)),
            "spread_best_worst": float(values[best_idx] - values[worst_idx]),
            "gap_best_second": float(values[best_idx] - values[second_idx]),
            "selected_rank": selected_rank,
            "selected_value": float(values[sel_idx]),
            "selected_gap_from_best": float(values[sel_idx] - values[best_idx]),
            "selected_is_greedy": selected_is_greedy,
            "mode": "explore" if mode == "E" else "exploit",
            "epsilon": float(self.epsilon),
            "alpha": float(self.alpha),
            "td_error_from_prev": float(td_err),
            "selected_type": self._candidate_action_type(selected, at_depot),
            "selected_next_station": getattr(selected, "next_location", getattr(selected, "next_station", "")),
            "selected_pickups": len(getattr(selected, "pick_ups", [])),
            "selected_deliveries": len(getattr(selected, "delivery_bikes", [])),
            "selected_onsite_repairs": len(getattr(selected, "onsite_repairs", [])),
            "best_type": self._candidate_action_type(best, at_depot),
            "best_next_station": getattr(best, "next_location", getattr(best, "next_station", "")),
        })

    def flush_candidate_diagnostics(self, path: str) -> None:
        """Append buffered candidate spread diagnostics to CSV."""
        import csv, os  # noqa: PLC0415
        rows = self._candidate_diag_rows
        if not rows:
            return
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        self._candidate_diag_rows = []

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
        4. (If learning) run TD update using reward since last decision
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
        health_base = self._compute_health_base(state, vehicle) if self._use_health_features else None

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
            delta_depot_fixed_queue = 0
            delta_depot_in_repair = 0
 
            if vehicle.is_at_depot():
                # pick_ups at depot come from fixed_queue (repaired bikes), not depot.bikes.
                # delivery_bikes are broken bikes being dropped off — not functional deliveries.
                fq = getattr(vehicle.location, "fixed_queue", {})
                vehicle_bikes = {
                    getattr(b, 'bike_id', getattr(b, 'id')): b
                    for b in vehicle.get_bike_inventory()
                }
                for b_id in action.pick_ups:
                    if b_id in fq:
                        functional_pickups += 1
                depot_dropoffs = sum(
                    1 for b_id in action.delivery_bikes
                    if getattr(vehicle_bikes.get(b_id), 'damage_status', None) == 'depot'
                )
                delta_func = -functional_pickups   # vehicle GAINS repaired bikes; no functional deliveries
                delta_depot_cargo = -depot_dropoffs
                delta_depot_fixed_queue = -functional_pickups
                delta_depot_in_repair = depot_dropoffs
            else:
                for b_id in action.pick_ups:
                    b = station_bikes.get(b_id)
                    if b and getattr(b, 'damage_status', None) == 'depot':
                        depot_pickups += 1
                    elif b:
                        functional_pickups += 1
                delta_func = len(action.delivery_bikes) - functional_pickups
                delta_depot_cargo = depot_pickups
 
            delta_onsite_repairs = len(getattr(action, "onsite_repairs", []))
            
            dest_id = getattr(action, "next_location", getattr(action, "next_station", None))

            try:
                svc = action.get_action_time(0.0) if hasattr(action, "get_action_time") else 0.0
            except Exception:
                svc = 0.0
            try:
                travel = state.get_vehicle_travel_time(vehicle.location.id, dest_id) if dest_id else 0.0
            except Exception:
                travel = 0.0
            action_eval_time = state.time + svc + travel
 
            '''if not getattr(self, "_has_printed_eval_time", False):
                print(f"[EVAL_TIME] t_now={state.time:.1f}  dest={dest_id}  svc={svc:.1f}  travel={travel:.1f}  eval_t={action_eval_time:.1f}  shift={(action_eval_time - state.time):.1f}min")
                if k == len(candidates) - 1:
                    self._has_printed_eval_time = True'''
 
            time_rem = self._get_time_remaining(state, vehicle)
            shift_len = self._get_shift_length(state, vehicle)
 
            phi = self.extract_features(
                state, vehicle,
                base_func, base_onsite, base_depot,
                delta_func, delta_depot_cargo,
                delta_onsite_repairs,
                delta_depot_fixed_queue,
                delta_depot_in_repair,
                time_remaining=time_rem,
                shift_length=shift_len,
                next_station_id=dest_id,
                eval_time=action_eval_time,
                projected_time=action_eval_time,
                dist_to_next=travel,
                candidate_action=action,
                health_base=health_base,
            )
            phis.append(phi)

        phis = self._prepare_candidate_features(phis)
        for k, phi in enumerate(phis):
            values[k] = self.value(phi)

        # --- DEBUG 2: SPATIAL PARALYSIS ---
        if not getattr(self, "_has_printed_spatial", False) and "proximity_to_demand_gravity" in self.FEATURE_NAMES:
            idx = self.FEATURE_NAMES.index("proximity_to_demand_gravity")
            print(f"\n[DEBUG - SPATIAL] Gravity values for 8 candidates from {vehicle.location.id}:")
            for k, action in enumerate(candidates):
                dest = getattr(action, "next_location", getattr(action, "next_station", None))
                print(f"  -> Going to {dest} | Gravity Feature: {phis[k][idx]:.6f}")
            self._has_printed_spatial = True
 
        # ── Step 4: TD update ─────────────────────────────────────────────
        td_err = 0.0
        if self.learning_mode and self._prev_phi is not None:
            # Consume the reward signal safely
            reward = self.reward_calc.compute_step_reward(state.metrics)
            reward += self.reward_calc.compute_fleet_penalty(state)
            reward += self.reward_calc.compute_late_shift_penalty(vehicle, state)
            reward = self.reward_calc.center_reward(reward)
            if self.log_greedy_comparison:
                pending = self._pending_comparison_row
                if pending is not None:
                    pending["realized_reward"] = round(reward, 6)
            
            prev_time = self._prev_time if self._prev_time is not None else state.time
            elapsed_minutes = state.time - prev_time
            
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
            sel_idx = int(self._rng.integers(len(candidates)))
            mode = "E"
        else:
            sel_idx = int(np.argmax(values))
            mode = "X"
 
        selected = candidates[sel_idx]
        self._log_candidate_diagnostics(state, vehicle, candidates, values, sel_idx, mode, td_err)
        _ns = getattr(selected, 'next_location', getattr(selected, 'next_station', None))
        _pk = len(getattr(selected, 'pick_ups', []))
        _dl = len(getattr(selected, 'delivery_bikes', []))
        #print(f"[{mode}] {_ns} pk={_pk} dl={_dl} (idx={sel_idx}, v={values[sel_idx]:.4f})")
 
        # ── Noon snapshot: ranked candidate table (once per simulated day) ───
        if state.hour() == 12 and getattr(self, '_noon_log_day', -1) != state.day():
            self._noon_log_day = state.day()
            at_depot = vehicle.is_at_depot()
 
            def _cls(a):
                return self._candidate_action_type(a, at_depot)
 
            ranked = sorted(zip(values.tolist(), candidates), key=lambda x: -x[0])
            v_best  = ranked[0][0]
            v_worst = ranked[-1][0]
            v_spread = v_best - v_worst
            v_mean  = float(np.mean(values))
            v_std   = float(np.std(values))
            print(f"\n{'═'*62}")
            print(f"NOON  {format_sim_time(state.time)}  {vehicle.id} @ {vehicle.location.id}  n={len(candidates)}")
            print(f"  spread={v_spread:.4f}  best={v_best:.4f}  worst={v_worst:.4f}  mean={v_mean:.4f}  std={v_std:.4f}")
            print(f"  {'#':<3} {'Type':<12} {'Next':<7} {'pk':>3} {'dl':>3} {'rep':>4}  {'V':>9}  {'Δbest':>7}")
            print(f"  {'─'*57}")
            for rank, (v, a) in enumerate(ranked, 1):
                atype = _cls(a)
                ns    = str(getattr(a, 'next_location', '?'))
                pk    = len(a.pick_ups)
                dl    = len(a.delivery_bikes)
                rep   = len(getattr(a, 'onsite_repairs', []))
                delta = v - v_best
                mark  = "  ◄" if (a is selected) else ""
                print(f"  {rank:<3} {atype:<12} {ns:<7} {pk:>3} {dl:>3} {rep:>4}  {v:>9.4f}  {delta:>+7.4f}{mark}")
            print(f"  mode={'exploit' if mode == 'X' else 'explore'}  ε={self.epsilon:.3f}  α={self.alpha}")
            print(f"{'═'*62}\n")
 
        # ── Step 6: Log depot decisions (optional) ────────────────────────
        self._log_depot_decision(state, vehicle, selected, phis[sel_idx], values[sel_idx])
 
        # ── Step 7: cache post-decision features for next TD update ───────
        self._prev_phi = phis[sel_idx]
        self._prev_time = state.time

        # ── Step 8: Log the Brain's Decision (NEW) ────────────────────────
        if getattr(self, 'log_rl_decisions', False):
            phi_dict = dict(zip(self.FEATURE_NAMES, phis[sel_idx].tolist()))
            log_entry = {
                'time': state.time,
                'vehicle_id': vehicle.id,
                'station_id': vehicle.location.id,
                'action_next_station': getattr(selected, 'next_location', 'N/A'),
                'action_pickups': len(getattr(selected, 'pick_ups', [])),
                'action_deliveries': len(getattr(selected, 'delivery_bikes', [])),
                'expected_value_V': values[sel_idx],
                'td_error': float(td_err),
            }
            log_entry.update(phi_dict)
            self.rl_logs.append(log_entry)
 
        # ── Step 9: Forward to RunLogger if attached ──────────────────────
        if self.logger is not None:
            self._log_to_run_logger(state, vehicle, selected, phis[sel_idx], values[sel_idx])
 
        record_depot_action_stats(selected, vehicle)

        if self.log_greedy_comparison:
            self._log_greedy_comparison(state, vehicle, candidates, values, sel_idx)

        self._ep_decision_count += 1

        return selected

     
    def _log_to_run_logger(self, state, vehicle, action, phi: np.ndarray, vfa_value: float) -> None:
        """Build a RunLogger decision row for VFA-standalone evaluation."""
        logger = self.logger
        if logger is None:
            return
        current_time = state.time
        clock_min  = current_time % (24 * 60)
        day        = int(current_time // (24 * 60))
        clock_hour = int(clock_min // 60)
        minute     = int(clock_min % 60)
 
        inv          = vehicle.get_bike_inventory()
        func_before  = sum(1 for b in inv if getattr(b, "damage_status", None) not in ("depot", "onsite"))
        depot_before = sum(1 for b in inv if getattr(b, "damage_status", None) == "depot")
        total_before = len(inv)
 
        raw_bikes     = getattr(vehicle.location, "bikes", {})
        station_bikes = (raw_bikes if isinstance(raw_bikes, dict)
                         else {getattr(b, "bike_id", getattr(b, "id")): b for b in raw_bikes})
 
        is_at_depot      = vehicle.is_at_depot()
        pick_up_ids      = list(getattr(action, "pick_ups", []))
        delivery_ids     = list(getattr(action, "delivery_bikes", []))
 
        func_pickups = depot_pickups = 0
        load_from_queue = 0

        if is_at_depot:
            # sim.Action does not carry load_from_queue. At the simulator
            # boundary, repaired depot loads are encoded as pick_ups whose
            # ids are currently in depot.fixed_queue.
            fixed_queue = getattr(vehicle.location, "fixed_queue", {})
            load_from_queue = sum(1 for b_id in pick_up_ids if b_id in fixed_queue)
        else:
            for b_id in pick_up_ids:
                b = station_bikes.get(b_id)
                if b and getattr(b, "damage_status", None) == "depot":
                    depot_pickups += 1
                elif b:
                    func_pickups += 1

        vehicle_bikes = {getattr(b, "bike_id", getattr(b, "id")): b for b in inv}
        if is_at_depot:
            depot_deliveries = sum(
                1 for b_id in delivery_ids
                if getattr(vehicle_bikes.get(b_id), "damage_status", None) == "depot"
            )
            func_deliveries = 0
        else:
            depot_deliveries = 0
            func_deliveries = len(delivery_ids)

        onsite_repairs = len(getattr(action, "onsite_repairs", []))

        if is_at_depot:
            func_after  = func_before + load_from_queue
            depot_after = max(depot_before - depot_deliveries, 0)
            
        else:
            func_after  = func_before - func_deliveries + func_pickups
            depot_after = depot_before + depot_pickups
        total_after = max(func_after, 0) + max(depot_after, 0)
 
        dest = getattr(action, "next_location", None)
        try:
            travel_time = state.get_vehicle_travel_time(vehicle.location.id, dest) if dest else 0.0
        except Exception:
            travel_time = 0.0
 
        try:
            action_duration = action.get_action_time(0.0) if hasattr(action, "get_action_time") else 0.0
        except Exception:
            action_duration = 0.0
 
        n_damaged  = sum(1 for b in station_bikes.values() if getattr(b, "damage_status", None) is not None)
        maint_flag = (n_damaged > 0) or is_at_depot
        is_maint   = (onsite_repairs > 0) or (depot_pickups > 0) or is_at_depot
 
        bikes_involved = str(pick_up_ids + delivery_ids)
 
        # Feature values: active features filled, inactive left absent → written as "" by restval
        phi_dict = {f"phi_{name}": round(val, 6)
                    for name, val in zip(self.FEATURE_NAMES, phi.tolist())}
 
        logger.log_decision({
            "day":    day,
            "hour":   clock_hour,
            "minute": minute,
            "vehicle_id":               vehicle.id,
            "vehicle_policy_type":      self.__class__.__name__,
            "current_station_id":       vehicle.location.id,
            "is_at_depot":              is_at_depot,
            "functional_load_before":   func_before,
            "depot_load_before":        depot_before,
            "total_load_before":        total_before,
            "functional_deliveries":    func_deliveries,
            "functional_pickups":       func_pickups,
            "onsite_repairs":           onsite_repairs,
            "depot_pickups":            depot_pickups,
            "depot_deliveries":         depot_deliveries,
            "load_from_queue":          load_from_queue,
            "bikes_involved":           bikes_involved,
            "action_duration_min":      round(action_duration, 2),
            "next_station_id":          str(dest) if dest else "",
            "travel_time_min":          round(travel_time, 2),
            "functional_load_after":    max(func_after, 0),
            "depot_load_after":         max(depot_after, 0),
            "total_load_after":         total_after,
            "immediate_reward":         0.0,
            "vfa_value":                round(float(vfa_value), 6),
            # accumulated_rollout_reward and tail_value intentionally absent → empty cells
            "final_decision_score":     round(float(vfa_value), 6),
            "maintenance_flag_present": maint_flag,
            "selected_action_is_maintenance": is_maint,
            **phi_dict,
        })
 

    def _log_greedy_comparison(self, state, vehicle, candidates, values, sel_idx: int) -> None:
        """Compare VFA choice against GreedyMaintenancePolicy at the same decision state.

        Populates self._comparison_rows. Call flush_comparison_log() after each episode.
        Set self.log_greedy_comparison = True and optionally self._comparison_episode = ep
        before running to enable.
        """
        # Lazy-import to avoid circular dependency at module load time.
        from policies.greedy_policy_maintenance import GreedyMaintenancePolicy  # noqa: PLC0415

        greedy_policy = self._comparison_greedy
        if greedy_policy is None:
            greedy_policy = GreedyMaintenancePolicy()
            self._comparison_greedy = greedy_policy
            simulator = getattr(self, "_sim", None)
            if hasattr(greedy_policy, "init_sim") and simulator is not None:
                greedy_policy.init_sim(simulator)

        try:
            greedy_action = greedy_policy.get_best_action(state, vehicle)
            greedy_dest = getattr(greedy_action, "next_location", getattr(greedy_action, "next_station", None))
            greedy_onsite = len(getattr(greedy_action, "onsite_repairs", []))
            greedy_error = ""
        except Exception as exc:
            greedy_dest = ""
            greedy_onsite = -1
            greedy_error = type(exc).__name__

        vfa_action  = candidates[sel_idx]
        vfa_dest    = getattr(vfa_action, "next_location", getattr(vfa_action, "next_station", None))
        vfa_onsite  = len(getattr(vfa_action, "onsite_repairs", []))

        # Count depot removals in VFA action (broken bikes picked up from station).
        _station_bike_map = {getattr(b, "bike_id", getattr(b, "id")): b for b in
                             (list(vehicle.location.bikes.values()) if isinstance(vehicle.location.bikes, dict)
                              else list(vehicle.location.bikes))}
        vfa_depot = sum(
            1 for b_id in getattr(vfa_action, "pick_ups", [])
            if getattr(_station_bike_map.get(b_id), "damage_status", None) == "depot"
        )

        # Find greedy-equivalent candidate in the VFA set (same dest + same onsite count).
        greedy_value_in_candidates = None
        if not greedy_error:
            for k, cand in enumerate(candidates):
                c_dest   = getattr(cand, "next_location", getattr(cand, "next_station", None))
                c_onsite = len(getattr(cand, "onsite_repairs", []))
                if c_dest == greedy_dest and c_onsite == greedy_onsite:
                    greedy_value_in_candidates = float(values[k])
                    break

        vfa_score    = float(values[sel_idx])
        agreed       = (vfa_dest == greedy_dest and vfa_onsite == greedy_onsite)
        score_gap = (
            vfa_score - greedy_value_in_candidates
            if greedy_value_in_candidates is not None else None
        )

        vehicle_bikes = list(vehicle.get_bike_inventory())
        vehicle_depot_load = sum(
            1 for b in vehicle_bikes
            if getattr(b, "damage_status", None) == "depot"
        )
        vehicle_functional_load = sum(
            1 for b in vehicle_bikes
            if getattr(b, "damage_status", None) not in ("onsite", "depot")
        )

        if greedy_error:
            main_diff = f"greedy_error:{greedy_error}"
        elif vfa_onsite < greedy_onsite:
            main_diff = "vfa_less_onsite"
        elif vfa_dest != greedy_dest:
            main_diff = "different_route"
        elif greedy_value_in_candidates is None:
            main_diff = "greedy_not_in_candidates"
        else:
            main_diff = "none"

        _t       = state.time
        _day     = int(_t // 1440)
        _clock   = _t % 1440
        _hh      = int(_clock // 60)
        _mm      = int(_clock % 60)
        from policies.sjovik_sund.mdp.action_bridge import depot_stats as _ds  # noqa: PLC0415
        _snap = self._warmup_depot_snapshot
        row = {
            "episode":                    self._comparison_episode,
            "decision_id":                self._ep_decision_count,
            "day":                        _day,
            "time_hhmm":                  f"{_hh:02d}:{_mm:02d}",
            "time_min":                   round(_t, 1),
            "current_station":            vehicle.location.id,
            "greedy_next_station":        greedy_dest,
            "vfa_next_station":           vfa_dest,
            "vfa_onsite_repairs":         vfa_onsite,
            "vfa_depot_removals":         vfa_depot,
            "n_candidates":               len(candidates),
            "greedy_in_candidate_set":    greedy_value_in_candidates is not None,
            "greedy_candidate_vfa_score": greedy_value_in_candidates,
            "vfa_selected_score":         vfa_score,
            "score_gap":                  score_gap,
            "same_action":                agreed,
            "same_next_station":          vfa_dest == greedy_dest,
            "main_difference":            main_diff,
            "realized_reward":            None,  # backfilled at next decision
            "vehicle_total_load":          len(vehicle_bikes),
            "vehicle_functional_load":     vehicle_functional_load,
            "vehicle_depot_bound_load":    vehicle_depot_load,
            "depot_visits_so_far":        _ds["visits"]       - _snap.get("visits", 0),
            "depot_loaded_so_far":        _ds["loaded_bikes"] - _snap.get("loaded_bikes", 0),
            "depot_unloaded_so_far":      _ds["unloaded_bikes"] - _snap.get("unloaded_bikes", 0),
            "depot_skipped_loads_so_far": _ds["skipped_loads"] - _snap.get("skipped_loads", 0),
        }
        # Commit the previous pending row (now its reward is known) and hold this one.
        if self._pending_comparison_row is not None:
            self._comparison_rows.append(self._pending_comparison_row)
        self._pending_comparison_row = row

    def flush_comparison_log(self, path: str) -> None:
        """Append buffered comparison rows to CSV and clear the buffer."""
        import csv, os  # noqa: PLC0415
        # Commit the final pending row (last decision of episode — no next call to commit it).
        pending = self._pending_comparison_row
        if pending is not None:
            self._comparison_rows.append(pending)
            self._pending_comparison_row = None
        rows = self._comparison_rows
        if not rows:
            return
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        self._comparison_rows = []

    # ─────────────────────────────────────────────────────────────────────────
    # Episode boundary reset  (called by EpisodeTrainingPolicy.__init__)
    # ─────────────────────────────────────────────────────────────────────────

    def reset_episode(self) -> None:
        self.rl_logs = []
        self._prev_phi = None
        self._prev_time = None
        self._eligibility_trace[:] = 0.0
        self._td_lambda_step_count = 0
        self._ep_update_count   = 0
        self._ep_decision_count = 0
        self._online_update_count = 0
        self._transition_batch_update_count = 0
        self._online_diag_phis = []
        self._online_diag_tderrs = []
        self._online_diag_raw_tderrs = []
        self.reward_calc.reset_episode()
    # ─────────────────────────────────────────────────────────────────────────
    # Serialisation
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist θ and hyper-parameters to disk."""
        payload = {
            "theta":         self.theta,
            "n_features":    self.n_features,
            "alpha":         self.alpha,
            "gamma":         self.gamma,
            "feature_names": self.FEATURE_NAMES,
            "use_bias_feature": self.use_bias_feature,
            "use_terminal_update": self.use_terminal_update,
            "use_batch_td_clip": self.use_batch_td_clip,
            "batch_td_clip_value": self.batch_td_clip_value,
            "use_online_td_updates": self.use_online_td_updates,
            "transition_update_interval": self.transition_update_interval,
            "use_td_lambda": self.use_td_lambda,
            "td_lambda": self.td_lambda,
            "initial_bias": self.initial_bias,
            "use_feature_centering": self.use_feature_centering,
            "feature_centering_beta": self.feature_centering_beta,
            "feature_center": self._feature_center,
            "feature_center_initialized": self._feature_center_initialized,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"  [VFA] theta saved -> {path}")

    @classmethod
    def load(cls, path: Path, **kwargs) -> "LinearVFAPolicy":
        #Load a trained model from disk (learning_mode=False by default).
        with open(path, "rb") as f:
            payload = pickle.load(f)
        # Use stored feature names if available and not overridden by caller
        if "active_features" not in kwargs and payload.get("feature_names") is not None:
            kwargs["active_features"] = payload["feature_names"]
        if "use_bias_feature" not in kwargs:
            kwargs["use_bias_feature"] = bool(payload.get("use_bias_feature", False) or "bias" in payload.get("feature_names", []))
        kwargs.setdefault("use_terminal_update", bool(payload.get("use_terminal_update", False)))
        kwargs.setdefault("use_batch_td_clip", bool(payload.get("use_batch_td_clip", False)))
        kwargs.setdefault("batch_td_clip_value", float(payload.get("batch_td_clip_value", 10.0)))
        kwargs.setdefault("use_online_td_updates", bool(payload.get("use_online_td_updates", False)))
        kwargs.setdefault("transition_update_interval", int(payload.get("transition_update_interval", 0)))
        kwargs.setdefault("initial_bias", payload.get("initial_bias", None))
        kwargs.setdefault("use_feature_centering", bool(payload.get("use_feature_centering", False)))
        kwargs.setdefault("feature_centering_beta", float(payload.get("feature_centering_beta", 0.01)))
        policy         = cls(
            n_features   = payload["n_features"],
            alpha        = payload["alpha"],
            gamma        = payload["gamma"],
            learning_mode= False,
            **kwargs,
        )
        policy.use_td_lambda = bool(payload.get("use_td_lambda", False))
        policy.td_lambda = float(payload.get("td_lambda", 0.0))
        policy.theta   = payload["theta"]
        policy.weights = list(policy.theta)
        if payload.get("feature_center") is not None:
            policy._feature_center = np.array(payload["feature_center"], dtype=np.float64)
            policy._feature_center_initialized = bool(payload.get("feature_center_initialized", True))
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
            # Learning: VFA + TD update
            return self.vfa_policy.get_best_action(state, vehicle)

    def __repr__(self) -> str:
        return f"EpisodeTrainingPolicy()"
