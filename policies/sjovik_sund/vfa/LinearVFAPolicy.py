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
    ) -> None:
        super().__init__(maintenance_enabled=maintenance_enabled)
        
        self.maintenance_enabled = maintenance_enabled
        self.shift_timing_enabled = shift_timing_enabled
        self.log_depot_visits = log_depot_visits
        self.depot_log_file = depot_log_file
        self.FEATURE_NAMES = _get_feature_names(self.maintenance_enabled, self.shift_timing_enabled)
        self.N_FEATURES = len(self.FEATURE_NAMES)

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
        self.theta: np.ndarray = self._rng.standard_normal(n_features) * 0.01

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

    # ─────────────────────────────────────────────────────────────────────────
    # Lazy initialisation  (uses sim.State, not the full simulator)
    # ─────────────────────────────────────────────────────────────────────────

    def _lazy_init(self, state) -> None:
        """
        Build station metadata caches from sim.State.

        Called automatically on the first get_best_action() call.
        Safe to call again after a hard state reset (set _initialized=False).
        """
        stations = sorted(state.get_stations(), key=lambda s: s.id)
        N = len(stations)

        self._station_ids = [s.id for s in stations]
        self._sid_to_idx  = {sid: k for k, sid in enumerate(self._station_ids)}

        # Target inventory matrix  (7 days × 24 hours × N stations)
        self._target_matrix = np.array(
            [
                [[s.get_target_state(d, h) for s in stations] for h in range(24)]
                for d in range(7)
            ],
            dtype=np.float32,
        )

        # Time-averaged arrival rate per station  shape (N,)
        # Used for the demand-weighted depot-backlog feature (φ_4)
        self._activity = np.array(
            [
                np.mean(
                    [s.get_arrive_intensity(d, h) for d in range(7) for h in range(24)]
                )
                for s in stations
            ],
            dtype=np.float32,
        )

        # Closest-depot ID (used for φ_5 distance calculation)
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
        Build (func, onsite, depot) integer arrays of shape (N,) from the
        canonical MDP state snapshot.
        """
        N = len(self._station_ids)
        func = np.zeros(N, dtype=np.int32)
        onsite = np.zeros(N, dtype=np.int32)
        depot = np.zeros(N, dtype=np.int32)

        mdp_state = extract_mdp_state(
            sim_state=state,
            active_vehicle_id=vehicle.id,
            config=self.config,
            depot_id=self._depot_id,
        )

        for sid, inv in mdp_state.stations.items():
            k = self._sid_to_idx[sid]
            func[k] = inv.functional
            onsite[k] = inv.onsite
            depot[k] = inv.depot

        return func, onsite, depot

    # ─────────────────────────────────────────────────────────────────────────
    # Feature vector  φ(S^x)  – delegated to vfa_features.py
    # ─────────────────────────────────────────────────────────────────────────

    def extract_features(
        self,
        state,
        vehicle,
        delta_func: int = 0,
        delta_depot_cargo: int = 0,
    ) -> np.ndarray:
        """
        Compute φ(S^x) for the post-decision state resulting from applying a
        candidate action at the vehicle's current station.

                This method prepares simulator-derived inputs (inventories, target
                slice, vehicle cargo, depot distance) and delegates canonical feature
                definitions to vfa_features.extract().

                Canonical feature source:
                    - policies/sjovik_sund/vfa/vfa_features.py
                        * FEATURE_NAMES / N_FEATURES
                        * extract(...)
                        * as_dict(...)

        Args:
            state:             sim.State at the current decision epoch
            vehicle:           Active sim.Vehicle
            delta_func:        Change in functional bikes at current station
                               (positive = delivery, negative = pickup)
            delta_depot_cargo: Change in vehicle depot-bike cargo after action

        Returns:
            φ  np.ndarray of shape (n_features,)   dtype float64
        """
        func, onsite, depot = self._extract_inventories(state, vehicle)

        # ── Apply post-decision delta at the vehicle's current station ─────
        # (skip if at depot; depot inventory handled separately in MDP)
        if vehicle.location.id in self._sid_to_idx:
            cur_idx = self._sid_to_idx[vehicle.location.id]
            func[cur_idx] = max(0, func[cur_idx] + delta_func)

        # ── Vehicle depot cargo in post-decision state ─────────────────────
        # Use canonical MDP extraction helper to keep policy/MDP semantics aligned.
        vehicle_status = extract_vehicle_status(vehicle, state.time, self.config)
        func_cargo_veh = vehicle_status.functional_cargo + delta_func
        depot_cargo_veh = vehicle_status.depot_cargo + delta_depot_cargo
        K = max(int(vehicle_status.capacity), 1)

        # ── Time-indexed target inventory (N,) ────────────────────────────
        d, h   = state.day() % 7, state.hour() % 24
        target = self._target_matrix[d, h]              # shape (N,)

        # ── Compute dist(v, depot) for φ_5 ───────────────────────────────
        dist_to_depot = (
            state.get_travel_time(vehicle.location.id, self._depot_id)
            if self._depot_id and self._depot_id != vehicle.location.id
            else 0.0
        )

        # ── Delegate to vfa_features.extract() – the canonical feature source
        phi = _extract_phi(
            func=func.astype(np.float64),
            onsite=onsite.astype(np.float64),
            depot=depot.astype(np.float64),
            target=target.astype(np.float64),
            activity=self._activity.astype(np.float64),
            func_cargo_veh=float(func_cargo_veh),
            depot_cargo_veh=float(depot_cargo_veh),
            vehicle_capacity=K,
            dist_to_depot=dist_to_depot,
            maintenance_enabled=self.maintenance_enabled,
            shift_timing_enabled=self.shift_timing_enabled,
            time_remaining=self._get_time_remaining(state, vehicle),
            shift_length=self._get_shift_length(state, vehicle),
        )

        assert len(phi) == len(self.FEATURE_NAMES), (
            f"extract_features() returned {len(phi)} values but "
            f"FEATURE_NAMES has {len(self.FEATURE_NAMES)} entries. "
            f"Keep the return array and FEATURE_NAMES in sync."
        )

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
        phi = self.extract_features(state, vehicle, delta_func, delta_depot_cargo)
        return _phi_as_dict(phi, self.maintenance_enabled, self.shift_timing_enabled)

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

        cur_s = state.metrics.get_aggregate_value("starvation")      or 0
        cur_c = state.metrics.get_aggregate_value("long_congestion") or 0

        reward = -(
            C_STARV * (cur_s - self._prev_starvations) +
            C_CONG  * (cur_c - self._prev_congestions)
        )

        self._prev_starvations = cur_s
        self._prev_congestions = cur_c
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
            return

        td_error = reward + self.gamma * self.value(phi_next) - self.value(self._prev_phi)
        # Clip TD error to prevent weight explosion (numerical safety net).
        td_error = float(np.clip(td_error, -50.0, 50.0))
        self.theta   += self.alpha * td_error * self._prev_phi
        self.weights  = list(self.theta)   # keep the logging attribute in sync

    # ─────────────────────────────────────────────────────────────────────────
    # Action generation  (action-space splitting)
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_candidates(self, state, vehicle) -> List[sim.Action]:
        """
        Generate a tractable set of candidate actions using action-space splitting:

          Micro (inventory) – push current station toward its target state.
                              This is fixed greedily; only the routing varies.
          Macro (routing)   – enumerate the N_CANDIDATES nearest next stations
                              sorted by travel time from the current location.

        This avoids enumerating the full exponential joint action space.
        """
        # ── Micro: decide MDP-level inventory action at current station ─────
        # Convert to simulator Action at the boundary via action_bridge.
        if vehicle.is_at_depot():
            rebalancing = 0
        else:
            target    = round(vehicle.location.get_target_state(state.day(), state.hour()))
            n_station = len(vehicle.location.bikes)
            n_vehicle = len(vehicle.get_bike_inventory())
            vehicle_capacity = int(
                getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", n_vehicle))
            )
            delta     = target - n_station        # >0 → deliver,  <0 → pickup

            if delta > 0:
                rebalancing = min(n_vehicle, delta)
            elif delta < 0:
                # Only pick up undamaged bikes; respect vehicle capacity
                n = min(n_station, -delta, max(vehicle_capacity - n_vehicle, 0))
                rebalancing = -n
            else:
                rebalancing = 0

        # ── Macro: nearest N_CANDIDATES next stations (by travel time) ────
        # Include depot as a candidate destination for end-of-day planning
        cur_id = vehicle.location.id
        pool   = [s for s in state.get_stations() if s.id != cur_id]
        
        # Add depot to candidate pool if it exists and is not current location
        # (depot is excluded from get_stations(), so we add it explicitly)
        depot_stations = state.get_depots()
        if depot_stations:
            depot = depot_stations[0]  # use first/closest depot
            if depot.id != cur_id:
                pool.append(depot)
        
        # Sort by travel time and keep top N_CANDIDATES
        pool.sort(key=lambda s: state.get_travel_time(cur_id, s.id))
        pool = pool[: self.N_CANDIDATES]

        candidates = []
        for s in pool:
            mdp_action = MdpAction(
                current_station=cur_id,
                rebalancing=int(rebalancing),
                onsite_repairs=0,
                depot_removals=0,
                load_from_queue=0,
                next_station=s.id,
            )
            candidates.append(mdp_action_to_sim_action(mdp_action, state, vehicle))

        if not candidates:
            depot_id   = state.get_closest_depot(vehicle)
            mdp_action = MdpAction(
                current_station=cur_id,
                rebalancing=0,
                onsite_repairs=0,
                depot_removals=0,
                load_from_queue=0,
                next_station=depot_id,
            )
            candidates = [mdp_action_to_sim_action(mdp_action, state, vehicle)]

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

        # ── Step 3: score each candidate ──────────────────────────────────
        phis: List[np.ndarray] = []
        values = np.empty(len(candidates), dtype=np.float64)

        for k, action in enumerate(candidates):
            # Net change in functional bikes at current station
            delta_func = len(action.delivery_bikes) - len(action.pick_ups)
            phi = self.extract_features(state, vehicle, delta_func)
            phis.append(phi)
            values[k] = self.value(phi)

        # ── Step 4: TD(0) update ──────────────────────────────────────────
        # Bootstrap with the greedy (min-value) next post-decision state,
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
            self._prev_congestions = state.metrics.get_aggregate_value("long_congestion") or 0

        # ── Step 5: select action ─────────────────────────────────────────
        if self.learning_mode:
            selected, sel_idx = self._boltzmann_select(candidates, values)
        else:
            sel_idx  = int(np.argmin(values))
            selected = candidates[sel_idx]

        # ── Step 6: Log depot decisions (optional) ────────────────────────
        self._log_depot_decision(state, vehicle, selected, phis[sel_idx], values[sel_idx])

        # ── Step 7: cache post-decision features for next TD update ───────
        self._prev_phi = phis[sel_idx]

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

    def reset_episode(self) -> None:
        """
        Clear per-episode TD tracking state.

        Does NOT reset θ or the station caches — those persist across episodes.
        """
        self._prev_phi         = None
        self._prev_starvations = 0
        self._prev_congestions = 0

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
        """Load a trained model from disk (learning_mode=False by default)."""
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
