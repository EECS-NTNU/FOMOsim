"""
LinearVFAPolicy.py  –  Time-Indexed Linear Value Function Approximation

Implements (strictly no rollout / lookahead logic here):
  1. Feature extraction  φ(S^x)  from the **post-decision state** (numpy-vectorised)
  2. VFA scoring         V(S^x) = θᵀ φ(S^x)
  3. Boltzmann (softmax) action selection  P(a) ∝ exp(−V(S^x_a) / τ)
  4. TD(0) weight update:
         θ ← θ + α (r + γ V(S^x_next) − V(S^x_cur)) φ(S^x_cur)

Feature definitions live in extract_features() – that is the single
canonical source for both the maths and the implementation.
To add / remove features see the instructions above FEATURE_NAMES.
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
    FEATURE_NAMES as _FEATURE_NAMES,
    N_FEATURES    as _N_FEATURES,
    extract       as _extract_phi,
    as_dict       as _phi_as_dict,
)
from policies.sjovik_sund.mdp.mdp_formulation import (
    extract_station_inventory,
    extract_vehicle_status,
    extract_mdp_state,
)


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

    # ── Feature registry ─────────────────────────────────────────────────
    # Sourced from vfa_features.py – that is the single place to edit features.
    FEATURE_NAMES: List[str] = _FEATURE_NAMES
    N_FEATURES:    int        = _N_FEATURES

    # Number of next-station candidates evaluated per decision
    N_CANDIDATES: int = 8

    def __init__(
        self,
        n_features: int = None,   # defaults to len(FEATURE_NAMES); set explicitly to override
        alpha: float = 0.01,
        gamma: float = 0.99,
        tau: float = 5.0,
        learning_mode: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__(maintenance_enabled=True)

        if n_features is None:
            n_features = len(self.FEATURE_NAMES)

        if n_features != len(self.FEATURE_NAMES):
            raise ValueError(
                f"n_features={n_features} but FEATURE_NAMES has "
                f"{len(self.FEATURE_NAMES)} entries. "
                f"Update FEATURE_NAMES when adding or removing features."
            )

        self.n_features    = n_features
        self.alpha         = alpha
        self.gamma         = gamma
        self.tau           = tau            # Boltzmann temperature – set by training loop
        self.learning_mode = learning_mode
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
        self, state
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build (func, onsite, depot) integer arrays of shape (N,) from the
        current simulator state.

        Iterates once over all stations; subsequent feature maths is vectorised.
        """
        N      = len(self._station_ids)
        func   = np.zeros(N, dtype=np.int32)
        onsite = np.zeros(N, dtype=np.int32)
        depot  = np.zeros(N, dtype=np.int32)

        for station in state.get_stations():
            k = self._sid_to_idx[station.id]
            inv = extract_station_inventory(station)
            func[k], onsite[k], depot[k] = inv.functional, inv.onsite, inv.depot

        return func, onsite, depot

    # ─────────────────────────────────────────────────────────────────────────
    # Feature vector  φ(S^x)  –  fully vectorised after inventory extraction
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

        This is the CANONICAL mathematical definition of every feature.
        Edit the computation blocks below (and FEATURE_NAMES) to change features.

        Notation
        ────────
          I_i^func   functional bikes at station i  (post-decision)
          Î_i^func   time-indexed target inventory at station i
          I_i^onsite onsite-repairable bikes at station i
          I_i^depot  depot-level damaged bikes at station i
          λ_i        time-averaged arrival rate at station i
          q_v^depot  depot-damaged bikes currently on the vehicle
          K          vehicle capacity
          dist(v,d)  travel time from vehicle location to nearest depot

        Features
        ────────
          φ_1  rebalancing_imbalance   Σ_i |I_i^func − Î_i^func|
          φ_2  trailer_cannibalization q_v^depot / K
          φ_3  onsite_backlog          Σ_i I_i^onsite
          φ_4  demand_weighted_depot   Σ_i (I_i^depot × λ_i)
          φ_5  depot_pull              φ_2 × dist(v, depot)

        Args:
            state:             sim.State at the current decision epoch
            vehicle:           Active sim.Vehicle
            delta_func:        Change in functional bikes at current station
                               (positive = delivery, negative = pickup)
            delta_depot_cargo: Change in vehicle depot-bike cargo after action

        Returns:
            φ  np.ndarray of shape (n_features,)   dtype float64
        """
        func, onsite, depot = self._extract_inventories(state)

        # ── Apply post-decision delta at the vehicle's current station ─────
        cur_idx = self._sid_to_idx[vehicle.location.id]
        func[cur_idx] = max(0, func[cur_idx] + delta_func)

        # ── Vehicle depot cargo in post-decision state ─────────────────────
        depot_cargo_veh = (
            sum(
                1 for b in vehicle.get_bike_inventory()
                if getattr(b, "damage_status", None) == "depot"
            )
            + delta_depot_cargo
        )
        K = max(vehicle.capacity, 1)

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
            depot_cargo_veh=float(depot_cargo_veh),
            vehicle_capacity=K,
            dist_to_depot=dist_to_depot,
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
        return _phi_as_dict(phi)

    def value(self, phi: np.ndarray) -> float:
        """V(S^x) = θᵀ φ(S^x)."""
        return float(np.dot(self.theta, phi))

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
        # ── Micro: decide what to load/unload at the current station ──────
        if vehicle.is_at_depot():
            deliver: List = []
            pickup:  List = []
        else:
            target    = round(vehicle.location.get_target_state(state.day(), state.hour()))
            n_station = len(vehicle.location.bikes)
            n_vehicle = len(vehicle.get_bike_inventory())
            delta     = target - n_station        # >0 → deliver,  <0 → pickup

            if delta > 0:
                n       = min(n_vehicle, delta)
                deliver = [b.bike_id for b in vehicle.get_bike_inventory()[:n]]
                pickup  = []
            elif delta < 0:
                # Only pick up undamaged bikes; respect vehicle capacity
                n       = min(n_station, -delta, vehicle.capacity - n_vehicle)
                deliver = []
                pickup  = [
                    b.bike_id for b in vehicle.location.bikes[:n]
                    if getattr(b, "damage_status", None) is None
                ]
            else:
                deliver, pickup = [], []

        # ── Macro: nearest N_CANDIDATES next stations (by travel time) ────
        cur_id = vehicle.location.id
        pool   = [s for s in state.get_stations() if s.id != cur_id]
        pool.sort(key=lambda s: state.get_travel_time(cur_id, s.id))
        pool   = pool[: self.N_CANDIDATES]

        # sim.Action(battery_swaps, pick_ups, delivery_bikes, next_location)
        candidates = [sim.Action([], pickup, deliver, s.id) for s in pool]

        if not candidates:
            depot_id   = state.get_closest_depot(vehicle)
            candidates = [sim.Action([], [], [], depot_id)]

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

        # ── Step 5: select action ─────────────────────────────────────────
        if self.learning_mode:
            selected, sel_idx = self._boltzmann_select(candidates, values)
        else:
            sel_idx  = int(np.argmin(values))
            selected = candidates[sel_idx]

        # ── Step 6: cache post-decision features for next TD update ───────
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
        print(f"  [VFA] θ saved → {path}")

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
        return f"EpisodeTrainingPolicy(τ={self.vfa_policy.tau:.4f})"
