"""
HybridRolloutPolicy.py — Analytical Rollout Policy

Replaces the simulation-based rollout with a fast analytical approximation:
  - Demand:      Poisson sampling from station-level leave/arrive intensity profiles
  - Degradation: Weibull-based failure sampling (optional, off by default)
  - Tail value:  evaluated via the frozen LinearVFA at the projected terminal state

Decision flow:
  1. Generate all candidates via candidate_generator
  2. For each candidate, compute post-action inventories
  3. Roll each candidate forward over `lookahead_minutes` with `num_scenarios` Monte Carlo draws
  4. Score = mean(accumulated_reward + discounted_tail_VFA)
  5. Return the highest-scoring candidate
  No VFA pruning, no screening.
"""
from __future__ import annotations

import time
import numpy as np
import sim
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, cast
from helpers import format_sim_time

from policies.policy import Policy
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
from policies.sjovik_sund.mdp.candidate_generator import generate_candidates
from sim.bike_degradation_modeling.damage_configuration import DAMAGE_CATEGORIES
from sim.bike_degradation_modeling.bike_component_degradation_model import ComponentFailureModel
from settings import MAINTENANCE_REPAIR, MINUTES_CONSTANT_PER_ACTION, MINUTES_PER_ACTION

if TYPE_CHECKING:
    from policies.sjovik_sund.simulation_logging import SimulationRunLogger as RunLogger


@dataclass(frozen=True)
class _ActionEffect:
    functional_pickups: int
    functional_deliveries: int
    depot_pickups: int
    depot_dropoffs: int
    onsite_repairs: int
    load_from_queue: int
    station_delta_func: int
    station_delta_depot: int
    station_delta_onsite: int
    delta_func_cargo: int
    delta_depot_cargo: int


class HybridRolloutPolicy(Policy):
    """
    Analytical rollout policy backed by a frozen LinearVFAPolicy.

    Parameters
    ----------
    trained_vfa           : frozen VFA used for terminal-state scoring
    lookahead_minutes     : rollout horizon H (minutes)
    num_scenarios         : Monte Carlo draws per candidate (CRN across candidates)
    n_routing_candidates  : number of routing destinations passed to candidate_generator
    n_time_steps          : number of equal-width sub-intervals in [0, H]
    use_degradation       : whether to sample Weibull component failures during rollout
    avg_km_per_trip       : average trip distance (km) — used for degradation estimation
    logger                : optional RunLogger for structured CSV output
    debug_print           : print ranked candidate table at each decision
    NOTE: Degradation estimation is off by default - not included in thesis
    """

    def __init__(
        self,
        trained_vfa: LinearVFAPolicy,
        lookahead_minutes: float = 60.0,
        num_scenarios: int = 8,
        n_routing_candidates: int = 10,
        n_time_steps: int = 12,
        use_degradation: bool = False,
        avg_km_per_trip: float = 2.5,
        logger: Optional[RunLogger] = None,
        debug_print: bool = True,
    ):
        super().__init__(maintenance_enabled=trained_vfa.maintenance_enabled)
        self.vfa = trained_vfa
        self._sync_maintenance_flag()
        self.vfa.learning_mode = False  # strictly no weight updates
        self.lookahead_minutes    = lookahead_minutes
        self.num_scenarios        = num_scenarios
        self.n_routing_candidates = n_routing_candidates
        self.n_time_steps         = n_time_steps
        self.use_degradation      = use_degradation
        self.avg_km_per_trip      = avg_km_per_trip
        self.logger: Optional[RunLogger] = logger
        self.debug_print = debug_print
        self.weights     = getattr(trained_vfa, "weights", [])
        self._simulator  = None

        # Pre-compute fleet-average Weibull hazard rate per km for degradation sampling
        self._avg_failure_rate_per_km: float = self._compute_avg_failure_rate()

    # ─────────────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────────────

    def _sync_maintenance_flag(self) -> None:
        """Keep Hybrid policy and wrapped VFA on the same maintenance setting."""
        self.vfa.maintenance_enabled = bool(self.maintenance_enabled)

    def _compute_avg_failure_rate(self) -> float:
        """Fleet-average hazard rate per km, evaluated at each component's MTTF/2."""
        rates = [
            ComponentFailureModel.calculate_hazard_rate(
                params["mttf_km"] / 2.0, params["scale"], params["shape"]
            )
            for params in DAMAGE_CATEGORIES.values()
        ]
        return float(np.mean(rates))

    def __deepcopy__(self, memo):
        return self

    def init_sim(self, simulator) -> None:
        self._simulator = simulator
        self._sync_maintenance_flag()
        self.vfa.init_sim(simulator)

    # ─────────────────────────────────────────────────────────────────────────
    # Analytical rollout core
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _bike_id(bike) -> object:
        return getattr(bike, "bike_id", getattr(bike, "id", bike))

    def _classify_action_effect(self, vehicle, action) -> _ActionEffect:
        """Classify a simulator Action into station and cargo count effects."""
        raw_station_bikes = getattr(vehicle.location, "bikes", {})
        station_bikes = (
            raw_station_bikes if isinstance(raw_station_bikes, dict)
            else {self._bike_id(b): b for b in raw_station_bikes}
        )
        vehicle_bikes = {self._bike_id(b): b for b in vehicle.get_bike_inventory()}
        fixed_queue = getattr(vehicle.location, "fixed_queue", {})

        functional_pickups = depot_pickups = load_from_queue = 0
        for b_id in getattr(action, "pick_ups", []):
            if vehicle.is_at_depot() and b_id in fixed_queue:
                load_from_queue += 1
                continue
            b = station_bikes.get(b_id)
            if b and getattr(b, "damage_status", None) == "depot":
                depot_pickups += 1
            elif b:
                functional_pickups += 1

        functional_deliveries = depot_dropoffs = 0
        for b_id in getattr(action, "delivery_bikes", []):
            b = vehicle_bikes.get(b_id)
            if vehicle.is_at_depot() and b and getattr(b, "damage_status", None) == "depot":
                depot_dropoffs += 1
            else:
                functional_deliveries += 1

        onsite_repairs = len(getattr(action, "onsite_repairs", []))
        station_delta_func = functional_deliveries - functional_pickups + onsite_repairs
        station_delta_depot = -depot_pickups
        station_delta_onsite = -onsite_repairs
        delta_func_cargo = functional_pickups + load_from_queue - functional_deliveries
        delta_depot_cargo = depot_pickups - depot_dropoffs

        return _ActionEffect(
            functional_pickups=functional_pickups,
            functional_deliveries=functional_deliveries,
            depot_pickups=depot_pickups,
            depot_dropoffs=depot_dropoffs,
            onsite_repairs=onsite_repairs,
            load_from_queue=load_from_queue,
            station_delta_func=station_delta_func,
            station_delta_depot=station_delta_depot,
            station_delta_onsite=station_delta_onsite,
            delta_func_cargo=delta_func_cargo,
            delta_depot_cargo=delta_depot_cargo,
        )

    def _apply_action_delta(
        self,
        base_func: np.ndarray,
        base_onsite: np.ndarray,
        base_depot: np.ndarray,
        vehicle,
        action,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, _ActionEffect]:
        """
        Return station arrays after the current action plus classified count effects.
        """
        func   = base_func.copy().astype(np.float64)
        onsite = base_onsite.copy().astype(np.float64)
        depot  = base_depot.copy().astype(np.float64)

        effect = self._classify_action_effect(vehicle, action)

        vfa = self.vfa
        if not vehicle.is_at_depot() and vehicle.location.id in vfa._sid_to_idx:
            idx = vfa._sid_to_idx[vehicle.location.id]
            func[idx] = max(
                0.0,
                func[idx] + effect.station_delta_func - effect.onsite_repairs, #Note: confusing logic but correct - substracts onsite to avoid double counting with next line
            )
            if effect.onsite_repairs:
                applied    = min(effect.onsite_repairs, int(onsite[idx]))
                onsite[idx] = max(0.0, onsite[idx] - applied)
                func[idx]  += float(applied)
            if effect.depot_pickups:
                depot[idx] = max(0.0, depot[idx] + effect.station_delta_depot)

        return func, onsite, depot, effect

    def _current_operation_time(self, effect: _ActionEffect) -> float:
        """Current-location work time, excluding travel to the next station."""
        handling_count = (
            effect.functional_pickups
            + effect.functional_deliveries
            + effect.depot_pickups
            + effect.depot_dropoffs
            + effect.load_from_queue
        )
        return (
            handling_count * MINUTES_PER_ACTION
            + effect.onsite_repairs * MAINTENANCE_REPAIR
            + MINUTES_CONSTANT_PER_ACTION #NOTE: This is set to zero in settings and thus not included even though calculated. Can be removed.
        )

    def _action_arrival_time(self, state, vehicle, action, dest_id: Optional[str]) -> float:
        """Absolute arrival time after current station work, fixed overhead, and travel."""
        if not dest_id:
            return float("inf")
        try:
            travel_time = float(state.get_vehicle_travel_time(vehicle.location.id, dest_id))
        except Exception:
            travel_time = float("inf")
        if not np.isfinite(travel_time):
            return float("inf")
        effect = self._classify_action_effect(vehicle, action)
        return float(state.time) + self._current_operation_time(effect) + travel_time

    def _is_depot_id(self, state, loc_id: Optional[str]) -> bool:
        if loc_id is None:
            return False
        try:
            return any(d.id == loc_id for d in state.get_depots())
        except Exception:
            return str(loc_id).startswith("D")

    def _base_operation_profiles(
        self,
        dest_idx: int,
        eval_time: float,
        func: np.ndarray,
        onsite: np.ndarray,
        depot: np.ndarray,
        func_cargo: int,
        depot_cargo: int,
        vehicle_capacity: int,
    ) -> list[tuple[int, int, int]]:
        """Generate compact synthetic operation profiles at the rollout destination."""
        vfa = self.vfa
        assert vfa._target_matrix is not None
        assert vfa._capacities is not None

        eval_day = int(eval_time // (24 * 60)) % 7
        eval_hour = int((eval_time // 60) % 24)
        cur_func = int(max(0, round(float(func[dest_idx]))))
        cur_onsite = int(max(0, round(float(onsite[dest_idx]))))
        cur_depot = int(max(0, round(float(depot[dest_idx]))))
        target = int(round(float(vfa._target_matrix[eval_day, eval_hour, dest_idx])))
        station_spare_cap = max(
            0,
            int(vfa._capacities[dest_idx]) - cur_func - cur_onsite - cur_depot,
        )
        free_cap = max(0, vehicle_capacity - func_cargo - depot_cargo)

        # Match candidate_generator._rebalancing_options_target_centered():
        # exact-to-target plus operations toward 75% and 125% of target.
        rebalancing_options = {0}
        delta = cur_func - target
        if delta > 0:
            rebalancing_options.add(-min(delta, free_cap))
        elif delta < 0:
            rebalancing_options.add(min(-delta, func_cargo))

        target_plus = round(target * 1.25)
        delta_plus = cur_func - target_plus
        if delta_plus < 0:
            rebalancing_options.add(min(-delta_plus, func_cargo, station_spare_cap))

        target_minus = round(target * 0.75)
        delta_minus = cur_func - target_minus
        if delta_minus > 0:
            rebalancing_options.add(-min(delta_minus, free_cap))

        fractions = (0.0, 0.25, 0.50, 0.75, 1.0)
        onsite_options = {round(cur_onsite * f) for f in fractions} if cur_onsite > 0 else {0}
        depot_options = {round(cur_depot * f) for f in fractions} if cur_depot > 0 else {0}

        profiles: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        for reb in rebalancing_options:
            if reb > 0 and reb > min(func_cargo, station_spare_cap):
                continue
            if reb < 0 and -reb > min(cur_func, free_cap):
                continue
            space_after_reb = free_cap + max(0, reb) + min(0, reb)
            for repairs in onsite_options:
                applied_repairs = min(int(repairs), cur_onsite)
                for removals in depot_options:
                    valid_removals = min(int(removals), cur_depot, max(0, space_after_reb))
                    profile = (int(reb), int(applied_repairs), int(valid_removals))
                    if profile not in seen:
                        seen.add(profile)
                        profiles.append(profile)
        return profiles

    def _base_routing_options(
        self,
        state,
        vehicle,
        current_idx: int,
        eval_time: float,
        func: np.ndarray,
        onsite: np.ndarray,
        depot: np.ndarray,
        func_cargo: int,
        depot_cargo: int,
        vehicle_capacity: int,
        n_options: int = 5,
    ) -> list[str]:
        """Fast array-based routing options for the synthetic VFA base step."""
        vfa = self.vfa
        assert vfa._target_matrix is not None
        assert vfa._leave_profile is not None
        assert vfa._arrive_profile is not None
        assert vfa._travel_time_matrix is not None

        eval_day = int(eval_time // (24 * 60)) % 7
        eval_hour = int((eval_time // 60) % 24)
        day_type = 1 if eval_day in (5, 6) else 0
        target = vfa._target_matrix[eval_day, eval_hour]
        delta = func - target
        net_demand = vfa._arrive_profile[day_type, eval_hour] - vfa._leave_profile[day_type, eval_hour]

        free_space = max(0, vehicle_capacity - func_cargo - depot_cargo)
        free_ratio = free_space / max(1, vehicle_capacity)
        can_deliver = (delta < 0) & (func_cargo > 0)
        can_pickup = (delta > 0) & (free_space > 0)

        score = np.zeros_like(delta, dtype=np.float64)
        if free_ratio <= 0.2:
            score[can_deliver] = np.abs(delta[can_deliver])
        elif free_ratio >= 0.8:
            score[can_pickup] = delta[can_pickup]
        else:
            mask = can_deliver | can_pickup
            score[mask] = np.abs(delta[mask])

        ttv = np.full_like(score, 8.0, dtype=np.float64)
        rising = net_demand > 0
        falling = net_demand < 0
        capacities = vfa._capacities if vfa._capacities is not None else np.maximum(func, 1.0)
        free_docks = np.maximum(0.0, capacities - func - onsite - depot)
        ttv[rising] = free_docks[rising] / np.maximum(net_demand[rising], 1e-6)
        ttv[falling] = np.maximum(0.0, func[falling]) / np.maximum(-net_demand[falling], 1e-6)
        urgency = 1.0 - np.minimum(ttv, 8.0) / 8.0

        worsens = ((delta < 0) & (net_demand < 0)) | ((delta > 0) & (net_demand > 0))
        corrects = ((delta < 0) & (net_demand > 0)) | ((delta > 0) & (net_demand < 0))
        active = score > 0
        score[active & worsens] *= 1.0 + urgency[active & worsens]
        score[active & corrects] *= np.maximum(0.1, 1.0 - 0.5 * urgency[active & corrects])

        if self.maintenance_enabled:
            score += 1.5 * (onsite + depot)

        travel_times = vfa._travel_time_matrix[current_idx]
        distance_mask = (score > 0) & (travel_times > 0)
        score[distance_mask] /= np.power(travel_times[distance_mask], 0.35)
        score[current_idx] = -np.inf

        try:
            tabu = {v.location.id for v in state.get_vehicles() if getattr(v, "id", None) != getattr(vehicle, "id", None)}
            for sid in tabu:
                if sid in vfa._sid_to_idx:
                    score[vfa._sid_to_idx[sid]] = -np.inf
        except Exception:
            pass

        n = min(max(1, n_options), len(score))
        top = np.argsort(score)[::-1][:n]
        return [vfa._station_ids[i] for i in top if np.isfinite(score[i])]

    def _apply_vfa_base_step(
        self,
        state,
        vehicle,
        dest_id: Optional[str],
        eval_time: float,
        func_proj: np.ndarray,
        onsite_proj: np.ndarray,
        depot_proj: np.ndarray,
        func_cargo: int,
        depot_cargo: int,
        vehicle_capacity: int,
        health_base: Optional[dict] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, Optional[str], int, int]:
        """
        Apply one event-lite VFA base-policy operation at the reached destination.

        The base step chooses a synthetic local operation and a next station by
        VFA value, applies only the local operation to the rollout arrays, and
        returns the chosen next station for terminal-value conditioning.
        """
        vfa = self.vfa
        if dest_id is None:
            self._record_base_step_debug(eval_time, dest_id, None, None, func_cargo, depot_cargo, func_cargo, depot_cargo, "no-destination")
            return func_proj, onsite_proj, depot_proj, func_cargo, depot_cargo, None, 0, 0

        if dest_id not in vfa._sid_to_idx:
            if self._is_depot_id(state, dest_id):
                self._record_base_step_debug(eval_time, dest_id, (0, 0, 0), None, func_cargo, depot_cargo, func_cargo, 0, "depot-unload")
                return func_proj, onsite_proj, depot_proj, func_cargo, 0, None, 0, depot_cargo
            self._record_base_step_debug(eval_time, dest_id, None, None, func_cargo, depot_cargo, func_cargo, depot_cargo, "non-station")
            return func_proj, onsite_proj, depot_proj, func_cargo, depot_cargo, None, 0, 0

        dest_idx = vfa._sid_to_idx[dest_id]
        func_cargo_before = func_cargo
        depot_cargo_before = depot_cargo
        profiles = self._base_operation_profiles(
            dest_idx, eval_time, func_proj, onsite_proj, depot_proj,
            func_cargo, depot_cargo, vehicle_capacity,
        )

        best_value = -float("inf")
        best_profile = (0, 0, 0)
        best_next: Optional[str] = None

        for reb, repairs, removals in profiles:
            cand_func = func_proj.copy()
            cand_onsite = onsite_proj.copy()
            cand_depot = depot_proj.copy()

            cand_func[dest_idx] = max(0.0, cand_func[dest_idx] + reb)
            if repairs:
                applied_repairs = min(repairs, int(cand_onsite[dest_idx]))
                cand_onsite[dest_idx] = max(0.0, cand_onsite[dest_idx] - applied_repairs)
                cand_func[dest_idx] += applied_repairs
            if removals:
                cand_depot[dest_idx] = max(0.0, cand_depot[dest_idx] - removals)

            cand_func_cargo = int(max(0, func_cargo - reb))
            cand_depot_cargo = int(max(0, depot_cargo + removals))
            next_options = self._base_routing_options(
                state, vehicle, dest_idx, eval_time,
                cand_func, cand_onsite, cand_depot,
                cand_func_cargo, cand_depot_cargo, vehicle_capacity,
                n_options=min(5, self.n_routing_candidates), #Required at lest 5 routing options
            ) or [None]

            for next_id in next_options:
                phi = vfa.extract_features(
                    state, vehicle,
                    cand_func, cand_onsite, cand_depot,
                    delta_func=0,
                    delta_depot_cargo=0,
                    delta_onsite_repairs=0,
                    eval_time=eval_time,
                    next_station_id=next_id,
                    explicit_vehicle_loc_id=dest_id,
                    explicit_functional_cargo=cand_func_cargo,
                    explicit_depot_cargo=cand_depot_cargo,
                    explicit_capacity=vehicle_capacity,
                    health_base=health_base,
                )
                value = float(vfa.value(phi))
                if value > best_value:
                    best_value = value
                    best_profile = (reb, repairs, removals)
                    best_next = next_id

        reb, repairs, removals = best_profile
        func_proj = func_proj.copy()
        onsite_proj = onsite_proj.copy()
        depot_proj = depot_proj.copy()
        func_proj[dest_idx] = max(0.0, func_proj[dest_idx] + reb)
        if repairs:
            applied_repairs = min(repairs, int(onsite_proj[dest_idx]))
            onsite_proj[dest_idx] = max(0.0, onsite_proj[dest_idx] - applied_repairs)
            func_proj[dest_idx] += applied_repairs
        if removals:
            depot_proj[dest_idx] = max(0.0, depot_proj[dest_idx] - removals)
        func_cargo = int(max(0, func_cargo - reb))
        depot_cargo = int(max(0, depot_cargo + removals))

        self._record_base_step_debug(
            eval_time, dest_id, best_profile, best_next,
            func_cargo_before, depot_cargo_before, func_cargo, depot_cargo,
            f"vfa={best_value:.4f}",
        )
        return func_proj, onsite_proj, depot_proj, func_cargo, depot_cargo, best_next, 0, 0

    def _record_base_step_debug(
        self,
        eval_time: float,
        dest_id: Optional[str],
        profile: Optional[tuple[int, int, int]],
        next_id: Optional[str],
        func_before: int,
        depot_before: int,
        func_after: int,
        depot_after: int,
        note: str,
    ) -> None:
        """Collect compact debug evidence that event-lite base steps are firing."""
        self._debug_base_steps = getattr(self, "_debug_base_steps", 0) + 1
        samples = getattr(self, "_debug_base_step_samples", None)
        if samples is None:
            samples = []
            self._debug_base_step_samples = samples
        if len(samples) < 8:
            samples.append({
                "t": round(float(eval_time), 1),
                "dest": dest_id,
                "profile": profile,
                "next": next_id,
                "cargo_before": (int(func_before), int(depot_before)),
                "cargo_after": (int(func_after), int(depot_after)),
                "note": note,
            })

    def _run_analytical_scenario(
        self,
        state,
        vehicle,
        func: np.ndarray,
        onsite: np.ndarray,
        depot: np.ndarray,
        rng: np.random.Generator,
        dest_id: Optional[str] = None,
        func_cargo: int = 0,
        depot_cargo: int = 0,
        vehicle_capacity: int = 0,
        dest_arrival_min: float = float("inf"),
        health_base: Optional[dict] = None,
        depot_fixed_queue_delta: int = 0,
        depot_in_repair_delta: int = 0,
    ) -> tuple[float, float]:
        """
        Project station inventories forward analytically over `lookahead_minutes`.

        At the step when the vehicle arrives at `dest_id` (dest_arrival_min), one
        event-lite VFA base-policy operation is applied at the destination. This
        keeps the fast analytical rollout, but uses the frozen VFA as the base
        policy and terminal-value estimator.

        Returns
        -------
        accumulated_reward : discounted sum of step-level starvation/congestion penalties
        discounted_tail    : γ^H × V(terminal_state) from the frozen VFA
        """
        vfa = self.vfa
        assert vfa._capacities    is not None
        assert vfa._leave_profile  is not None
        assert vfa._arrive_profile is not None

        func_proj   = func.copy()
        onsite_proj = onsite.copy()
        depot_proj  = depot.copy()

        cfg    = vfa.reward_calc.config
        gamma  = vfa.gamma
        sf     = float(getattr(vfa.reward_calc, "_scale_factor", 1.0))
        dt     = self.lookahead_minutes / self.n_time_steps   # minutes per time step
        t0     = float(state.time)
        accumulated_reward = 0.0

        dest_service_done = False   # applied at most once per scenario
        terminal_next_station_id: Optional[str] = dest_id

        for step in range(self.n_time_steps):
            step_start = t0 + step * dt
            step_end   = step_start + dt

            # Blend demand rates across hour boundaries within the step interval.
            # Mirrors the minutes_current_hour / minutes_next_hour weighting in
            # SB_BS_PILOT.generate_scenarioes(), fixing intra-step hour transitions.
            next_boundary = (int(step_start // 60) + 1) * 60.0  # next whole hour in sim-time

            if next_boundary >= step_end:
                # Entire step within one calendar hour — no blending needed
                h0       = int((step_start // 60) % 24)
                day0     = int(step_start // (24 * 60)) % 7
                dt_type  = 1 if day0 in [5, 6] else 0
                mean_leave  = vfa._leave_profile[dt_type, h0]  * (dt / 60.0)
                mean_arrive = vfa._arrive_profile[dt_type, h0] * (dt / 60.0)
            else:
                # Step spans a calendar hour boundary — blend proportionally
                dt_h0 = next_boundary - step_start   # minutes in current hour
                dt_h1 = step_end - next_boundary      # minutes in next hour

                h0   = int((step_start // 60) % 24)
                day0 = int(step_start // (24 * 60)) % 7
                h1   = (h0 + 1) % 24
                day1 = day0 if h1 != 0 else (day0 + 1) % 7  # day wraps at midnight

                dt0_type = 1 if day0 in [5, 6] else 0
                dt1_type = 1 if day1 in [5, 6] else 0

                mean_leave  = (vfa._leave_profile[dt0_type, h0]  * (dt_h0 / 60.0)
                             + vfa._leave_profile[dt1_type, h1]  * (dt_h1 / 60.0))
                mean_arrive = (vfa._arrive_profile[dt0_type, h0] * (dt_h0 / 60.0)
                             + vfa._arrive_profile[dt1_type, h1] * (dt_h1 / 60.0))

            # ── Event-lite VFA base-policy injection ──────────────────────────
            if (not dest_service_done and step_start >= dest_arrival_min):
                (
                    func_proj,
                    onsite_proj,
                    depot_proj,
                    func_cargo,
                    depot_cargo,
                    terminal_next_station_id,
                    base_fixed_queue_delta,
                    base_in_repair_delta,
                ) = self._apply_vfa_base_step(
                    state, vehicle, dest_id, step_start,
                    func_proj, onsite_proj, depot_proj,
                    func_cargo, depot_cargo, vehicle_capacity,
                    health_base=health_base,
                )
                depot_fixed_queue_delta += base_fixed_queue_delta
                depot_in_repair_delta += base_in_repair_delta
                dest_service_done = True

            # ── Poisson demand sampling ───────────────────────────────────────
            # mean_leave / mean_arrive are already in expected-trips-per-step units
            n_depart = rng.poisson(mean_leave)   # shape (N,)
            n_arrive = rng.poisson(mean_arrive)  # shape (N,)

            available      = np.maximum(func_proj, 0.0).astype(int)
            actual_depart  = np.minimum(n_depart, available)
            starvations    = int(np.sum(n_depart - actual_depart))   # unmet departures

            func_proj -= actual_depart

            free_docks = np.maximum(0.0, vfa._capacities - func_proj - onsite_proj - depot_proj).astype(int)
            actual_arrive = np.minimum(n_arrive, free_docks)
            congestions = int(np.sum(n_arrive - actual_arrive))
            func_proj += actual_arrive

            # ── Weibull degradation (optional) ────────────────────────────────
            if self.use_degradation:
                sys_bikes      = max(float(np.sum(func_proj)), 1.0)
                # mean_leave is already in expected-trips-per-step units
                trips_per_bike = float(np.sum(mean_leave)) / sys_bikes
                km_per_bike    = trips_per_bike * self.avg_km_per_trip
                fail_prob      = min(self._avg_failure_rate_per_km * km_per_bike, 0.5)

                n_func_int   = np.maximum(func_proj, 0.0).astype(int)
                n_new_broken = rng.binomial(n_func_int, fail_prob)    # (N,)
                n_to_depot   = rng.binomial(n_new_broken, 0.5)        # ~50% depot
                n_to_onsite  = n_new_broken - n_to_depot
                func_proj   -= n_new_broken
                func_proj    = np.maximum(func_proj, 0.0)
                depot_proj  += n_to_depot
                onsite_proj += n_to_onsite

            # ── Reward accumulation ───────────────────────────────────────────
            step_reward = (
                cfg.weight_starvation  * starvations
                + cfg.weight_congestion * congestions
            ) * sf
            # Step rewards summarize events over [step_start, step_end); use the
            # interval midpoint as a simple quadrature convention for discounting.
            time_elapsed = (step_start + 0.5 * dt) - t0
            discount = gamma ** max(time_elapsed / 60.0, 0.0)
            accumulated_reward += discount * step_reward

        # ── Terminal VFA evaluation ───────────────────────────────────────────
        terminal_loc_id = dest_id if dest_id is not None else vehicle.location.id

        terminal_phi = vfa.extract_features(
            state, vehicle,
            func_proj, onsite_proj, depot_proj,
            delta_func=0,
            delta_depot_cargo=0,
            delta_onsite_repairs=0,
            delta_depot_fixed_queue=depot_fixed_queue_delta,
            delta_depot_in_repair=depot_in_repair_delta,
            eval_time=t0 + self.lookahead_minutes,
            next_station_id=terminal_next_station_id,
            explicit_vehicle_loc_id=terminal_loc_id,
            explicit_functional_cargo=func_cargo,
            explicit_depot_cargo=depot_cargo,
            explicit_capacity=vehicle_capacity,
            health_base=health_base,
        )
        terminal_value  = vfa.value(terminal_phi)
        discounted_tail = (gamma ** (self.lookahead_minutes / 60.0)) * terminal_value

        return accumulated_reward, discounted_tail

    # ─────────────────────────────────────────────────────────────────────────
    # Main decision entry point
    # ─────────────────────────────────────────────────────────────────────────

    def get_best_action(self, state, vehicle) -> Optional[sim.Action]:
        _t_start = time.perf_counter()

        self._sync_maintenance_flag()

        if not self.vfa._initialized:
            self.vfa._lazy_init(state)

        # ── 1. Generate candidates ────────────────────────────────────────────
        need_candidate_metadata = self.debug_print or self.logger is not None
        if need_candidate_metadata:
            all_candidates, candidate_metadata = generate_candidates(
                state, vehicle, self.maintenance_enabled,
                n_routing=self.n_routing_candidates, return_metadata=True,
            )
            action_to_meta: Dict = {id(a): candidate_metadata[i] for i, a in enumerate(all_candidates)}
        else:
            all_candidates = cast(List[sim.Action], generate_candidates(
                state, vehicle, self.maintenance_enabled,
                n_routing=self.n_routing_candidates,
            ))
            candidate_metadata = []
            action_to_meta = {}

        n_total_candidates = len(all_candidates)

        # ── 2. Snapshot current inventories ──────────────────────────────────
        base_func, base_onsite, base_depot = self.vfa._extract_inventories(state, vehicle)
        health_base = (
            self.vfa._compute_health_base(state, vehicle)
            if getattr(self.vfa, "_use_health_features", False)
            else None
        )

        # ── 3. Mute VFA logging during rollout ────────────────────────────────
        old_log_rl     = getattr(self.vfa, "log_rl_decisions", False)
        old_log_depot  = getattr(self.vfa, "log_depot_visits", False)
        old_vfa_logger = getattr(self.vfa, "logger", None)
        self.vfa.log_rl_decisions = False
        self.vfa.log_depot_visits = False
        self.vfa.logger = None

        # ── 4. Common Random Numbers: one seed per scenario ───────────────────
        crn_seeds = [int(state.rng.integers(1_000_000)) for _ in range(self.num_scenarios)]
        self._debug_base_steps = 0
        self._debug_base_step_samples = []

        # ── 5. Evaluate ALL candidates via analytical rollout ─────────────────
        # Pre-compute vehicle state once — cargo before any action this epoch.
        inv            = vehicle.get_bike_inventory()
        veh_func_cargo = sum(1 for b in inv if getattr(b, "damage_status", None)
                             not in ("depot", "onsite"))
        veh_depot_cargo = sum(1 for b in inv if getattr(b, "damage_status", None) == "depot")
        veh_capacity    = int(getattr(vehicle, "bike_inventory_capacity",
                                      getattr(vehicle, "capacity", len(inv))))

        best_action: Optional[sim.Action] = None
        best_q       = -float("inf")
        best_r       = 0.0
        best_t       = 0.0
        winning_rank = -1
        results: List[tuple] = []  # (q, r, t, action, orig_idx)

        for orig_idx, action in enumerate(all_candidates):
            dest_id = getattr(action, "next_location", getattr(action, "next_station", None))
            func_post, onsite_post, depot_post, effect = \
                self._apply_action_delta(base_func, base_onsite, base_depot, vehicle, action)

            # Immediate VFA score for the post-action state (used for debug ordering)
            try:
                immediate_phi = self.vfa.extract_features(
                    state, vehicle,
                    func_post, onsite_post, depot_post,
                    delta_func=0,
                    delta_depot_cargo=0,
                    delta_onsite_repairs=0,
                    delta_depot_fixed_queue=-effect.load_from_queue,
                    delta_depot_in_repair=effect.depot_dropoffs,
                    eval_time=float(state.time),
                    next_station_id=dest_id,
                    candidate_action=action,
                    health_base=health_base,
                )
                immediate_vfa = float(self.vfa.value(immediate_phi))
            except Exception as e:
                print(f"DEBUG: VFA Exception: {e}")
                immediate_vfa = float("-inf")

            # Vehicle cargo after current-station operations
            func_cargo_post  = max(
                0,
                veh_func_cargo + effect.delta_func_cargo,
            )
            depot_cargo_post = max(0, veh_depot_cargo + effect.delta_depot_cargo)

            # Absolute sim-time when vehicle reaches dest_id
            dest_arrival_min = self._action_arrival_time(state, vehicle, action, dest_id)

            q_sum = r_sum = t_sum = 0.0
            for seed in crn_seeds:
                rng = np.random.default_rng(seed)
                r, t = self._run_analytical_scenario(
                    state, vehicle, func_post, onsite_post, depot_post, rng,
                    dest_id=dest_id,
                    func_cargo=func_cargo_post,
                    depot_cargo=depot_cargo_post,
                    vehicle_capacity=veh_capacity,
                    dest_arrival_min=dest_arrival_min,
                    health_base=health_base,
                    depot_fixed_queue_delta=-effect.load_from_queue,
                    depot_in_repair_delta=effect.depot_dropoffs,
                )
                q_sum += r + t
                r_sum += r
                t_sum += t

            n          = self.num_scenarios
            expected_q = q_sum / n
            results.append((immediate_vfa, expected_q, r_sum / n, t_sum / n, action, orig_idx))

            if expected_q > best_q:
                best_q       = expected_q
                best_action  = action
                best_r       = r_sum / n
                best_t       = t_sum / n
                winning_rank = orig_idx

        vfa_top1 = None
        if results:
            vfa_top1 = max(results, key=lambda x: x[0])[4]
        rollout_changed = bool(best_action is not None and vfa_top1 is not None and best_action is not vfa_top1)

        # ── 6. Restore VFA logging ────────────────────────────────────────────
        self.vfa.log_rl_decisions = old_log_rl
        self.vfa.log_depot_visits = old_log_depot
        self.vfa.logger = old_vfa_logger

        # ── 7. Debug candidate table ──────────────────────────────────────────
        if self.debug_print:
            # Order the debug table by the immediate VFA ranking (immediate_vfa stored at index 0)
            ranked = sorted(results, key=lambda x: x[0], reverse=True)
            print(f"\n{'='*72}")
            print(
                f"[ANALYTICAL ROLLOUT] {n_total_candidates} candidates | "
                f"{format_sim_time(state.time)} | {vehicle.id} @ {vehicle.location.id} | "
                f"H={self.lookahead_minutes:.0f}min  S={self.num_scenarios}"
            )
            print(
                f"  CRN seeds generated: {len(crn_seeds)} "
                f"| candidate-scenarios: {n_total_candidates * self.num_scenarios} "
                f"| event-lite VFA base steps: {getattr(self, '_debug_base_steps', 0)}"
            )
            if crn_seeds:
                print(f"  CRN sample: {crn_seeds[:min(5, len(crn_seeds))]}")
            for sample in getattr(self, "_debug_base_step_samples", []):
                print(
                    "  [BASE STEP] "
                    f"t={sample['t']} dest={sample['dest']} "
                    f"profile={sample['profile']} next={sample['next']} "
                    f"cargo={sample['cargo_before']}→{sample['cargo_after']} "
                    f"{sample['note']}"
                )
            # Show immediate VFA, then rollout expected Q, R, T. Arrow points to final chosen action (best_action).
            print(f"  {'Rk':<4} {'Profile':<22} {'→ Station':<12} {'VFA':>10} {'E[Q]':>10} {'E[R]':>10} {'E[T]':>10}")
            print(f"  {'-'*80}")
            for rank, (immediate_vfa, q, r, t, a, _) in enumerate(ranked[:100], 1):
                dest = getattr(a, "next_location", getattr(a, "next_station", "?"))
                meta = action_to_meta.get(id(a), {})
                prof = meta.get("profile_type", "?")
                mark = "  ◄" if (a is best_action) else ""
                print(f"  {rank:<4} {prof:<22} {str(dest):<12} {immediate_vfa:>10.4f} {q:>10.4f} {r:>10.4f} {t:>10.4f}{mark}")
            print()

        decision_runtime_s = time.perf_counter() - _t_start

        # ── 8. RunLogger ──────────────────────────────────────────────────────
        if self.logger is not None and best_action is not None:
            winning_meta = action_to_meta.get(id(best_action), {})
            self._log_decision(
                state, vehicle, best_action,
                best_r, best_t, best_q,
                n_total_candidates=n_total_candidates,
                winning_rank=winning_rank,
                vfa_top1_next_station=getattr(vfa_top1, "next_location", None) if vfa_top1 is not None else "",
                rollout_changed=rollout_changed,
                decision_runtime_s=decision_runtime_s,
                winning_profile_type=winning_meta.get("profile_type", ""),
            )

        return best_action

    # ─────────────────────────────────────────────────────────────────────────
    # RunLogger integration  (mirrors simulation rollout schema)
    # ─────────────────────────────────────────────────────────────────────────

    def _log_decision(
        self,
        state,
        vehicle,
        action,
        rollout_reward: float,
        tail_value: float,
        final_score: float,
        n_total_candidates: int = 0,
        winning_rank: int = 0,
        vfa_top1_next_station: str = "",
        rollout_changed: bool = False,
        decision_runtime_s: float = 0.0,
        winning_profile_type: str = "",
    ) -> None:
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
        station_bikes = (
            raw_bikes if isinstance(raw_bikes, dict)
            else {getattr(b, "bike_id", getattr(b, "id")): b for b in raw_bikes}
        )

        is_at_depot  = vehicle.is_at_depot()
        fixed_queue  = getattr(vehicle.location, "fixed_queue", {})
        func_pickups = depot_pickups = load_from_queue = 0
        for b_id in getattr(action, "pick_ups", []):
            if is_at_depot and b_id in fixed_queue:
                load_from_queue += 1
                continue
            b = station_bikes.get(b_id)
            if b and getattr(b, "damage_status", None) == "depot":
                depot_pickups += 1
            else:
                func_pickups += 1

        delivery_ids_raw = list(getattr(action, "delivery_bikes", []))
        vehicle_bikes = {self._bike_id(b): b for b in inv}
        if is_at_depot:
            depot_deliveries = sum(
                1 for b_id in delivery_ids_raw
                if getattr(vehicle_bikes.get(b_id), "damage_status", None) == "depot"
            )
            func_deliveries = len(delivery_ids_raw) - depot_deliveries
        else:
            depot_deliveries = 0
            func_deliveries = len(delivery_ids_raw)
        onsite_repairs   = len(getattr(action, "onsite_repairs", []))

        if is_at_depot:
            func_after  = func_before - func_deliveries + load_from_queue
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

        delivery_ids   = [self._bike_id(b) for b in delivery_ids_raw]
        bikes_involved = str(list(getattr(action, "pick_ups", [])) + delivery_ids)

        logger.log_decision({
            "day":    day,
            "hour":   clock_hour,
            "minute": minute,
            "vehicle_id":                       vehicle.id,
            "vehicle_policy_type":              self.__class__.__name__,
            "current_station_id":             vehicle.location.id,
            "is_at_depot":                    is_at_depot,
            "functional_load_before":         func_before,
            "depot_load_before":              depot_before,
            "total_load_before":              total_before,
            "functional_deliveries":          func_deliveries,
            "functional_pickups":             func_pickups,
            "onsite_repairs":                 onsite_repairs,
            "depot_pickups":                  depot_pickups,
            "depot_deliveries":               depot_deliveries,
            "load_from_queue":                load_from_queue,
            "bikes_involved":                 bikes_involved,
            "action_duration_min":            round(action_duration, 2),
            "next_station_id":                str(dest) if dest else "",
            "travel_time_min":                round(travel_time, 2),
            "functional_load_after":          max(func_after, 0),
            "depot_load_after":               max(depot_after, 0),
            "total_load_after":               total_after,
            "immediate_reward":               0.0,
            "accumulated_rollout_reward":     round(rollout_reward, 6),
            "tail_value":                     round(tail_value, 6),
            "final_decision_score":           round(final_score, 6),
            "maintenance_flag_present":       maint_flag,
            "selected_action_is_maintenance": is_maint,
            "n_total_candidates":             n_total_candidates,
            "vfa_top1_next_station":           str(vfa_top1_next_station) if vfa_top1_next_station else "",
            "rollout_changed_decision":        rollout_changed,
            "winning_candidate_rank":         winning_rank,
            "decision_runtime_s":             round(decision_runtime_s, 4),
            "winning_profile_type":           winning_profile_type,
        })
