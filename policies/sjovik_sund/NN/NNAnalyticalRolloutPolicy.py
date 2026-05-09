"""
NNAnalyticalRolloutPolicy.py - fast analytical rollout with NN terminal value.

This mirrors the analytical rollout strategy used by
policies.sjovik_sund.vfa.HybridRolloutPolicy, but replaces the linear VFA
terminal value with NNValueNetwork(encode_state(projected_terminal_mdp_state)).
It avoids simulator cloning inside rollout.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from policies.policy import Policy
from policies.sjovik_sund.NN.nn_model import NNValueNetwork
from policies.sjovik_sund.NN.nn_state_encoder import encode_state
from policies.sjovik_sund.NN.NNRolloutPolicy import REWARD_DENORM_SCALE
from policies.sjovik_sund.mdp.candidate_generator_nn import generate_candidates
from policies.sjovik_sund.mdp.mdp_config import MDPConfig
from policies.sjovik_sund.mdp.mdp_formulation import (
    DepotInventory,
    MDPState,
    PostDecisionState,
    StationInventory,
    VehicleStatus,
    extract_mdp_state,
)
from policies.sjovik_sund.mdp.reward import RewardCalculator
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
from settings import MAINTENANCE_REPAIR, MINUTES_CONSTANT_PER_ACTION, MINUTES_PER_ACTION


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


class NNAnalyticalRolloutPolicy(Policy):
    def __init__(
        self,
        nn_model: NNValueNetwork,
        lookahead_minutes: float = 60.0,
        num_scenarios: int = 8,
        n_rollout_candidates: int = 999,
        n_routing_candidates: int = 10,
        n_time_steps: int = 4,
        gamma: float = 0.99,
        maintenance_enabled: bool = False,
        depot_id: str | None = None,
        congestion_weight: float = -1.0,
        debug_print: bool = False,
    ):
        super().__init__(maintenance_enabled=maintenance_enabled)
        self.nn_model = nn_model
        self.nn_model.eval()

        self.lookahead_minutes = lookahead_minutes
        self.num_scenarios = num_scenarios
        self.n_rollout_candidates = n_rollout_candidates
        self.n_routing_candidates = n_routing_candidates
        self.n_time_steps = n_time_steps
        self.gamma = gamma
        self.depot_id = depot_id
        self.debug_print = debug_print

        self._mdp_config = (
            MDPConfig.full_maintenance() if maintenance_enabled
            else MDPConfig.no_maintenance()
        )
        self._reward_config = RewardCalculator(gamma=gamma).config
        self._reward_config.weight_congestion = congestion_weight

        # Helper only for analytical caches: station order, demand profiles,
        # capacities, target matrix, and travel-time matrix. Its weights are
        # never used for decisions.
        with contextlib.redirect_stdout(io.StringIO()):
            self._helper_vfa = LinearVFAPolicy(
                learning_mode=False,
                maintenance_enabled=maintenance_enabled,
                gamma=gamma,
            )
        self._helper_vfa._debug_type_check = True
        self._helper_vfa._has_printed_vision = True
        self._simulator = None

    def __deepcopy__(self, memo):
        return self

    def init_sim(self, simulator) -> None:
        self._simulator = simulator
        self._helper_vfa.init_sim(simulator)

    def _ensure_initialized(self, state) -> None:
        if not self._helper_vfa._initialized:
            self._helper_vfa._lazy_init(state)

    def _mdp_action_allowed(self, action) -> bool:
        if not self._mdp_config.allow_onsite_repairs and action.onsite_repairs != 0:
            return False
        if not self._mdp_config.allow_depot_removals:
            if action.depot_removals != 0 or action.depot_dropoffs != 0 or action.load_from_queue != 0:
                return False
        return True

    @staticmethod
    def _bike_id(bike) -> object:
        return getattr(bike, "bike_id", getattr(bike, "id", bike))

    def _classify_action_effect(self, vehicle, action) -> _ActionEffect:
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
        helper = self._helper_vfa
        func = base_func.copy().astype(np.float64)
        onsite = base_onsite.copy().astype(np.float64)
        depot = base_depot.copy().astype(np.float64)

        effect = self._classify_action_effect(vehicle, action)
        if not vehicle.is_at_depot() and vehicle.location.id in helper._sid_to_idx:
            idx = helper._sid_to_idx[vehicle.location.id]
            func[idx] = max(0.0, func[idx] + effect.station_delta_func - effect.onsite_repairs)
            if effect.onsite_repairs:
                applied = min(effect.onsite_repairs, int(onsite[idx]))
                onsite[idx] = max(0.0, onsite[idx] - applied)
                func[idx] += float(applied)
            if effect.depot_pickups:
                depot[idx] = max(0.0, depot[idx] + effect.station_delta_depot)

        return func, onsite, depot, effect

    def _current_operation_time(self, effect: _ActionEffect) -> float:
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
            + MINUTES_CONSTANT_PER_ACTION
        )

    def _action_arrival_time(self, state, vehicle, action, dest_id: Optional[str]) -> float:
        if not dest_id:
            return float("inf")
        try:
            travel_time = float(state.get_vehicle_travel_time(vehicle.location.id, dest_id))
        except Exception:
            travel_time = float("inf")
        if not np.isfinite(travel_time):
            return float("inf")
        return float(state.time) + self._current_operation_time(self._classify_action_effect(vehicle, action)) + travel_time

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
        helper = self._helper_vfa
        target_matrix = helper._target_matrix
        capacities = helper._capacities
        assert target_matrix is not None and capacities is not None

        eval_day = int(eval_time // (24 * 60)) % 7
        eval_hour = int((eval_time // 60) % 24)
        cur_func = int(max(0, round(float(func[dest_idx]))))
        cur_onsite = int(max(0, round(float(onsite[dest_idx]))))
        cur_depot = int(max(0, round(float(depot[dest_idx]))))
        target = int(round(float(target_matrix[eval_day, eval_hour, dest_idx])))
        station_spare_cap = max(0, int(capacities[dest_idx]) - cur_func - cur_onsite - cur_depot)
        free_cap = max(0, vehicle_capacity - func_cargo - depot_cargo)

        delta = cur_func - target
        greedy = -min(delta, free_cap) if delta > 0 else min(-delta, func_cargo) if delta < 0 else 0
        max_pickup = -min(cur_func, free_cap)
        max_delivery = min(func_cargo, station_spare_cap)
        rebalancing_options = {0, greedy, max_pickup, max_delivery}
        if max_pickup < 0:
            rebalancing_options.add(int(max_pickup / 2))
        if max_delivery > 0:
            rebalancing_options.add(int(max_delivery / 2))

        if self.maintenance_enabled:
            fractions = (0.0, 0.25, 0.50, 0.75, 1.0)
            onsite_options = {round(cur_onsite * f) for f in fractions} if cur_onsite > 0 else {0}
            depot_options = {round(cur_depot * f) for f in fractions} if cur_depot > 0 else {0}
        else:
            onsite_options = {0}
            depot_options = {0}

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
        func_cargo: int,
        depot_cargo: int,
        vehicle_capacity: int,
        n_options: int = 5,
    ) -> list[str]:
        helper = self._helper_vfa
        assert helper._target_matrix is not None
        assert helper._leave_profile is not None
        assert helper._arrive_profile is not None
        assert helper._travel_time_matrix is not None
        assert helper._capacities is not None

        eval_day = int(eval_time // (24 * 60)) % 7
        eval_hour = int((eval_time // 60) % 24)
        day_type = 1 if eval_day in (5, 6) else 0
        target = helper._target_matrix[eval_day, eval_hour]
        delta = func - target
        net_demand = helper._arrive_profile[day_type, eval_hour] - helper._leave_profile[day_type, eval_hour]

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
        ttv[rising] = np.maximum(0.0, helper._capacities[rising] - func[rising]) / np.maximum(net_demand[rising], 1e-6)
        ttv[falling] = np.maximum(0.0, func[falling]) / np.maximum(-net_demand[falling], 1e-6)
        urgency = 1.0 - np.minimum(ttv, 8.0) / 8.0

        worsens = ((delta < 0) & (net_demand < 0)) | ((delta > 0) & (net_demand > 0))
        corrects = ((delta < 0) & (net_demand > 0)) | ((delta > 0) & (net_demand < 0))
        active = score > 0
        score[active & worsens] *= 1.0 + urgency[active & worsens]
        score[active & corrects] *= np.maximum(0.1, 1.0 - 0.5 * urgency[active & corrects])

        travel_times = helper._travel_time_matrix[current_idx]
        distance_mask = (score > 0) & (travel_times > 0)
        score[distance_mask] /= np.power(travel_times[distance_mask], 0.35)
        score[current_idx] = -np.inf

        try:
            tabu = {v.location.id for v in state.get_vehicles() if getattr(v, "id", None) != getattr(vehicle, "id", None)}
            for sid in tabu:
                if sid in helper._sid_to_idx:
                    score[helper._sid_to_idx[sid]] = -np.inf
        except Exception:
            pass

        n = min(max(1, n_options), len(score))
        top = np.argsort(score)[::-1][:n]
        return [helper._station_ids[i] for i in top if np.isfinite(score[i])]

    def _build_projected_mdp_state(
        self,
        base_mdp: MDPState,
        state,
        vehicle,
        func: np.ndarray,
        onsite: np.ndarray,
        depot: np.ndarray,
        eval_time: float,
        terminal_loc_id: Optional[str],
        terminal_next_station_id: Optional[str],
        func_cargo: int,
        depot_cargo: int,
        vehicle_capacity: int,
    ) -> MDPState:
        helper = self._helper_vfa
        assert helper._target_matrix is not None
        assert helper._leave_profile is not None
        assert helper._arrive_profile is not None
        assert helper._capacities is not None

        eval_day = int(eval_time // (24 * 60)) % 7
        eval_hour = int((eval_time // 60) % 24)
        day_type = 1 if eval_day in (5, 6) else 0

        stations = {}
        for idx, sid in enumerate(helper._station_ids):
            stations[sid] = StationInventory(
                station_id=sid,
                functional=int(max(0, round(float(func[idx])))),
                onsite=int(max(0, round(float(onsite[idx])))) if self.maintenance_enabled else 0,
                depot=int(max(0, round(float(depot[idx])))) if self.maintenance_enabled else 0,
                capacity=int(max(1, round(float(helper._capacities[idx])))),
                target=int(round(float(helper._target_matrix[eval_day, eval_hour, idx]))),
                expected_departure_rate=float(helper._leave_profile[day_type, eval_hour, idx]),
                expected_arrival_rate=float(helper._arrive_profile[day_type, eval_hour, idx]),
            )

        vehicles = dict(base_mdp.vehicles)
        destination = terminal_next_station_id or terminal_loc_id or vehicle.location.id
        eta = eval_time
        if terminal_next_station_id and terminal_loc_id and terminal_next_station_id != terminal_loc_id:
            try:
                eta += float(state.get_vehicle_travel_time(terminal_loc_id, terminal_next_station_id))
            except Exception:
                pass
        vehicles[vehicle.id] = VehicleStatus(
            vehicle_id=vehicle.id,
            destination_station=destination,
            eta=eta,
            functional_cargo=int(max(0, func_cargo)),
            depot_cargo=int(max(0, depot_cargo)) if self.maintenance_enabled else 0,
            capacity=int(max(1, vehicle_capacity)),
        )

        travel_origin = destination
        travel_times = {}
        for sid in stations:
            try:
                travel_times[sid] = state.get_vehicle_travel_time(travel_origin, sid)
            except Exception:
                travel_times[sid] = 0.0

        depot_obj = base_mdp.depot
        if depot_obj is not None:
            depot_obj = DepotInventory(
                station_id=depot_obj.station_id,
                fixed_queue=depot_obj.fixed_queue,
                in_repair=depot_obj.in_repair,
                capacity=depot_obj.capacity,
            )

        return MDPState(
            time=eval_time,
            active_vehicle_id=vehicle.id,
            stations=stations,
            depot=depot_obj,
            vehicles=vehicles,
            config=self._mdp_config,
            shift_end_time=getattr(vehicle, "shift_end_time", base_mdp.shift_end_time),
            travel_times=travel_times,
        )

    def _nn_value_from_mdp(self, mdp_state: MDPState) -> float:
        encoded = encode_state(mdp_state)
        device = next(self.nn_model.parameters()).device
        with torch.no_grad():
            value_tensor = self.nn_model(
                encoded["station_block"].to(device),
                encoded["vehicle_block"].to(device),
                encoded["global_context"].to(device),
            )
        return float(value_tensor.item()) * REWARD_DENORM_SCALE

    def _nn_value_from_projection(
        self,
        base_mdp: MDPState,
        state,
        vehicle,
        func: np.ndarray,
        onsite: np.ndarray,
        depot: np.ndarray,
        eval_time: float,
        terminal_loc_id: Optional[str],
        terminal_next_station_id: Optional[str],
        func_cargo: int,
        depot_cargo: int,
        vehicle_capacity: int,
    ) -> float:
        terminal_mdp = self._build_projected_mdp_state(
            base_mdp, state, vehicle, func, onsite, depot, eval_time,
            terminal_loc_id, terminal_next_station_id,
            func_cargo, depot_cargo, vehicle_capacity,
        )
        return self._nn_value_from_mdp(terminal_mdp)

    def _apply_nn_base_step(
        self,
        base_mdp: MDPState,
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, Optional[str]]:
        helper = self._helper_vfa
        if dest_id is None:
            return func_proj, onsite_proj, depot_proj, func_cargo, depot_cargo, None
        if dest_id not in helper._sid_to_idx:
            if self._is_depot_id(state, dest_id):
                return func_proj, onsite_proj, depot_proj, func_cargo, 0, None
            return func_proj, onsite_proj, depot_proj, func_cargo, depot_cargo, None

        dest_idx = helper._sid_to_idx[dest_id]
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
                cand_func, cand_func_cargo, cand_depot_cargo, vehicle_capacity,
                n_options=min(5, self.n_routing_candidates),
            ) or [None]

            for next_id in next_options:
                value = self._nn_value_from_projection(
                    base_mdp, state, vehicle,
                    cand_func, cand_onsite, cand_depot,
                    eval_time, dest_id, next_id,
                    cand_func_cargo, cand_depot_cargo, vehicle_capacity,
                )
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
        return func_proj, onsite_proj, depot_proj, func_cargo, depot_cargo, best_next

    def _run_analytical_scenario(
        self,
        base_mdp: MDPState,
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
    ) -> tuple[float, float]:
        helper = self._helper_vfa
        assert helper._capacities is not None
        assert helper._leave_profile is not None
        assert helper._arrive_profile is not None

        func_proj = func.copy()
        onsite_proj = onsite.copy()
        depot_proj = depot.copy()

        cfg = self._reward_config
        dt = self.lookahead_minutes / self.n_time_steps
        t0 = float(state.time)
        accumulated_reward = 0.0
        dest_service_done = False
        terminal_next_station_id: Optional[str] = dest_id

        for step in range(self.n_time_steps):
            step_start = t0 + step * dt
            step_end = step_start + dt
            next_boundary = (int(step_start // 60) + 1) * 60.0

            if next_boundary >= step_end:
                h0 = int((step_start // 60) % 24)
                day0 = int(step_start // (24 * 60)) % 7
                dt_type = 1 if day0 in [5, 6] else 0
                mean_leave = helper._leave_profile[dt_type, h0] * (dt / 60.0)
                mean_arrive = helper._arrive_profile[dt_type, h0] * (dt / 60.0)
            else:
                dt_h0 = next_boundary - step_start
                dt_h1 = step_end - next_boundary
                h0 = int((step_start // 60) % 24)
                day0 = int(step_start // (24 * 60)) % 7
                h1 = (h0 + 1) % 24
                day1 = day0 if h1 != 0 else (day0 + 1) % 7
                dt0_type = 1 if day0 in [5, 6] else 0
                dt1_type = 1 if day1 in [5, 6] else 0
                mean_leave = (
                    helper._leave_profile[dt0_type, h0] * (dt_h0 / 60.0)
                    + helper._leave_profile[dt1_type, h1] * (dt_h1 / 60.0)
                )
                mean_arrive = (
                    helper._arrive_profile[dt0_type, h0] * (dt_h0 / 60.0)
                    + helper._arrive_profile[dt1_type, h1] * (dt_h1 / 60.0)
                )

            if not dest_service_done and step_start >= dest_arrival_min:
                (
                    func_proj,
                    onsite_proj,
                    depot_proj,
                    func_cargo,
                    depot_cargo,
                    terminal_next_station_id,
                ) = self._apply_nn_base_step(
                    base_mdp, state, vehicle, dest_id, step_start,
                    func_proj, onsite_proj, depot_proj,
                    func_cargo, depot_cargo, vehicle_capacity,
                )
                dest_service_done = True

            n_depart = rng.poisson(mean_leave)
            n_arrive = rng.poisson(mean_arrive)
            available = np.maximum(func_proj, 0.0).astype(int)
            actual_depart = np.minimum(n_depart, available)
            starvations = int(np.sum(n_depart - actual_depart))

            func_proj -= actual_depart
            func_proj += n_arrive

            overflow = func_proj > helper._capacities
            congestions = int(np.sum(overflow))
            func_proj = np.minimum(func_proj, helper._capacities)

            step_reward = (cfg.weight_starvation * starvations + cfg.weight_congestion * congestions) * (1.0 - self.gamma)
            time_elapsed = step_start - t0
            discount = self.gamma ** max(time_elapsed / 60.0, 0.0)
            accumulated_reward += discount * step_reward

        terminal_value = self._nn_value_from_projection(
            base_mdp, state, vehicle,
            func_proj, onsite_proj, depot_proj,
            t0 + self.lookahead_minutes,
            dest_id if dest_id is not None else vehicle.location.id,
            terminal_next_station_id,
            func_cargo,
            depot_cargo,
            vehicle_capacity,
        )
        discounted_tail = (self.gamma ** (self.lookahead_minutes / 60.0)) * terminal_value
        return accumulated_reward, discounted_tail

    def get_best_action(self, state, vehicle):
        self._ensure_initialized(state)
        base_mdp = extract_mdp_state(
            sim_state=state,
            active_vehicle_id=vehicle.id,
            config=self._mdp_config,
            depot_id=self.depot_id,
            shift_end_time=getattr(vehicle, "shift_end_time", None),
        )

        pairs = generate_candidates(
            state=state,
            vehicle=vehicle,
            maintenance_enabled=self.maintenance_enabled,
            return_pairs=True,
            wide_search=True,
            n_candidates=self.n_routing_candidates,
            training_mode=False,
        )
        pairs = [(mdp_a, sim_a) for mdp_a, sim_a in pairs if self._mdp_action_allowed(mdp_a)]
        if not pairs:
            return None

        device = next(self.nn_model.parameters()).device
        pre_scores = []
        with torch.no_grad():
            for mdp_action, sim_action in pairs:
                try:
                    post_state, action_duration, _ = PostDecisionState.apply(base_mdp, mdp_action)
                    dest = mdp_action.next_station
                    dest_tt = {sid: state.get_vehicle_travel_time(dest, sid) for sid in base_mdp.stations}
                    encoded = encode_state(
                        post_state,
                        dest_travel_times=dest_tt,
                        mdp_action=mdp_action,
                        action_duration=action_duration,
                    )
                    value = self.nn_model(
                        encoded["station_block"].to(device),
                        encoded["vehicle_block"].to(device),
                        encoded["global_context"].to(device),
                    ).item()
                    pre_scores.append((value * REWARD_DENORM_SCALE, mdp_action, sim_action))
                except Exception:
                    continue

        pre_scores.sort(key=lambda x: x[0], reverse=True)
        kept = pre_scores[:self.n_rollout_candidates]
        if not kept:
            return None

        base_func, base_onsite, base_depot = self._helper_vfa._extract_inventories(state, vehicle)
        inv = vehicle.get_bike_inventory()
        veh_func_cargo = sum(1 for b in inv if getattr(b, "damage_status", None) not in ("depot", "onsite"))
        veh_depot_cargo = sum(1 for b in inv if getattr(b, "damage_status", None) == "depot")
        veh_capacity = int(getattr(vehicle, "bike_inventory_capacity", getattr(vehicle, "capacity", len(inv))))

        crn_seeds = [int(state.rng.integers(1_000_000)) for _ in range(self.num_scenarios)]
        best_action = None
        best_q = -float("inf")

        for _, _, action in kept:
            dest_id = getattr(action, "next_location", getattr(action, "next_station", None))
            func_post, onsite_post, depot_post, effect = self._apply_action_delta(
                base_func, base_onsite, base_depot, vehicle, action
            )
            func_cargo_post = max(0, veh_func_cargo + effect.delta_func_cargo)
            depot_cargo_post = max(0, veh_depot_cargo + effect.delta_depot_cargo)
            dest_arrival_min = self._action_arrival_time(state, vehicle, action, dest_id)

            q_sum = 0.0
            for seed in crn_seeds:
                rng = np.random.default_rng(seed)
                r, t = self._run_analytical_scenario(
                    base_mdp, state, vehicle,
                    func_post, onsite_post, depot_post, rng,
                    dest_id=dest_id,
                    func_cargo=func_cargo_post,
                    depot_cargo=depot_cargo_post,
                    vehicle_capacity=veh_capacity,
                    dest_arrival_min=dest_arrival_min,
                )
                q_sum += r + t
            expected_q = q_sum / max(1, self.num_scenarios)
            if expected_q > best_q:
                best_q = expected_q
                best_action = action

        if self.debug_print:
            print(
                f"[NN ANALYTICAL] {len(pairs)} candidates -> {len(kept)} kept | "
                f"S={self.num_scenarios} H={self.lookahead_minutes:.0f} | best_q={best_q:.4f}"
            )
        return best_action
