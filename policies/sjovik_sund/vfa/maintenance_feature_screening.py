#!/usr/bin/env python3
"""
maintenance_feature_screening.py

Collect candidate post-decision feature vectors and TD transitions, then run
the pre-training diagnostics used to prune the maintenance feature pool:

  - descriptive statistics and sparsity
  - scaling checks
  - Pearson/Spearman redundancy
  - feature-target correlations
  - VIF and feature-matrix condition number

The script is intentionally diagnostic.  It does not choose a final feature set
automatically; it produces CSV evidence for the thesis feature-screening step.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path("/private/tmp") / "fomosim_matplotlib_cache"),
)
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/private/tmp") / "fomosim_cache"))

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE_ROOT))

from policies.sjovik_sund.mdp.reward import RewardCalculator, RewardConfig
from policies.sjovik_sund.run_simulation_ingvild import SimulationConfig, run_simulation
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy, VFA_DEBUG_FLAGS
from policies.sjovik_sund.vfa.vfa_features import (
    get_base_rebalancing_feature_names,
    get_feature_metadata,
    get_maintenance_feature_pool_names,
)
from policies.greedy_policy_maintenance import GreedyMaintenancePolicy


NEAR_CONSTANT_STD = 1e-4
SPARSE_FRACTION_ZERO = 0.95
POOR_SCALING_P99_ABS = 5.0
REDUNDANCY_CORR = 0.85
SEVERE_REDUNDANCY_CORR = 0.95
WEAK_SIGNAL_CORR = 0.03
VIF_FLAG = 10.0
VIF_SERIOUS = 20.0


class DiagnosticRewardCalculator(RewardCalculator):
    """RewardCalculator that exposes the last realized service-level deltas."""

    def __init__(self, config: RewardConfig | None = None, gamma: float = 0.99):
        super().__init__(config=config, gamma=gamma)
        self.last_delta_starvation = 0.0
        self.last_delta_congestion = 0.0
        self.last_delta_trips = 0.0

    def compute_step_reward(self, simulator_metrics) -> float:
        cur_s = simulator_metrics.get_aggregate_value("starvations") or 0
        cur_c = simulator_metrics.get_aggregate_value("long congestions") or 0
        cur_t = simulator_metrics.get_aggregate_value("trips") or 0

        self.last_delta_starvation = float(cur_s - self._prev_starvations)
        self.last_delta_congestion = float(cur_c - self._prev_congestions)
        self.last_delta_trips = float(cur_t - self._prev_trips)

        reward = 0.0
        reward += self.config.weight_starvation * self.last_delta_starvation
        reward += self.config.weight_congestion * self.last_delta_congestion
        if self.config.weight_trip_served != 0.0:
            served = max(0.0, self.last_delta_trips - self.last_delta_starvation - self.last_delta_congestion)
            reward += self.config.weight_trip_served * served

        self._prev_starvations = cur_s
        self._prev_congestions = cur_c
        self._prev_trips = cur_t
        return reward * self._scale_factor


def _broken_fleet_ratio(state) -> float:
    """Current realized broken-fleet ratio used as a transition target proxy."""
    all_bikes = list(state.get_all_bikes())
    depot_q = sum(len(bikes) for d in state.get_depots() for _, bikes in d.in_repair)
    total = len(all_bikes) + depot_q
    if total <= 0:
        return 0.0
    broken = sum(
        1
        for b in all_bikes
        if getattr(b, "damage_status", None) in ("onsite", "depot")
    )
    return float(broken + depot_q) / float(total)


class FeatureScreeningPolicy(LinearVFAPolicy):
    """Linear VFA wrapper that logs candidates and realized TD transitions."""

    def __init__(self, *args, **kwargs):
        self.behavior_policy_name = kwargs.pop("behavior_policy_name", "untrained_vfa")
        gamma = float(kwargs.get("gamma", 0.97))
        kwargs.setdefault(
            "reward_calculator",
            DiagnosticRewardCalculator(config=RewardConfig(), gamma=gamma),
        )
        super().__init__(*args, **kwargs)
        self.behavior_policy = (
            GreedyMaintenancePolicy()
            if self.behavior_policy_name == "greedy_maintenance"
            else None
        )
        self.candidate_feature_log: list[np.ndarray] = []
        self.transition_feature_log: list[np.ndarray] = []
        self.transition_target_log: list[dict[str, float]] = []
        self._screen_current_broken_ratio = 0.0

    def _action_phi(self, state, vehicle, action, base_func, base_onsite, base_depot, health_base):
        """Compute post-decision features for one sim.Action."""
        raw_bikes = getattr(vehicle.location, "bikes", [])
        if isinstance(raw_bikes, dict):
            station_bikes = raw_bikes
        else:
            station_bikes = {getattr(b, "bike_id", getattr(b, "id")): b for b in raw_bikes}

        functional_pickups = 0
        depot_pickups = 0
        if vehicle.is_at_depot():
            fixed_queue = getattr(vehicle.location, "fixed_queue", {})
            vehicle_bikes = {
                getattr(b, "bike_id", getattr(b, "id")): b
                for b in vehicle.get_bike_inventory()
            }
            for bike_id in getattr(action, "pick_ups", []):
                if bike_id in fixed_queue:
                    functional_pickups += 1
            depot_dropoffs = sum(
                1
                for bike_id in getattr(action, "delivery_bikes", [])
                if getattr(vehicle_bikes.get(bike_id), "damage_status", None) == "depot"
            )
            delta_func = -functional_pickups
            delta_depot_cargo = -depot_dropoffs
        else:
            for bike_id in getattr(action, "pick_ups", []):
                bike = station_bikes.get(bike_id)
                if bike and getattr(bike, "damage_status", None) == "depot":
                    depot_pickups += 1
                elif bike:
                    functional_pickups += 1
            delta_func = len(getattr(action, "delivery_bikes", [])) - functional_pickups
            delta_depot_cargo = depot_pickups

        delta_onsite_repairs = len(getattr(action, "onsite_repairs", []))
        dest_id = getattr(action, "next_location", getattr(action, "next_station", None))
        try:
            service_time = action.get_action_time(0.0) if hasattr(action, "get_action_time") else 0.0
        except Exception:
            service_time = 0.0
        try:
            travel = state.get_vehicle_travel_time(vehicle.location.id, dest_id) if dest_id else 0.0
        except Exception:
            travel = 0.0

        action_eval_time = state.time + service_time + travel
        return self.extract_features(
            state,
            vehicle,
            base_func,
            base_onsite,
            base_depot,
            delta_func,
            delta_depot_cargo,
            delta_onsite_repairs,
            time_remaining=self._get_time_remaining(state, vehicle),
            shift_length=self._get_shift_length(state, vehicle),
            next_station_id=dest_id,
            eval_time=action_eval_time,
            projected_time=action_eval_time,
            dist_to_next=travel,
            candidate_action=action,
            health_base=health_base,
        )

    def get_best_action(self, state, vehicle):
        self._screen_current_broken_ratio = _broken_fleet_ratio(state)
        if self.behavior_policy_name == "untrained_vfa":
            return super().get_best_action(state, vehicle)

        if not self._initialized:
            self._lazy_init(state)

        candidates = self._generate_candidates(state, vehicle)
        base_func, base_onsite, base_depot = self._extract_inventories(state, vehicle)
        health_base = self._compute_health_base(state, vehicle) if self._use_health_features else None

        candidate_phis = [
            self._action_phi(state, vehicle, action, base_func, base_onsite, base_depot, health_base)
            for action in candidates
        ]
        candidate_phis = self._prepare_candidate_features(candidate_phis)
        values = np.array([self.value(phi) for phi in candidate_phis], dtype=np.float64)

        if self.behavior_policy_name == "greedy_maintenance":
            assert self.behavior_policy is not None
            selected = self.behavior_policy.get_best_action(state, vehicle)
            behavior_phi = self._action_phi(
                state, vehicle, selected, base_func, base_onsite, base_depot, health_base
            )
            behavior_phi = self._prepare_candidate_features([behavior_phi])[0]
        elif self.behavior_policy_name == "random_candidate":
            sel_idx = int(self._rng.integers(len(candidates)))
            selected = candidates[sel_idx]
            behavior_phi = candidate_phis[sel_idx]
        else:
            raise ValueError(f"Unknown behavior_policy: {self.behavior_policy_name}")

        if self.learning_mode and self._prev_phi is not None:
            reward = self.reward_calc.compute_step_reward(state.metrics)
            reward += self.reward_calc.compute_fleet_penalty(state)
            reward += self.reward_calc.compute_late_shift_penalty(vehicle, state)
            reward = self.reward_calc.center_reward(reward)
            elapsed_minutes = state.time - self._prev_time
            self.td_update(reward, behavior_phi, elapsed_minutes)
        elif self.learning_mode and self._prev_phi is None:
            _ = self.reward_calc.compute_step_reward(state.metrics)

        self._prev_phi = behavior_phi
        self._prev_time = state.time
        self._ep_decision_count += 1
        return selected

    def extract_features(self, *args, **kwargs):
        phi = super().extract_features(*args, **kwargs)
        self.candidate_feature_log.append(phi.copy())
        return phi

    def td_update(self, reward: float, phi_next: np.ndarray, elapsed_minutes: float) -> float:
        if self._prev_phi is not None:
            v_cur = self.value(self._prev_phi)
            v_next = self.value(phi_next)
            discount = self.gamma ** (elapsed_minutes / 60.0)
            midpoint_discount = self.gamma ** ((elapsed_minutes / 2.0) / 60.0)
            td_target = (reward * midpoint_discount) + (discount * v_next)

            rc = self.reward_calc
            self.transition_feature_log.append(self._prev_phi.copy())
            self.transition_target_log.append({
                "reward": float(reward),
                "td_target": float(td_target),
                "td_error": float(td_target - v_cur),
                "future_starvation": float(getattr(rc, "last_delta_starvation", np.nan)),
                "future_congestion": float(getattr(rc, "last_delta_congestion", np.nan)),
                "future_broken_fleet_ratio": float(self._screen_current_broken_ratio),
            })
        return super().td_update(reward, phi_next, elapsed_minutes)


def _safe_corr_with_targets(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in features.columns:
        x = features[feature].astype(float)
        row = {"feature": feature}
        for target in targets.columns:
            y = targets[target].astype(float)
            if x.std(ddof=0) < 1e-12 or y.std(ddof=0) < 1e-12:
                r = np.nan
            else:
                r = x.corr(y, method="pearson")
            row[f"corr_{target}"] = r
        rows.append(row)
    return pd.DataFrame(rows).set_index("feature")


def _corr_pairs(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            val = corr.loc[a, b]
            if pd.notna(val) and abs(float(val)) >= threshold:
                rows.append({
                    "feature_a": a,
                    "feature_b": b,
                    "corr": float(val),
                    "abs_corr": abs(float(val)),
                })
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False) if rows else pd.DataFrame(
        columns=["feature_a", "feature_b", "corr", "abs_corr"]
    )


def _standardized_matrix(df: pd.DataFrame, max_rows: int = 20000) -> tuple[np.ndarray, list[str]]:
    numeric = df.astype(float).replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
    std = numeric.std(axis=0, ddof=0)
    keep = std[std > 1e-10].index.tolist()
    numeric = numeric[keep]
    if len(numeric) > max_rows:
        numeric = numeric.sample(max_rows, random_state=17)
    x = numeric.to_numpy(dtype=np.float64)
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-10)
    return x, keep


def _vif_scores(df: pd.DataFrame) -> pd.Series:
    x, names = _standardized_matrix(df)
    if x.shape[1] < 2:
        return pd.Series(dtype=float)

    scores = {}
    for j, name in enumerate(names):
        y = x[:, j]
        others = np.delete(x, j, axis=1)
        beta, *_ = np.linalg.lstsq(others, y, rcond=None)
        pred = others @ beta
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        scores[name] = np.inf if r2 >= 1.0 else 1.0 / max(1.0 - r2, 1e-12)
    return pd.Series(scores, name="vif").sort_values(ascending=False)


def _condition_number(df: pd.DataFrame) -> float:
    x, _ = _standardized_matrix(df)
    if x.size == 0 or x.shape[1] < 2:
        return float("nan")
    return float(np.linalg.cond(x))


def _feature_stats(df: pd.DataFrame, target_corr: pd.DataFrame, vif: pd.Series) -> pd.DataFrame:
    stats = df.describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).T
    stats = stats.rename(columns={
        "1%": "p01",
        "5%": "p05",
        "25%": "p25",
        "50%": "p50",
        "75%": "p75",
        "95%": "p95",
        "99%": "p99",
    })
    stats["fraction_zero"] = (df == 0.0).mean(axis=0)
    stats["p99_abs"] = df.abs().quantile(0.99)
    stats["near_constant"] = stats["std"].fillna(0.0) < NEAR_CONSTANT_STD
    stats["sparse"] = stats["fraction_zero"] > SPARSE_FRACTION_ZERO
    stats["poor_scaling"] = (stats["p99_abs"] > POOR_SCALING_P99_ABS) | (stats["min"] < -1e-6)

    corr_abs = target_corr.abs()
    stats["max_abs_target_corr"] = corr_abs.max(axis=1)
    stats["weak_signal"] = stats["max_abs_target_corr"].fillna(0.0) < WEAK_SIGNAL_CORR
    stats["vif"] = vif.reindex(stats.index)
    stats["vif_flag"] = stats["vif"] > VIF_FLAG
    stats["vif_serious"] = stats["vif"] > VIF_SERIOUS
    return stats


def _metadata_table(features: Iterable[str]) -> pd.DataFrame:
    metadata = get_feature_metadata()
    rows = []
    for feature in features:
        meta = metadata.get(feature, {})
        rows.append({
            "feature": feature,
            "family": meta.get("family", ""),
            "high_means": meta.get("high_means", ""),
            "expected_sign": meta.get("expected_sign", ""),
            "priority": meta.get("priority", ""),
        })
    return pd.DataFrame(rows).set_index("feature")


def build_default_feature_set() -> list[str]:
    """Base rebalancing features plus the maintenance candidate pool."""
    return get_base_rebalancing_feature_names() + get_maintenance_feature_pool_names()


def run_screening(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    active_features = args.features if args.features else build_default_feature_set()
    print("Maintenance feature screening")
    print(f"  instance       : {args.instance}")
    print(f"  episodes       : {args.episodes}")
    print(f"  days/episode   : {args.days}")
    print(f"  warmup days    : {args.warmup_days}")
    print(f"  behavior policy: {args.behavior_policy}")
    print(f"  active features: {len(active_features)}")
    print(f"  output         : {output_dir}")

    policy = FeatureScreeningPolicy(
        active_features=active_features,
        learning_mode=True,
        maintenance_enabled=True,
        logistics_enabled=True,
        demand_horizon_enabled=True,
        gamma=args.gamma,
        alpha=args.alpha,
        epsilon=args.epsilon,
        use_bias_feature=False,
        use_terminal_update=False,
        transition_update_interval=0,
        behavior_policy_name=args.behavior_policy,
    )
    policy.use_td_lambda = False
    policy.td_lambda = 0.0
    for key in VFA_DEBUG_FLAGS:
        if key.startswith("check"):
            VFA_DEBUG_FLAGS[key] = False

    config = SimulationConfig()
    config.start_hour = args.start_hour

    warmup_hours = float(args.warmup_days) * 24.0
    for ep in range(args.episodes):
        run_simulation(
            seed=args.seed_start + ep,
            policy=policy,
            duration=24 * args.days,
            num_vehicles=args.vehicles,
            instance_name=args.instance,
            config=config,
            warmup_hours=warmup_hours,
        )
        policy.apply_batch_update()
        print(
            f"  episode {ep + 1}/{args.episodes}: "
            f"{len(policy.candidate_feature_log)} candidate phis, "
            f"{len(policy.transition_feature_log)} TD transitions"
        )

    candidate_df = pd.DataFrame(policy.candidate_feature_log, columns=policy.FEATURE_NAMES)
    transition_df = pd.DataFrame(policy.transition_feature_log, columns=policy.FEATURE_NAMES)
    target_df = pd.DataFrame(policy.transition_target_log)

    if candidate_df.empty:
        raise RuntimeError("No candidate feature vectors were collected.")

    target_corr = (
        _safe_corr_with_targets(transition_df, target_df)
        if not transition_df.empty and not target_df.empty
        else pd.DataFrame(index=candidate_df.columns)
    )
    pearson = candidate_df.corr(method="pearson")
    spearman = candidate_df.corr(method="spearman")
    pearson_pairs = _corr_pairs(pearson, REDUNDANCY_CORR)
    spearman_pairs = _corr_pairs(spearman, REDUNDANCY_CORR)
    vif = _vif_scores(candidate_df)
    condition_number = _condition_number(candidate_df)
    stats = _feature_stats(candidate_df, target_corr, vif)
    metadata = _metadata_table(candidate_df.columns)

    summary = metadata.join(stats, how="right")
    summary["recommendation_flag"] = ""
    summary.loc[summary["near_constant"], "recommendation_flag"] += "near_constant;"
    summary.loc[summary["sparse"], "recommendation_flag"] += "sparse;"
    summary.loc[summary["poor_scaling"], "recommendation_flag"] += "poor_scaling;"
    summary.loc[summary["weak_signal"], "recommendation_flag"] += "weak_signal;"
    summary.loc[summary["vif_flag"], "recommendation_flag"] += "vif;"
    summary.loc[summary["vif_serious"], "recommendation_flag"] += "vif_serious;"

    candidate_df.to_csv(output_dir / "candidate_features.csv", index=False)
    transition_df.to_csv(output_dir / "transition_features.csv", index=False)
    target_df.to_csv(output_dir / "transition_targets.csv", index=False)
    summary.to_csv(output_dir / "feature_screening_summary.csv")
    target_corr.to_csv(output_dir / "feature_target_correlations.csv")
    pearson.to_csv(output_dir / "feature_pearson_correlations.csv")
    spearman.to_csv(output_dir / "feature_spearman_correlations.csv")
    pearson_pairs.to_csv(output_dir / "redundant_pairs_pearson.csv", index=False)
    spearman_pairs.to_csv(output_dir / "redundant_pairs_spearman.csv", index=False)
    vif.to_csv(output_dir / "vif_scores.csv")

    with open(output_dir / "screening_thresholds.txt", "w") as fh:
        fh.write(f"near_constant: std < {NEAR_CONSTANT_STD}\n")
        fh.write(f"sparse: fraction_zero > {SPARSE_FRACTION_ZERO}\n")
        fh.write(f"poor_scaling: p99_abs > {POOR_SCALING_P99_ABS} or min < 0\n")
        fh.write(f"redundancy: abs(corr) > {REDUNDANCY_CORR}, severe > {SEVERE_REDUNDANCY_CORR}\n")
        fh.write(f"weak_signal: max abs target corr < {WEAK_SIGNAL_CORR}\n")
        fh.write(f"vif_flag: VIF > {VIF_FLAG}, serious > {VIF_SERIOUS}\n")
        fh.write(f"condition_number: {condition_number:.6g}\n")

    flagged = summary[summary["recommendation_flag"].astype(bool)]
    print("\nScreening complete.")
    print(f"  condition number: {condition_number:.3g}")
    print(f"  flagged features: {len(flagged)}/{len(summary)}")
    if not flagged.empty:
        cols = ["family", "priority", "std", "fraction_zero", "p99_abs", "max_abs_target_corr", "vif", "recommendation_flag"]
        print(flagged[cols].sort_values("recommendation_flag").to_string())
    print("\nSaved:")
    for name in [
        "feature_screening_summary.csv",
        "feature_target_correlations.csv",
        "redundant_pairs_pearson.csv",
        "redundant_pairs_spearman.csv",
        "vif_scores.csv",
        "screening_thresholds.txt",
    ]:
        print(f"  {output_dir / name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen maintenance VFA feature candidates.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument(
        "--warmup_days",
        type=float,
        default=7.0,
        help=(
            "Evaluation warmup in days before feature/transition logging starts. "
            "Uses GreedyMaintenancePolicy (or GreedyPolicy without maintenance) during warmup. "
            "Default: 7."
        ),
    )
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--instance", type=str, default="TD_W34_old")
    parser.add_argument("--vehicles", type=int, default=1)
    parser.add_argument("--start_hour", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="models/maintenance_feature_screening")
    parser.add_argument(
        "--behavior_policy",
        choices=["greedy_maintenance", "untrained_vfa", "random_candidate"],
        default="greedy_maintenance",
        help="Policy used to generate the visited states. Greedy maintenance is recommended for screening.",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Optional explicit feature list. Defaults to base rebalancing + maintenance pool.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_screening(parse_args())
