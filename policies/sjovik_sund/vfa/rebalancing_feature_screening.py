#!/usr/bin/env python3
"""
rebalancing_feature_screening.py

Run the same pre-training diagnostics as maintenance_feature_screening.py, but
for the rebalancing feature pool:

  - current imbalance features
  - future demand/imbalance features
  - spatial/logistic rebalancing features
  - destination-local rebalancing features

The default behavior policy is GreedyMaintenancePolicy, so the sampled states
represent the maintenance-aware operating regime used in the final experiments
rather than a pure rebalancing-only baseline.
By default, each episode uses the same timing convention as VFA training:
7 days of greedy-maintenance warmup followed by 21 analysis days.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "fomosim_matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/private/tmp") / "fomosim_cache"))

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE_ROOT))

from policies.greedy_policy_maintenance import GreedyMaintenancePolicy 
from policies.sjovik_sund.run_simulation_ingvild import SimulationConfig, run_simulation
from policies.sjovik_sund.vfa.LinearVFAPolicy import VFA_DEBUG_FLAGS
from policies.sjovik_sund.vfa.maintenance_feature_screening import (
    FeatureScreeningPolicy,
    NEAR_CONSTANT_STD,
    POOR_SCALING_P99_ABS,
    REDUNDANCY_CORR,
    SEVERE_REDUNDANCY_CORR,
    SPARSE_FRACTION_ZERO,
    VIF_FLAG,
    VIF_SERIOUS,
    WEAK_SIGNAL_CORR,
    _condition_number,
    _corr_pairs,
    _feature_stats,
    _safe_corr_with_targets,
    _vif_scores,
)


REBALANCING_FEATURE_POOL = [
    # Current system imbalance.
    "rebalancing_imbalance",
    "squared_starvation_penalty",
    "squared_congestion_penalty",
    "starvation_severity_max",
    "congestion_severity_max",
    "starvation_count",
    "congestion_count",

    # Future demand/imbalance.
    "gross_starvation_risk",
    "gross_congestion_risk",
    "net_starvation_shortfall",
    "net_congestion_shortfall",


    # Destination-local rebalancing features.
    #"destination_starv_ratio",
    #"destination_cong_ratio",
    #"destination_travel_penalty",
    #"destination_roi_starvation",
    #"cur_station_func_deficit",
    #"functional_load_late_pressure",
    #"non_depot_late_load_pressure",
]


REBALANCING_FEATURE_METADATA = {
    "rebalancing_imbalance": ("current_imbalance", "total deviation from target inventory", "negative", "high"),
    "squared_starvation_penalty": ("current_imbalance", "mean squared shortage depth", "negative", "high"),
    "squared_congestion_penalty": ("current_imbalance", "mean squared excess inventory depth", "negative", "high"),
    "starvation_severity_max": ("current_imbalance", "severe shortage at the worst stations", "negative", "medium"),
    "congestion_severity_max": ("current_imbalance", "severe congestion at the worst stations", "negative", "medium"),
    "starvation_count": ("current_imbalance", "many stations are nearly empty", "negative", "medium"),
    "congestion_count": ("current_imbalance", "many stations are nearly full", "negative", "medium"),
    "gross_starvation_risk": ("future_imbalance", "expected departures exceed available bikes", "negative", "high"),
    "gross_congestion_risk": ("future_imbalance", "expected arrivals exceed free docks", "negative", "high"),
    "net_starvation_shortfall": ("future_imbalance", "net demand creates future shortage", "negative", "medium"),
    "net_congestion_shortfall": ("future_imbalance", "net demand creates future congestion", "negative", "medium"),
    #"destination_starv_ratio": ("destination_local", "candidate destination is short of bikes", "context", "medium"),
    #"destination_cong_ratio": ("destination_local", "candidate destination has excess bikes", "context", "medium"),
    #"destination_travel_penalty": ("destination_local", "candidate destination is far away", "negative", "medium"),
    #"destination_roi_starvation": ("destination_local", "shortage relief per unit travel is high", "positive", "medium"),
    #"cur_station_func_deficit": ("destination_local", "current station remains short after action", "negative", "medium"),
    #"functional_load_late_pressure": ("logistics", "functional cargo remains on vehicle late in shift", "negative", "medium"),
    #"non_depot_late_load_pressure": ("logistics", "vehicle carries load late while routing away from depot", "negative", "low"),
}


class RebalancingFeatureScreeningPolicy(FeatureScreeningPolicy):
    """Feature logger that can execute a greedy rebalancing behavior policy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.behavior_policy_name == "greedy_rebalancing":
            self.behavior_policy = GreedyMaintenancePolicy()

    def get_best_action(self, state, vehicle):
        if self.behavior_policy_name in ("untrained_vfa", "greedy_maintenance", "random_candidate"):
            return super().get_best_action(state, vehicle)
        if self.behavior_policy_name != "greedy_rebalancing":
            raise ValueError(f"Unknown behavior_policy: {self.behavior_policy_name}")

        self._screen_current_broken_ratio = 0.0
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

        assert self.behavior_policy is not None
        selected = self.behavior_policy.get_best_action(state, vehicle)
        behavior_phi = self._action_phi(state, vehicle, selected, base_func, base_onsite, base_depot, health_base)
        behavior_phi = self._prepare_candidate_features([behavior_phi])[0]

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


def _metadata_table(features: Iterable[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        family, high_means, expected_sign, priority = REBALANCING_FEATURE_METADATA.get(
            feature, ("", "", "", "")
        )
        rows.append({
            "feature": feature,
            "family": family,
            "high_means": high_means,
            "expected_sign": expected_sign,
            "priority": priority,
        })
    return pd.DataFrame(rows).set_index("feature")


def run_screening(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    active_features = args.features if args.features else list(REBALANCING_FEATURE_POOL)

    print("Rebalancing feature screening")
    print(f"  instance       : {args.instance}")
    print(f"  episodes       : {args.episodes}")
    print(f"  days/episode   : {args.days}")
    print(f"  warmup days    : {args.warmup_days}")
    print(f"  behavior policy: {args.behavior_policy}")
    print(f"  active features: {len(active_features)}")
    print(f"  output         : {output_dir}")

    policy = RebalancingFeatureScreeningPolicy(
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
        policy.reset_episode()
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
        fh.write(f"warmup_days: {args.warmup_days}\n")
        fh.write(f"analysis_days: {args.days}\n")
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
    print(f"\nSaved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen rebalancing VFA feature candidates.")
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
    parser.add_argument("--output_dir", type=str, default="models/rebalancing_feature_screening")
    parser.add_argument(
        "--behavior_policy",
        choices=["greedy_maintenance", "untrained_vfa", "random_candidate"],
        default="greedy_maintenance",
        help="Policy used to generate visited states.",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Optional explicit feature list. Defaults to the rebalancing feature pool.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_screening(parse_args())
