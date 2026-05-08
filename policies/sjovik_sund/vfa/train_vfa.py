#!/usr/bin/env python3
"""
train_vfa.py  -  Offline Episodic Training of Time-Indexed Linear VFA

Architecture
─────────────
Hybrid "Horizontal" ADP (Ulmer 2020 / Brinkmann 2019-2020):
  - VFA is trained completely offline via episodic TD(0) simulation.
  - Once frozen, the VFA acts as a tail-value estimator inside an online
    Rollout Algorithm (implemented separately).

Episode structure  (28-day simulation per episode)
──────────────────────────────────────────────────
  Days 1 - 7   Warm-up   : GreedyPolicy drives the system.
                            No TD updates → builds up a realistic
                            "messy" state without biasing θ.
  Days 8 - 28  Learning  : LinearVFAPolicy.
                            TD(0) updates occur at every vehicle decision.

Usage
─────
    python train_vfa.py
    python train_vfa.py --episodes 200 --save models/my_vfa.pkl
    python train_vfa.py --episodes 50 --seed 100 --instance TD_W34_37
"""

from __future__ import annotations

import sys
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

# ── Workspace root on sys.path ────────────────────────────────────────────────
WORKSPACE_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(WORKSPACE_ROOT))

from helpers import timeInMinutes
from policies.greedy_policy import GreedyPolicy
from policies.greedy_policy_maintenance import GreedyMaintenancePolicy
from policies.do_nothing_policy import DoNothing
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy, EpisodeTrainingPolicy, VFA_DEBUG_FLAGS
from policies.sjovik_sund.vfa.vfa_features import get_feature_names as _get_feature_names
from policies.sjovik_sund.run_simulation_ingvild import run_simulation, SimulationConfig, write_simulation_outputs
from policies.sjovik_sund.mdp.reward import RewardConfig, RewardCalculator
from policies.sjovik_sund.mdp.action_bridge import reset_truncation_counts, get_truncation_summary
from policies.sjovik_sund.mdp.candidate_generator import reset_candidate_debug_counts, get_candidate_debug_summary
from settings import ENABLE_COMPONENT_FAILURES


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters  (all subject to change)
# ─────────────────────────────────────────────────────────────────────────────

NUM_EPISODES  : int   = 300      # total training episodes


WARMUP_DAYS   : int   = 7         # greedy warm-up, no TD updates
LEARNING_DAYS : int   = 21        # VFA + TD(0) 
EPISODE_DAYS  : int   = WARMUP_DAYS + LEARNING_DAYS        # days per episode (total)

# --- Learning Rate (Alpha) & Exploration (Epsilon) ---
ALPHA_START   : float = 0.01   # initial alpha for TD updates, will be overwritten in case of argument passing
EPSILON_START : float = 0.0     # initial exploration rate
EPSILON_END   : float = 0.0     # final exploration rate
TRANSITION_UPDATE_INTERVAL: int = 100  # apply mean TD batch update after this many decision transitions
TD_LAMBDA     : float = 0.0     # eligibility-trace parameter; 0.0 gives TD(0)

GAMMA         : float = 0.97      # discount factor

# ── Feature configuration ───────────────────────────────────────────────────────
LOGISTICS_ENABLED : bool = False  # Enable Pillar 4: Spatial & Logistic Constraints features
BIAS_FEATURE_ENABLED: bool = True  # Add a constant intercept feature to active VFA feature sets.
N_FEATURES    : int   = len(_get_feature_names(ENABLE_COMPONENT_FAILURES, logistics_enabled=LOGISTICS_ENABLED, demand_horizon_enabled=True))  # auto-synced with vfa_features.py

INSTANCE_NAME : str   = "TD_W34_old" #"OS_W31"
NUM_VEHICLES  : int   = 1
START_HOUR    : int   = 0         # simulation clock starts at 00:00

# Where to save checkpoints and the final model
SAVE_DIR = Path(__file__).parent / "models"


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight per-episode stats collector (duck-types RunLogger.log_decision)
# ─────────────────────────────────────────────────────────────────────────────

class _EpisodeStatsCollector:
    """Accumulates per-episode operational stats from LinearVFAPolicy decision logs."""

    def __init__(self) -> None:
        self._fleet_total_start   = 0
        self._fleet_func_start    = 0
        self._fleet_onsite_start  = 0
        self._fleet_depot_start   = 0
        self.reset()

    def reset(self) -> None:
        self._ep_func_pickups     = 0
        self._ep_func_deliveries  = 0
        self._ep_onsite_repairs   = 0
        self._ep_depot_pickups    = 0
        self._ep_depot_deliveries = 0
        self._ep_depot_visits     = 0
        self._ep_restored_onsite  = 0
        self._ep_restored_depot   = 0

    def capture_fleet_start(self, state) -> None:
        all_bikes = list(state.get_all_bikes())
        n_onsite  = sum(1 for b in all_bikes if getattr(b, "damage_status", None) == "onsite")
        n_depot   = sum(1 for b in all_bikes if getattr(b, "damage_status", None) == "depot")
        depot_q   = sum(len(bl) for d in state.get_depots() for _, bl in d.in_repair)
        n_depot  += depot_q
        total     = len(all_bikes) + depot_q
        self._fleet_total_start  = total
        self._fleet_onsite_start = n_onsite
        self._fleet_depot_start  = n_depot
        self._fleet_func_start   = max(total - n_onsite - n_depot, 0)

    def log_decision(self, row: dict) -> None:
        """Called by LinearVFAPolicy._log_to_run_logger() at every VFA decision."""
        self._ep_func_pickups     += int(row.get("functional_pickups", 0))
        self._ep_func_deliveries  += int(row.get("functional_deliveries", 0))
        self._ep_onsite_repairs   += int(row.get("onsite_repairs", 0))
        self._ep_depot_pickups    += int(row.get("depot_pickups", 0))
        self._ep_depot_deliveries += int(row.get("depot_deliveries", 0))
        self._ep_depot_visits     += 1 if row.get("is_at_depot", False) else 0
        lfq = int(row.get("load_from_queue", 0))
        self._ep_restored_onsite  += int(row.get("onsite_repairs", 0))
        self._ep_restored_depot   += lfq


def _collect_episode_stats(
    simulator,
    stats: _EpisodeStatsCollector,
    vfa_policy,
    alpha: float,
    alpha_start: float,
    epsilon: float,
    epsilon_start: float,
) -> dict:
    """Build the per-episode stats row — all counts are VFA-phase only (warmup subtracted)."""
    m  = simulator.state.metrics
    ag = lambda key: m.get_aggregate_value(key) or 0

    # Warmup snapshots (zeroed by reset_episode, set at first VFA decision)
    wu_starv   = getattr(vfa_policy, "warmup_starvations_snapshot",       0)
    wu_cong    = getattr(vfa_policy, "warmup_congestions_snapshot",        0)
    wu_trips   = getattr(vfa_policy, "warmup_trips_snapshot",              0)
    wu_short   = getattr(vfa_policy, "warmup_short_congestions_snapshot",  0)
    wu_dep     = getattr(vfa_policy, "warmup_bike_departures_snapshot",    0)
    wu_arr     = getattr(vfa_policy, "warmup_bike_arrivals_snapshot",      0)
    wu_onsite_f = getattr(vfa_policy, "warmup_onsite_failures_snapshot",   0)
    wu_depot_f  = getattr(vfa_policy, "warmup_depot_failures_snapshot",    0)

    all_bikes  = list(simulator.state.get_all_bikes())
    depot_q    = sum(len(bl) for d in simulator.state.get_depots() for _, bl in d.in_repair)
    total_end  = len(all_bikes) + depot_q
    n_onsite_e = sum(1 for b in all_bikes if getattr(b, "damage_status", None) == "onsite")
    n_depot_e  = sum(1 for b in all_bikes if getattr(b, "damage_status", None) == "depot") + depot_q
    n_func_e   = max(total_end - n_onsite_e - n_depot_e, 0)
    t_end      = total_end or 1
    t_start    = stats._fleet_total_start or 1

    return {
        "alpha":                       round(alpha, 6),
        "alpha_initial":               alpha_start,
        "epsilon":                     round(epsilon, 6),
        #"epsilon_initial":             epsilon_start,
        #"starvations":                 max(0, ag("starvations")      - wu_starv),
        #"long_congestions":            max(0, ag("long congestions") - wu_cong),
        #"short_congestions":           max(0, ag("short congestions")- wu_short),
        #"total_trips":                 max(0, ag("trips")            - wu_trips),
        #"bike_departures":             max(0, ag("bike departure")   - wu_dep),
        #"bike_arrivals":               max(0, ag("bike arrival")     - wu_arr),
        #"total_onsite_repairs":        stats._ep_onsite_repairs,
       # "total_depot_pickups":         stats._ep_depot_pickups,
        #"total_depot_deliveries":      stats._ep_depot_deliveries,
        #"total_depot_visits":          stats._ep_depot_visits,
        #"total_functional_pickups":    stats._ep_func_pickups,
        #"total_functional_deliveries": stats._ep_func_deliveries,
        #"broken_ratio_start_onsite":   round(stats._fleet_onsite_start / t_start, 4),
       # "broken_ratio_start_depot":    round(stats._fleet_depot_start  / t_start, 4),
        #"functional_ratio_start":      round(stats._fleet_func_start   / t_start, 4),
        #"broken_ratio_end_onsite":     round(n_onsite_e / t_end, 4),
        #"broken_ratio_end_depot":      round(n_depot_e  / t_end, 4),
        #"functional_ratio_end":        round(n_func_e   / t_end, 4),
        #"new_breakdowns_onsite":       max(0, ag("onsite_failures") - wu_onsite_f),
        #"new_breakdowns_depot":        max(0, ag("depot_failures")  - wu_depot_f),
        #"restored_onsite":             stats._ep_restored_onsite,
        #"restored_depot":              stats._ep_restored_depot,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metric helper
# ─────────────────────────────────────────────────────────────────────────────

def _service_level(simulator, vfa_policy) -> float:
    """
    Computes the service level strictly for the LEARNING phase.
    It subtracts the starvations, congestions, and trips that occurred during the warm-up.
    """
    m = simulator.state.metrics
    
    # Total metrics at the end of Day 14
    total_trips = m.get_aggregate_value("trips") or 1
    total_starv = m.get_aggregate_value("starvations") or 0
    total_cong = m.get_aggregate_value("long congestions") or 0

    # --- FIXED: Use the policy snapshots, not the live calculator ---
    warmup_starv = getattr(vfa_policy, 'warmup_starvations_snapshot', 0)
    warmup_cong = getattr(vfa_policy, 'warmup_congestions_snapshot', 0)
    warmup_trips = getattr(vfa_policy, 'warmup_trips_snapshot', 0) 

    # Isolate the VFA's true performance (Days 5-14)
    vfa_starv = max(0, total_starv - warmup_starv)
    vfa_cong = max(0, total_cong - warmup_cong)
    vfa_trips = max(1, total_trips - warmup_trips) 

    return 1.0 - (vfa_starv + vfa_cong) / vfa_trips


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    num_episodes  : int   = NUM_EPISODES,
    save_path     : Optional[Path] = None,
    seed_offset   : int   = 0,
    instance_name : str   = INSTANCE_NAME,
    active_features: list | None = None,
    gamma         : float = GAMMA,
    alpha_start   : float = ALPHA_START,
    epsilon_start : float = EPSILON_START,
    epsilon_end   : float = EPSILON_END,
    td_lambda     : float = TD_LAMBDA,
    include_bias  : bool = BIAS_FEATURE_ENABLED,
    weight_starvation : float = -1.0,
    weight_congestion : float = -1.0,
    weight_fleet_degradation : float = -0.0,
    weight_trip_served : float = 0.0,
    not_at_depot_at_end_penalty : float = 0.0,
    functional_bikes_at_end_penalty : float = 0.0,
    use_reward_centering: bool = False,
    reward_centering_beta: float = 0.01,
    use_terminal_update: bool = True,
    use_batch_td_clip: bool = False,
    batch_td_clip_value: float = 10.0,
    use_online_td_updates: bool = False,
    transition_update_interval: int = TRANSITION_UPDATE_INTERVAL,
    use_feature_scale_diagnostics: bool = True,
    diagnostic_every_n_episodes: int = 25,
    initial_bias: float | None = -2.5,
    use_feature_centering: bool = False,
    feature_centering_beta: float = 0.01,
    log_candidate_diagnostics: bool = False,
    log_greedy_comparison: bool = False,
) -> LinearVFAPolicy:

    """
    Run the full episodic VFA training loop.

    Args:
        num_episodes:   Number of training episodes (default 200).
        save_path:      Where to write the final .pkl model.
                        Auto-generated from timestamp if None.
        seed_offset:    Episode i uses random seed  seed_offset + i,
                        so different runs don't share the same trajectory.
        instance_name:  Simulator instance (e.g. "TD_W34_old").
        active_features: Optional subset of canonical VFA feature names to use.

    Returns:
        The trained LinearVFAPolicy (θ frozen after training).
    """
    
    # Harmonic decay: α_t = α_0 / (1 + c·t)
    # Satisfies Robbins-Monro conditions (Σα→∞, Σα²<∞), unlike geometric decay.
    # c chosen so α reaches ~10% of α_start by the final episode.
    harmonic_c = 9.0 / max(num_episodes - 1, 1)
    alpha_end  = alpha_start / (1.0 + harmonic_c * (num_episodes - 1))

    # <-- Epsilon decay logic -->
    epsilon_decay = (epsilon_end / epsilon_start) ** (1.0 / max(num_episodes - 1, 1)) if epsilon_start > 0 else 1.0
    
    
    # ── Header ────────────────────────────────────────────────────────────────
    print("=" * 72)
    print("  OFFLINE VFA TRAINING  -  Time-Indexed Linear VFA")
    print("=" * 72)
    print(f"  Episodes          : {num_episodes}")
    print(
        f"  Episode duration  : {EPISODE_DAYS} days  "
        f"(warm-up = {WARMUP_DAYS}d,  learning = {LEARNING_DAYS}d)"
    )
    print(f"  alpha schedule      : {alpha_start:.5f} -> {alpha_end:.5f}")
    print(f"  epsilon schedule    : {epsilon_start:.5f} -> {epsilon_end:.5f}")
    if transition_update_interval > 0:
        print(f"  batch update cadence: every {transition_update_interval} transitions + episode remainder")
    else:
        print("  batch update cadence: once per episode")
    print(f"  bias feature        : {'enabled' if include_bias else 'disabled'}")
    print(f"  gamma               : {gamma}")
    print(f"  reward centering    : {use_reward_centering} (beta={reward_centering_beta})")
    print(f"  terminal update     : {use_terminal_update}")
    print(f"  batch TD clipping   : {use_batch_td_clip} (clip={batch_td_clip_value})")
    print(f"  online TD updates   : {use_online_td_updates}")
    print(f"  feature scale diag  : {use_feature_scale_diagnostics} (every {diagnostic_every_n_episodes} batch updates)")
    print(f"  TD lambda           : {td_lambda:g} ({'TD(lambda)' if td_lambda > 0.0 else 'TD(0)'})")
    print(f"  initial bias        : {initial_bias}")
    print(f"  feature centering   : {use_feature_centering} (beta={feature_centering_beta})")
    print(f"  candidate diag      : {log_candidate_diagnostics}")
    print(f"  greedy comparison   : {log_greedy_comparison}")
    print(f"  Instance          : {instance_name}")
    print("=" * 72 + "\n")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if active_features is not None:
        active_features = list(active_features)
        if include_bias and "bias" not in active_features:
            active_features = ["bias"] + active_features
        elif not include_bias:
            active_features = [f for f in active_features if f != "bias"]

    # ── Initialise VFA policy  (θ persists across ALL episodes) ───────────────
    reward_config = RewardConfig(
        weight_starvation=weight_starvation,
        weight_congestion=weight_congestion,
        weight_fleet_degradation=weight_fleet_degradation,
        weight_trip_served=weight_trip_served,
        not_at_depot_at_end_penalty=not_at_depot_at_end_penalty,
        functional_bikes_at_end_penalty=functional_bikes_at_end_penalty,
        use_reward_centering=use_reward_centering,
        reward_centering_beta=reward_centering_beta,
    )

    print(f" RewardConfig: weight_starvation={reward_config.weight_starvation}, weight_congestion={reward_config.weight_congestion}, weight_fleet_degradation={reward_config.weight_fleet_degradation}, weight_trip_served={reward_config.weight_trip_served}, use_reward_centering={reward_config.use_reward_centering}, reward_centering_beta={reward_config.reward_centering_beta}")

    if initial_bias not in (None, 0.0) and not include_bias:
        raise ValueError("initial_bias requires the bias feature to be enabled")
    if not 0.0 <= feature_centering_beta <= 1.0:
        raise ValueError(f"feature_centering_beta must be in [0, 1], got {feature_centering_beta}")
    if transition_update_interval < 0:
        raise ValueError(f"transition_update_interval must be >= 0, got {transition_update_interval}")
    if use_online_td_updates and transition_update_interval > 0:
        raise ValueError("use_online_td_updates and transition_update_interval are mutually exclusive")

    vfa_policy = LinearVFAPolicy(
        active_features=active_features,
        alpha         = alpha_start,
        gamma         = gamma,
        learning_mode = True,
        seed          = 42,
        maintenance_enabled=ENABLE_COMPONENT_FAILURES,
        logistics_enabled=LOGISTICS_ENABLED,
        reward_calculator=RewardCalculator(config=reward_config, gamma=gamma),
        use_bias_feature=include_bias,
        use_terminal_update=use_terminal_update,
        use_batch_td_clip=use_batch_td_clip,
        batch_td_clip_value=batch_td_clip_value,
        use_online_td_updates=use_online_td_updates,
        transition_update_interval=transition_update_interval,
        initial_bias=initial_bias,
        use_feature_centering=use_feature_centering,
        feature_centering_beta=feature_centering_beta,
    )
    
    if not 0.0 <= td_lambda <= 1.0:
        raise ValueError(f"td_lambda must be in [0, 1], got {td_lambda}")

    # --- TD(λ); td_lambda=0.0 keeps plain TD(0) ---
    vfa_policy.use_td_lambda = td_lambda > 0.0
    vfa_policy.td_lambda = td_lambda
    VFA_DEBUG_FLAGS["check7_feature_scale"] = use_feature_scale_diagnostics
    VFA_DEBUG_FLAGS["every_n_episodes"] = max(1, int(diagnostic_every_n_episodes))

    vfa_policy.log_candidate_diagnostics = log_candidate_diagnostics
    vfa_policy.log_greedy_comparison = log_greedy_comparison
    vfa_policy._collect_phis_inside_batch_update = transition_update_interval > 0
    if vfa_policy._collect_phis_inside_batch_update and getattr(vfa_policy, "_all_phis_for_corr", None) is None:
        vfa_policy._all_phis_for_corr = []

    # --- EXPERIENCE REPLAY TOGGLE ---
    # Set to True to use Mini-Batch SGD at every timestep
    # Set to False to keep your well-working Episodic Synchronous Batching
    vfa_policy.use_experience_replay = False
    vfa_policy.mini_batch_size = 32
    ###############################################################################
    # [Check 1] Feature ↔ theta mapping at startup
    print("\n[CHK1] Feature-Theta Mapping:")
    for i, name in enumerate(vfa_policy.FEATURE_NAMES):
        print(f"  {i:2d}  {name:<40}  θ_init={vfa_policy.theta[i]:+.6f}")
    print(f"  phi_len={vfa_policy.N_FEATURES}  theta_len={len(vfa_policy.theta)}")
    assert vfa_policy.N_FEATURES == len(vfa_policy.theta), "[CHK1] MISMATCH: phi length != theta length!"

    #greedy_policy = GreedyPolicy()
    greedy_policy = GreedyMaintenancePolicy()

    stats_collector = _EpisodeStatsCollector()
    vfa_policy.logger = stats_collector # type: ignore

    # Warm-up ends at this absolute simulation-time (minutes).
    sim_start_min   = timeInMinutes(hours=START_HOUR)
    warmup_end_time = sim_start_min + WARMUP_DAYS * 24 * 60


    service_levels: list = []
    episode_stats:  list = []
    weights_history: list = []  # Collect θ vectors per episode
    t0 = time.time()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    config_parts = []
    if include_bias:
        config_parts.append("bias")
    if use_reward_centering:
        config_parts.append(f"rc{reward_centering_beta:g}")
    if use_terminal_update:
        config_parts.append("term")
    if td_lambda > 0.0:
        config_parts.append(f"lam{td_lambda:g}")
    if use_batch_td_clip:
        config_parts.append(f"clip{batch_td_clip_value:g}")
    if use_online_td_updates:
        config_parts.append("online")
    if transition_update_interval > 0:
        config_parts.append(f"trans{transition_update_interval}")
    if use_feature_scale_diagnostics:
        config_parts.append(f"fsdiag{diagnostic_every_n_episodes}")
    if initial_bias not in (None, 0.0):
        config_parts.append(f"initb{str(initial_bias).replace('-', 'm').replace('.', 'p')}")
    if use_feature_centering:
        config_parts.append(f"fcenter{feature_centering_beta:g}")
    if log_candidate_diagnostics:
        config_parts.append("canddiag")
    if log_greedy_comparison:
        config_parts.append("gcmp")
    config_suffix = ("_" + "_".join(config_parts)) if config_parts else ""

    # Resolve weights CSV path before the episode loop so incremental writes
    # always work, even when no --save path was given.
    if save_path is not None:
        weights_csv_path = Path(str(save_path).replace(".pkl", "_weights_evolution.csv"))
    else:
        weights_csv_path = SAVE_DIR / f"vfa_training_{instance_name}{config_suffix}_{run_timestamp}_weights_evolution.csv"

    # ── Episode loop ──────────────────────────────────────────────────────────
    for ep in range(num_episodes):
        config        = SimulationConfig()
        stats_collector.reset()
        reset_truncation_counts()
        reset_candidate_debug_counts()

        ###########
        # ── Calculate current dynamic parameters ─────────────────────────
        current_alpha = alpha_start / (1.0 + harmonic_c * ep)
        current_epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** ep)) if epsilon_start > 0 else 0.0

        print(f"\n[DEBUG] Episode {ep+1}: Calculated alpha={current_alpha:.5f}, epsilon={current_epsilon:.5f}")
        
        # STRICT OVERRIDE: Force the policy to use this exact step-size
        vfa_policy.alpha = current_alpha
        vfa_policy.epsilon = current_epsilon
        
        print(f"\n{'='*50}")
        print(f"EPISODE {ep+1}/{num_episodes} | Alpha: {current_alpha:.5f} | Epsilon: {current_epsilon:.5f}")
        print(f"{'='*50}")
        ############
        vfa_policy._comparison_episode = ep + 1

        # ── Build episode policy ───────────────────────────────────────────
        # EpisodeTrainingPolicy:
        #   • calls vfa_policy.reset_episode() to clear per-episode TD state
        #   • routes to GreedyPolicy  while  state.time < warmup_end_time
        #   • routes to VFAPolicy     once   state.time >= warmup_end_time
        episode_policy = EpisodeTrainingPolicy(
            vfa_policy      = vfa_policy,
            warmup_policy   = greedy_policy,
            warmup_end_time = warmup_end_time,
        )

        # ── Fresh simulator, same θ ────────────────────────────────────────
        simulator = run_simulation(
            seed          = seed_offset + ep,
            policy        = episode_policy,
            duration      = 24 * EPISODE_DAYS,
            num_vehicles  = NUM_VEHICLES,
            instance_name = instance_name,
            config        = config,
        )

        vfa_policy.add_terminal_update(simulator.state)

        # --- WRITE CSV FILES INTO FOLDERS ---
        # Only use for specific debug as it takes up too much space
        '''filename = f"run_{run_timestamp}/ep_{ep:03d}/vfa_training_{instance_name}.csv"
        
        write_simulation_outputs(
            simulator=simulator,
            filename=filename,
            seed=seed_offset + ep,
            policy=episode_policy,
            duration=24 * EPISODE_DAYS,
            num_vehicles=NUM_VEHICLES,
            append_to_results=False
        )'''
        # ----------------------------

        sl = _service_level(simulator, vfa_policy)
        service_levels.append(sl)

        if log_greedy_comparison:
            cmp_path = str(save_path).replace(".pkl", "__vfa_comparison.csv") if save_path else str(SAVE_DIR / "greedy_vs_vfa_comparison.csv")
            vfa_policy.flush_comparison_log(cmp_path)
        if log_candidate_diagnostics:
            cand_path = str(save_path).replace(".pkl", "__candidate_diagnostics.csv") if save_path else str(SAVE_DIR / "candidate_diagnostics.csv")
            vfa_policy.flush_candidate_diagnostics(cand_path)

        ep_stat = _collect_episode_stats(
            simulator, stats_collector, vfa_policy,
            current_alpha, alpha_start, current_epsilon, epsilon_start,
        )
        ep_stat["gamma"] = gamma
        ep_stat["transition_update_interval"] = transition_update_interval
        episode_stats.append(ep_stat)

        remainder_batch_updates = 0
        if use_online_td_updates:
            vfa_policy.flush_online_update_diagnostics(ep + 1)
        elif transition_update_interval > 0:
            # Periodic transition batches are applied inside td_update().
            # Flush the episode remainder so batches never mix across episodes.
            if getattr(vfa_policy, 'batch_buffer', None):
                vfa_policy.apply_batch_update()
                remainder_batch_updates = 1
        else:
            # Original episode-batch mode: update once after each episode.
            # Collect phis for correlation analysis BEFORE the buffer clears
            if getattr(vfa_policy, '_all_phis_for_corr', None) is None:
                vfa_policy._all_phis_for_corr = []
            if getattr(vfa_policy, 'batch_buffer', None):
                vfa_policy._all_phis_for_corr.extend([item[0] for item in vfa_policy.batch_buffer])
                
            vfa_policy.apply_batch_update()
            remainder_batch_updates = 1

        transition_batch_updates = getattr(vfa_policy, "_transition_batch_update_count", 0)
        ep_stat["transition_batch_updates"] = transition_batch_updates
        ep_stat["remainder_batch_updates"] = remainder_batch_updates
        ep_stat["batch_updates_this_episode"] = transition_batch_updates + remainder_batch_updates

        # --- MID-RUN DIAGNOSTIC LOGGING ---
        if (ep + 1) % 5 == 0 and getattr(vfa_policy, '_all_phis_for_corr', None):
            recent_phis = vfa_policy._all_phis_for_corr[-10000:]
            temp_df = pd.DataFrame(recent_phis, columns=vfa_policy.FEATURE_NAMES)
            corr = temp_df.corr(method='pearson')

            print(f"\n  [DEBUG Ep {ep + 1}] Pearson Correlation with 'squared_starvation_penalty':")
            if 'squared_starvation_penalty' in corr.columns:
                target_col = corr['squared_starvation_penalty']
                # Print features with high absolute correlation
                high_corr = target_col[abs(target_col) > 0.3].sort_values()
                for f, c in high_corr.items():
                    print(f"      {f:<35} : {c:+.3f}")
            print()
            
        weights_history.append(vfa_policy.theta.copy())  # Store θ vector for this episode

    
        formatted_theta = np.array2string(vfa_policy.theta, formatter={'float_kind':lambda x: f"{x:+.3f}"})
        trunc_summary = get_truncation_summary()
        candidate_summary = get_candidate_debug_summary()

        print(
            f"  Ep {ep + 1:3d}/{num_episodes} | "
            f"SL={sl:.4f} | "
            f"Weights: {formatted_theta} | "
            f"bridge={trunc_summary} | "
            f"candidate_debug={candidate_summary} | "
            f"t={time.time() - t0:.0f}s"
        )


        # ── Periodic checkpoint every 50 episodes ─────────────────────────
        if (ep + 1) % 50 == 0:
            if save_path is not None:
                ck_path = save_path.with_name(f"{save_path.stem}_checkpoint_ep{ep + 1:04d}{save_path.suffix}")
            else:
                ck_path = SAVE_DIR / f"vfa_checkpoint_ep{ep + 1:04d}.pkl"
            vfa_policy.save(ck_path)

        # ── Periodically update weights evolution CSV every 5 episodes ──
        if (ep + 1) % 5 == 0:
            feature_names = vfa_policy.FEATURE_NAMES
            # Get the last 5 episodes' weights, service levels, and stats
            start_idx = ep - 4 if ep >= 4 else 0
            end_idx = ep + 1
            partial_weights  = weights_history[start_idx:end_idx]
            partial_service  = service_levels[start_idx:end_idx]
            partial_stats_ep = episode_stats[start_idx:end_idx]
            partial_episodes = list(range(start_idx, end_idx))
            partial_df = pd.DataFrame(partial_weights, columns=feature_names)
            partial_df.insert(0, 'episode', partial_episodes)
            partial_df.insert(1, 'service_level', partial_service)
            stats_df = pd.DataFrame(partial_stats_ep)
            for col in stats_df.columns:
                partial_df[col] = stats_df[col].values

            # Append to CSV, write header only if file does not exist
            write_header = not weights_csv_path.exists()
            with open(weights_csv_path, 'a') as f:
                partial_df.to_csv(f, header=write_header, index=False)


    # ── Final save ────────────────────────────────────────────────────────────
    if save_path is None:
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = SAVE_DIR / f"vfa_trained_baseline_features_alpha{alpha_start}_seed{seed_offset}{config_suffix}_{ts}.pkl"

    vfa_policy.save(save_path)

    # Write any remaining weights not yet written (episodes after the last 5-episode flush)
    remainder = num_episodes % 5
    if remainder != 0:
        feature_names = vfa_policy.FEATURE_NAMES
        start_idx = (num_episodes // 5) * 5
        partial_weights   = weights_history[start_idx:]
        partial_service   = service_levels[start_idx:]
        partial_stats_ep  = episode_stats[start_idx:]
        partial_episodes  = list(range(start_idx, num_episodes))
        partial_df = pd.DataFrame(partial_weights, columns=feature_names)
        partial_df.insert(0, 'episode', partial_episodes)
        partial_df.insert(1, 'service_level', partial_service)
        stats_df = pd.DataFrame(partial_stats_ep)
        for col in stats_df.columns:
            partial_df[col] = stats_df[col].values

        write_header = not weights_csv_path.exists()
        with open(weights_csv_path, 'a') as f:
            partial_df.to_csv(f, header=write_header, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    best_ep = int(np.argmax(service_levels)) + 1

    print("\n" + "=" * 72)
    print("  TRAINING COMPLETE")
    print("=" * 72)
    print(f"  Final theta       : {np.array2string(vfa_policy.theta, precision=4)}")
    print(f"  |theta|           : {np.linalg.norm(vfa_policy.theta):.6f}")
    print(
        f"  Service level     : final = {service_levels[-1]:.4f}  |  "
        f"best = {max(service_levels):.4f}  (episode {best_ep})"
    )
    print(f"  Total time        : {elapsed / 60:.1f} min")
    print(f"  Model saved       : {save_path}")
    print(f"  Weight evolution  : {weights_csv_path}")
    print("=" * 72 + "\n")

    # Log the Feature Matrix Correlation
    if getattr(vfa_policy, '_all_phis_for_corr', None):
        print("  --- FINAL FEATURE CORRELATION MATRIX ---  ")
        phi_df = pd.DataFrame(vfa_policy._all_phis_for_corr, columns=vfa_policy.FEATURE_NAMES)
        corr_matrix = phi_df.corr(method='pearson')
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        # We selectively print correlations against negative performance features 
        # to diagnose multicollinearity with starvation.
        features_of_interest = []
        for feature in ['squared_starvation_penalty', 'fleet_broken_fraction', 'global_onsite_backlog']:
            if feature in corr_matrix.columns:
                features_of_interest.append(feature)
        
        if features_of_interest:
            print(corr_matrix[features_of_interest].round(3))
        else:
            print(corr_matrix.round(3))
        print("=" * 72 + "\n")

    return vfa_policy


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline episodic TD(0) training of a linear VFA for DSJBRMP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=NUM_EPISODES,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        metavar="PATH",
        help="Output path for the trained model (.pkl).  "
             "Default: models/vfa_trained_<timestamp>.pkl",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        metavar="OFFSET",
        help="Seed offset: episode i uses random seed = offset + i",
    )
    
    parser.add_argument(
        "--instance",
        type=str,
        default=INSTANCE_NAME,
        help="Simulator instance name",
    )
    
    parser.add_argument(
        "--gamma", 
        type=float, 
        default=GAMMA, 
        help="Discount factor for future rewards"
    )
    parser.add_argument(
        "--alpha_start",
        type=float,
        default=ALPHA_START,
        help="Initial learning rate (alpha) for TD updates"
    )
    parser.add_argument(
        "--epsilon_start",
        type=float,
        default=EPSILON_START,
        help="Initial exploration rate (epsilon) for epsilon-greedy policy"
    )
    parser.add_argument(
        "--epsilon_end",
        type=float,
        default=EPSILON_END,
        help="Final exploration rate (epsilon) for epsilon-greedy policy"
    )
    parser.add_argument(
        "--td_lambda",
        type=float,
        default=TD_LAMBDA,
        help="TD(lambda) eligibility trace parameter. Use 0.0 for TD(0).",
    )
    parser.add_argument(
        "--use_bias_feature",
        dest="include_bias",
        action="store_true",
        default=BIAS_FEATURE_ENABLED,
        help="Use the constant bias/intercept feature. Enabled by default.",
    )
    parser.add_argument(
        "--weight_starvation",
        type=float,
        default=-1.0,
        help="Reward weight for starvation events (default: -1.0)",
    )
    parser.add_argument(
        "--weight_congestion",
        type=float,
        default=-1.0,
        help="Reward weight for congestion events (default: -1.0)",
    )
    parser.add_argument(
        "--use_reward_centering",
        action="store_true",
        help="Subtract a running reward mean before TD updates",
    )
    parser.add_argument(
        "--reward_centering_beta",
        type=float,
        default=0.01,
        help="EMA step size for reward centering baseline (default: 0.01)",
    )
    parser.add_argument(
        "--use_terminal_update",
        action="store_true",
        default=True,
        help="Append terminal transition with zero bootstrap at the end of each episode",
    )
    parser.add_argument(
        "--use_batch_td_clip",
        action="store_true",
        help="Clip TD errors inside the episode batch update",
    )
    parser.add_argument(
        "--batch_td_clip_value",
        type=float,
        default=10.0,
        help="Absolute TD error clip used when --use_batch_td_clip is set",
    )
    parser.add_argument(
        "--use_online_td_updates",
        action="store_true",
        help="Apply TD updates immediately at every transition instead of episode-batch updates.",
    )
    parser.add_argument(
        "--transition_update_interval",
        type=int,
        default=TRANSITION_UPDATE_INTERVAL,
        help="Apply one mean-gradient TD update every N buffered transitions. 0 keeps episode-batch updates.",
    )
    parser.add_argument(
        "--use_feature_scale_diagnostics",
        action="store_true",
        default=True,
        help="Print per-feature scale diagnostics during batch updates",
    )
    parser.add_argument(
        "--diagnostic_every_n_episodes",
        type=int,
        default=25,
        help="Frequency for heavy diagnostics such as feature scale reports",
    )
    parser.add_argument(
        "--initial_bias",
        type=float,
        default=-2.5,
        help="Initial value for the bias/intercept weight. Requires the bias feature to be enabled.",
    )
    parser.add_argument(
        "--use_feature_centering",
        action="store_true",
        help="Center non-bias VFA features with a running candidate-set mean before value/TD updates.",
    )
    parser.add_argument(
        "--feature_centering_beta",
        type=float,
        default=0.01,
        help="EMA step size for feature centering when --use_feature_centering is set.",
    )
    parser.add_argument(
        "--log_candidate_diagnostics",
        action="store_true",
        help="Write per-decision candidate value spread diagnostics to CSV.",
    )
    parser.add_argument(
        "--log_greedy_comparison",
        action="store_true",
        help="Write VFA-vs-greedy-maintenance decision comparison diagnostics to CSV.",
    )

    args = parser.parse_args()
    if args.transition_update_interval < 0:
        raise ValueError(f"--transition_update_interval must be >= 0, got {args.transition_update_interval}")
    if args.use_online_td_updates and args.transition_update_interval > 0:
        raise ValueError("--use_online_td_updates and --transition_update_interval are mutually exclusive")

    selected_features = []
    #IF training with specific features: wirte features inside list and pass to active_features

    train(
        num_episodes      = args.episodes,
        save_path         = Path(args.save) if args.save else None,
        seed_offset       = args.seed,
        instance_name     = args.instance,
        active_features   = None, #OR: selected_features
        gamma             = args.gamma,
        alpha_start       = args.alpha_start,
        epsilon_start     = args.epsilon_start,
        epsilon_end       = args.epsilon_end,
        td_lambda         = args.td_lambda,
        include_bias      = args.include_bias,
        weight_starvation = args.weight_starvation,
        weight_congestion = args.weight_congestion,
        use_reward_centering = args.use_reward_centering,
        reward_centering_beta = args.reward_centering_beta,
        use_terminal_update = args.use_terminal_update,
        use_batch_td_clip = args.use_batch_td_clip,
        batch_td_clip_value = args.batch_td_clip_value,
        use_online_td_updates = args.use_online_td_updates,
        transition_update_interval = args.transition_update_interval,
        use_feature_scale_diagnostics = args.use_feature_scale_diagnostics,
        diagnostic_every_n_episodes = args.diagnostic_every_n_episodes,
        initial_bias = args.initial_bias,
        use_feature_centering = args.use_feature_centering,
        feature_centering_beta = args.feature_centering_beta,
        log_candidate_diagnostics = args.log_candidate_diagnostics,
        log_greedy_comparison = args.log_greedy_comparison,
    )
