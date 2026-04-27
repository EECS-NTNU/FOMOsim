#!/usr/bin/env python3
"""
train_vfa.py  -  Offline Episodic Training of Time-Indexed Linear VFA

Architecture
─────────────
Hybrid "Horizontal" ADP (Ulmer 2020 / Brinkmann 2019-2020):
  - VFA is trained completely offline via episodic TD(0) simulation.
  - Once frozen, the VFA acts as a tail-value estimator inside an online
    Rollout Algorithm (implemented separately).

Episode structure  (14-day simulation per episode)
──────────────────────────────────────────────────
  Days 1 - 4   Warm-up   : GreedyPolicy drives the system.
                            No TD updates → builds up a realistic
                            "messy" state without biasing θ.
  Days 5 - 14  Learning  : LinearVFAPolicy.
                            TD(0) updates occur at every vehicle decision.

Usage
─────
    python train_vfa.py
    python train_vfa.py --episodes 200 --save models/my_vfa.pkl
    python train_vfa.py --episodes 50 --seed 100 --instance TD_W34_37
"""

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
from policies.do_nothing_policy import DoNothing
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy, EpisodeTrainingPolicy
from policies.sjovik_sund.vfa.vfa_features import get_feature_names as _get_feature_names
from policies.sjovik_sund.run_simulation_ingvild import run_simulation, SimulationConfig, write_simulation_outputs
from policies.sjovik_sund.mdp.reward import RewardConfig, RewardCalculator
from settings import ENABLE_COMPONENT_FAILURES


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters  (all subject to change)
# ─────────────────────────────────────────────────────────────────────────────

NUM_EPISODES  : int   = 200      # total training episodes

EPISODE_DAYS  : int   = 14       # days per episode (total)
WARMUP_DAYS   : int   = 2         # greedy warm-up, no TD updates
LEARNING_DAYS : int   = 12        # VFA + TD(0)  (days 5 – 14)

# --- Learning Rate (Alpha) & Exploration (Epsilon) ---
ALPHA_START   : float = 0.1   # initial alpha for TD updates, will be overwritten in case of argument passing
EPSILON_START : float = 0.2     # initial exploration rate
EPSILON_END   : float = 0.01    # final exploration rate

GAMMA         : float = 0.99      # discount factor

# ── Feature configuration ──────────────────────────────────────────────────────
LOGISTICS_ENABLED : bool = False  # Enable Pillar 4: Spatial & Logistic Constraints features
N_FEATURES    : int   = len(_get_feature_names(ENABLE_COMPONENT_FAILURES, logistics_enabled=LOGISTICS_ENABLED, demand_horizon_enabled=True))  # auto-synced with vfa_features.py

INSTANCE_NAME : str   = "TD_W34_old" #"OS_W31"
NUM_VEHICLES  : int   = 1
START_HOUR    : int   = 5         # simulation clock starts at 00:00

# Where to save checkpoints and the final model
SAVE_DIR = Path(__file__).parent / "models"


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
    weight_starvation : float = -1.0,
    weight_congestion : float = -1.0,
) -> LinearVFAPolicy:
    
    # BATCH SIZE configuration
    batch_size = 1  # Number of episodes to run before applying batch updates to θ
    
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
    print(f"  gamma               : {gamma}")
    print(f"  Instance          : {instance_name}")
    print("=" * 72 + "\n")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)


    # ── Initialise VFA policy  (θ persists across ALL episodes) ───────────────
    reward_config = RewardConfig(
        weight_starvation=weight_starvation,
        weight_congestion=weight_congestion,
    )
    
    print(f" RewardConfig: weight_starvation={reward_config.weight_starvation}, weight_congestion={reward_config.weight_congestion}")

    vfa_policy = LinearVFAPolicy(
        active_features=active_features,
        n_features=len(active_features) if active_features is not None else N_FEATURES,
        alpha         = alpha_start,
        gamma         = gamma,
        learning_mode = True,
        seed          = 42,
        maintenance_enabled=ENABLE_COMPONENT_FAILURES,
        logistics_enabled=LOGISTICS_ENABLED,
        reward_calculator=RewardCalculator(config=reward_config, gamma=gamma),
    )
    
    # --- EXPERIENCE REPLAY TOGGLE ---
    # Set to True to use Mini-Batch SGD at every timestep 
    # Set to False to keep your well-working Episodic Synchronous Batching
    vfa_policy.use_experience_replay = False
    vfa_policy.mini_batch_size = 32
    ###############################################################################
    greedy_policy = GreedyPolicy()

    # Warm-up ends at this absolute simulation-time (minutes).
    sim_start_min   = timeInMinutes(hours=START_HOUR)
    warmup_end_time = sim_start_min + WARMUP_DAYS * 24 * 60   # e.g. 420 + 5760


    service_levels: list = []
    weights_history: list = []  # Collect θ vectors per episode
    t0 = time.time()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Prepare weights evolution CSV path
    weights_csv_path = None
    if save_path is not None:
        weights_csv_path = Path(str(save_path).replace(".pkl", "_weights_evolution.csv"))
    # If save_path is None, will be set at the end as before

    # ── Episode loop ──────────────────────────────────────────────────────────
    for ep in range(num_episodes):
        config        = SimulationConfig()
        
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
        
        # Apply synchronous batch update every 'batch_size' episodes
        if (ep + 1) % batch_size == 0 or (ep + 1) == num_episodes:
            vfa_policy.apply_batch_update()
            
        weights_history.append(vfa_policy.theta.copy())  # Store θ vector for this episode

    
        formatted_theta = np.array2string(vfa_policy.theta, formatter={'float_kind':lambda x: f"{x:+.3f}"})
        
        print(
            f"  Ep {ep + 1:3d}/{num_episodes} | "
            f"SL={sl:.4f} | "
            f"Weights: {formatted_theta} | "
            f"t={time.time() - t0:.0f}s"
        )


        # ── Periodic checkpoint every 50 episodes ─────────────────────────
        if (ep + 1) % 50 == 0:
            ck_path = SAVE_DIR / f"vfa_checkpoint_ep{ep + 1:04d}.pkl"
            vfa_policy.save(ck_path)

        # ── Periodically update weights evolution CSV every 5 episodes ──
        if (ep + 1) % 5 == 0:
            # Determine CSV path if not already set
            if weights_csv_path is None and save_path is not None:
                weights_csv_path = Path(str(save_path).replace(".pkl", "_weights_evolution.csv"))
            elif weights_csv_path is None:
                # If save_path is None, skip writing until final
                continue

            feature_names = vfa_policy.FEATURE_NAMES
            # Get the last 5 episodes' weights and service levels
            start_idx = ep - 4 if ep >= 4 else 0
            end_idx = ep + 1
            partial_weights = weights_history[start_idx:end_idx]
            partial_service = service_levels[start_idx:end_idx]
            partial_episodes = list(range(start_idx, end_idx))
            partial_df = pd.DataFrame(partial_weights, columns=feature_names)
            partial_df.insert(0, 'episode', partial_episodes)
            partial_df.insert(1, 'service_level', partial_service)

            # Append to CSV, write header only if file does not exist
            assert weights_csv_path is not None
            write_header = not weights_csv_path.exists()
            with open(weights_csv_path, 'a') as f:
                partial_df.to_csv(f, header=write_header, index=False)


    # ── Final save ────────────────────────────────────────────────────────────
    if save_path is None:
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = SAVE_DIR / f"vfa_trained_baseline_features_alpha{alpha_start}_seed{seed_offset}_{ts}.pkl"
        weights_csv_path = Path(str(save_path).replace(".pkl", "_weights_evolution.csv"))

    vfa_policy.save(save_path)

    # Write any remaining weights not yet written (if num_episodes not divisible by 5)
    if len(weights_history) % 5 != 0:
        feature_names = vfa_policy.FEATURE_NAMES
        start_idx = (num_episodes // 5) * 5
        partial_weights = weights_history[start_idx:]
        partial_service = service_levels[start_idx:]
        partial_episodes = list(range(start_idx, num_episodes))
        partial_df = pd.DataFrame(partial_weights, columns=feature_names)
        partial_df.insert(0, 'episode', partial_episodes)
        partial_df.insert(1, 'service_level', partial_service)
        
        assert weights_csv_path is not None
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
        "--weight_starvation",
        type=float,
        default=-1.0,
        help="Reward weight for starvation events (default: -1.0)",
    )
    parser.add_argument(
        "--weight_congestion",
        type=float,
        default=-0.7,
        help="Reward weight for congestion events (default: -0.7)",
    )

    args = parser.parse_args()

    train(
        num_episodes      = args.episodes,
        save_path         = Path(args.save) if args.save else None,
        seed_offset       = args.seed,
        instance_name     = args.instance,
        gamma             = args.gamma,
        alpha_start       = args.alpha_start,
        epsilon_start     = args.epsilon_start,
        epsilon_end       = args.epsilon_end,
        weight_starvation = args.weight_starvation,
        weight_congestion = args.weight_congestion,
    )
