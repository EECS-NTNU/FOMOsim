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
  Days 5 - 14  Learning  : LinearVFAPolicy with Boltzmann exploration.
                            TD(0) updates occur at every vehicle decision.

Boltzmann temperature τ decays exponentially over NUM_EPISODES episodes:
    τ_ep = TAU_START x (TAU_END / TAU_START)^(ep / (NUM_EPISODES - 1))

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

# ── Workspace root on sys.path ────────────────────────────────────────────────
WORKSPACE_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(WORKSPACE_ROOT))

from helpers import timeInMinutes
from policies.greedy_policy import GreedyPolicy
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy, EpisodeTrainingPolicy
from policies.sjovik_sund.vfa.vfa_features import get_feature_names as _get_feature_names
from policies.sjovik_sund.run_simulation_ingvild import run_simulation, SimulationConfig, write_simulation_outputs
from settings import ENABLE_COMPONENT_FAILURES


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters  (all subject to change)
# ─────────────────────────────────────────────────────────────────────────────

NUM_EPISODES  : int   = 200       # total training episodes

EPISODE_DAYS  : int   = 14        # days per episode (total)
WARMUP_DAYS   : int   = 4         # greedy warm-up, no TD updates
LEARNING_DAYS : int   = 10        # VFA + Boltzmann + TD(0)  (days 5 – 14)

TAU_START     : float = 5.0       # initial Boltzmann temperature
TAU_END       : float = 0.1       # final   Boltzmann temperature
# Exponential decay factor (computed once; re-used every episode)
TAU_DECAY     : float = (TAU_END / TAU_START) ** (1.0 / max(NUM_EPISODES - 1, 1))

ALPHA         : float = 0.01      # TD learning rate
GAMMA         : float = 0.99      # discount factor

# ── Feature configuration ──────────────────────────────────────────────────────
SHIFT_TIMING_ENABLED : bool = False  # Enable end-of-shift anticipatory features
N_FEATURES    : int   = len(_get_feature_names(ENABLE_COMPONENT_FAILURES, shift_timing_enabled=SHIFT_TIMING_ENABLED))  # auto-synced with vfa_features.py

INSTANCE_NAME : str   = "TD_W34_old"
NUM_VEHICLES  : int   = 1
START_HOUR    : int   = 5         # simulation clock starts at 00:00

# Where to save checkpoints and the final model
SAVE_DIR = Path(__file__).parent / "models"


# ─────────────────────────────────────────────────────────────────────────────
# Metric helper
# ─────────────────────────────────────────────────────────────────────────────

def _service_level(simulator) -> float:
    """
    Service level = 1 - (starvations + congestion) / total_trip_requests.

    Returns 0.0 if no trips were generated (e.g. very short test run).
    """
    m      = simulator.state.metrics
    trips  = m.get_aggregate_value("trips")   or 1
    starv  = m.get_aggregate_value("starvations")    or 0
    cong  = m.get_aggregate_value("long congestions")      or 0
    return 1.0 - (starv + cong) / max(trips, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    num_episodes  : int   = NUM_EPISODES,
    save_path     : Path  = None,
    seed_offset   : int   = 0,
    instance_name : str   = INSTANCE_NAME,
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

    Returns:
        The trained LinearVFAPolicy (θ frozen after training).
    """
    # ── Header ────────────────────────────────────────────────────────────────
    print("=" * 72)
    print("  OFFLINE VFA TRAINING  -  Time-Indexed Linear VFA")
    print("=" * 72)
    print(f"  Episodes          : {num_episodes}")
    print(
        f"  Episode duration  : {EPISODE_DAYS} days  "
        f"(warm-up = {WARMUP_DAYS}d,  learning = {LEARNING_DAYS}d)"
    )
    print(
        f"  tau schedule        : {TAU_START:.2f}  ->  {TAU_END:.2f}  "
        f"(decay per episode = {TAU_DECAY:.6f})"
    )
    print(f"  alpha / gamma       : {ALPHA} / {GAMMA}")
    print(f"  Instance          : {instance_name}")
    print("=" * 72 + "\n")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Initialise VFA policy  (θ persists across ALL episodes) ───────────────
    vfa_policy = LinearVFAPolicy(
        n_features    = N_FEATURES,
        alpha         = ALPHA,
        gamma         = GAMMA,
        tau           = TAU_START,
        learning_mode = True,
        seed          = 42,
        maintenance_enabled=ENABLE_COMPONENT_FAILURES,
        shift_timing_enabled=SHIFT_TIMING_ENABLED,
    )

    greedy_policy = GreedyPolicy()
    #config        = SimulationConfig()

    # Warm-up ends at this absolute simulation-time (minutes).
    # The simulator clock starts at START_HOUR × 60 (e.g. 420 min = 07:00).
    sim_start_min   = timeInMinutes(hours=START_HOUR)
    warmup_end_time = sim_start_min + WARMUP_DAYS * 24 * 60   # e.g. 420 + 5760

    service_levels: list = []
    weights_history: list = []  # Collect θ vectors per episode
    t0 = time.time()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # ── Episode loop ──────────────────────────────────────────────────────────
    for ep in range(num_episodes):
        config        = SimulationConfig()
        
        # ── Boltzmann temperature for this episode ─────────────────────────
        tau = TAU_START * (TAU_DECAY ** ep)
        vfa_policy.set_temperature(tau)

        # ── Build episode policy ───────────────────────────────────────────
        # EpisodeTrainingPolicy:
        #   • calls vfa_policy.reset_episode() to clear per-episode TD state
        #   • routes to GreedyPolicy  while  state.time < warmup_end_time
        #   • routes to VFAPolicy     once   state.time >= warmup_end_time
        episode_policy = EpisodeTrainingPolicy(
            vfa_policy      = vfa_policy,
            greedy_policy   = greedy_policy,
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

        # --- NEW: WRITE CSV FILES INTO FOLDERS ---
        # Notice the forward slashes (/)! This tells the system to make folders.
        filename = f"run_{run_timestamp}/ep_{ep:03d}/vfa_training_{instance_name}.csv"
        
        write_simulation_outputs(
            simulator=simulator,
            filename=filename,
            seed=seed_offset + ep,
            policy=episode_policy,
            duration=24 * EPISODE_DAYS,
            num_vehicles=NUM_VEHICLES,
            append_to_results=False
        )
        # ----------------------------

        sl = _service_level(simulator)
        service_levels.append(sl)
        weights_history.append(vfa_policy.theta.copy())  # Store θ vector for this episode

    
        formatted_theta = np.array2string(vfa_policy.theta, formatter={'float_kind':lambda x: f"{x:+.3f}"})
        
        print(
            f"  Ep {ep + 1:3d}/{num_episodes} | "
            f"tau={tau:4.2f} | "
            f"SL={sl:.4f} | "
            f"Weights: {formatted_theta} | "
            f"t={time.time() - t0:.0f}s"
        )

        # ── Periodic checkpoint every 50 episodes ─────────────────────────
        if (ep + 1) % 50 == 0:
            ck_path = SAVE_DIR / f"vfa_checkpoint_ep{ep + 1:04d}.pkl"
            vfa_policy.save(ck_path)

    # ── Final save ────────────────────────────────────────────────────────────
    if save_path is None:
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = SAVE_DIR / f"vfa_trained_{ts}.pkl"

    vfa_policy.save(save_path)

    curve_path = Path(str(save_path).replace(".pkl", "_learning_curve.npy"))
    np.save(curve_path, np.array(service_levels))

    # ── Save weight evolution as CSV ────────────────────────────────────────────
    feature_names = _get_feature_names(ENABLE_COMPONENT_FAILURES, shift_timing_enabled=SHIFT_TIMING_ENABLED)
    weights_df = pd.DataFrame(
        weights_history,
        columns=feature_names,
    )
    weights_df.insert(0, 'episode', range(num_episodes))
    weights_df.insert(1, 'service_level', service_levels)
    
    weights_csv_path = Path(str(save_path).replace(".pkl", "_weights_evolution.csv"))
    weights_df.to_csv(weights_csv_path, index=False)

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
    print(f"  Learning curve    : {curve_path}")
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
        default=0,
        metavar="OFFSET",
        help="Seed offset: episode i uses random seed = offset + i",
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=INSTANCE_NAME,
        help="Simulator instance name",
    )
    args = parser.parse_args()

    train(
        num_episodes  = args.episodes,
        save_path     = Path(args.save) if args.save else None,
        seed_offset   = args.seed,
        instance_name = args.instance,
    )
