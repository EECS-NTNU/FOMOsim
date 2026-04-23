"""
run_correlation_analysis.py  —  VFA Feature Diagnostic

Runs N_EPISODES of the full episode structure (warmup + VFA learning, 14 days
each) and then analyses the feature vectors logged by LinearVFAPolicy.rl_logs.

Two outputs
───────────
  feature_study/feature_correlation_heatmap.png   — feature×feature Pearson r
  feature_study/feature_value_correlation.png     — per-feature r with V(S^x)
  feature_study/feature_statistics.csv            — mean/std/min/max per feature

The second plot is the signal-quality check: it reveals which features actually
co-vary with the learned value estimate, and whether D4/D5 carry useful signal
after the gross-flow fix.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Workspace root ─────────────────────────────────────────────────────────────
WORKSPACE_ROOT = Path(__file__).parents[3]
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))

from helpers import timeInMinutes
from policies.greedy_policy import GreedyPolicy
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy, EpisodeTrainingPolicy
from policies.sjovik_sund.run_simulation_ingvild import run_simulation, SimulationConfig
from settings import ENABLE_COMPONENT_FAILURES

# ── Study configuration ────────────────────────────────────────────────────────
N_EPISODES    : int = 5
EPISODE_DAYS  : int = 14
WARMUP_DAYS   : int = 2
START_HOUR    : int = 5
INSTANCE_NAME : str = "TD_W34_old"
NUM_VEHICLES  : int = 1
SEED_OFFSET   : int = 0

OUT_DIR = Path(__file__).parent / "feature_study"


def run_analysis() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build VFA policy with temporal features enabled ────────────────────────
    vfa = LinearVFAPolicy(
        learning_mode       = True,
        temporal_enabled    = True,
        maintenance_enabled = ENABLE_COMPONENT_FAILURES,
        seed                = 42,
    )
    vfa.use_experience_replay = False

    greedy          = GreedyPolicy()
    sim_start_min   = timeInMinutes(hours=START_HOUR)
    warmup_end_time = sim_start_min + WARMUP_DAYS * 24 * 60   # same formula as train_vfa

    print(f"Running {N_EPISODES} episodes × {EPISODE_DAYS} days "
          f"(warmup={WARMUP_DAYS}d, learning={EPISODE_DAYS - WARMUP_DAYS}d)...")

    # ── Episode loop — mirrors train_vfa structure ─────────────────────────────
    # EpisodeTrainingPolicy routes to GreedyPolicy during warmup and VFA+TD(0)
    # during the learning phase. apply_batch_update() is called after each
    # episode, matching the batch_size=1 default in train_vfa.
    for ep in range(N_EPISODES):
        episode_policy = EpisodeTrainingPolicy(
            vfa_policy      = vfa,
            warmup_policy   = greedy,
            warmup_end_time = warmup_end_time,
        )
        run_simulation(
            seed          = SEED_OFFSET + ep,
            policy        = episode_policy,
            duration      = 24 * EPISODE_DAYS,
            num_vehicles  = NUM_VEHICLES,
            instance_name = INSTANCE_NAME,
            config        = SimulationConfig(),
        )
        vfa.apply_batch_update()
        print(f"  Episode {ep + 1}/{N_EPISODES}: {len(vfa.rl_logs):,} VFA decisions logged so far")

    # ── Build dataframe from VFA decision logs ─────────────────────────────────
    # rl_logs is populated in LinearVFAPolicy.get_best_action for every VFA
    # decision (warmup decisions are excluded — EpisodeTrainingPolicy routes
    # those to GreedyPolicy instead). Each entry contains one value per feature
    # in FEATURE_NAMES, plus expected_value_V and td_error.
    print(f"\nTotal VFA decisions logged: {len(vfa.rl_logs):,}")
    df = pd.DataFrame(vfa.rl_logs)

    feature_cols = vfa.FEATURE_NAMES
    value_cols   = ["expected_value_V", "td_error"]
    keep_cols    = feature_cols + [c for c in value_cols if c in df.columns]
    df           = df[keep_cols].copy()

    # Drop columns that are constant (carry no information, cause NaN in corr)
    non_constant = df.nunique() > 1
    dropped      = df.columns[~non_constant].tolist()
    df           = df.loc[:, non_constant]
    if dropped:
        print(f"Dropped constant columns: {dropped}")

    present_features = [c for c in feature_cols if c in df.columns]

    # ── 1. Feature–feature correlation heatmap ────────────────────────────────
    corr_ff = df[present_features].corr(method="pearson")

    print("\n--- HIGHLY CORRELATED FEATURES (|r| > 0.85) ---")
    found_high = False
    corr_vals = corr_ff.to_numpy()
    for i in range(len(corr_ff.columns)):
        for j in range(i + 1, len(corr_ff.columns)):
            r = float(corr_vals[i, j])
            if not np.isnan(r) and abs(r) > 0.85:
                print(f"  WARNING: '{corr_ff.columns[i]}' ↔ '{corr_ff.columns[j]}'  r = {r:+.3f}")
                found_high = True
    if not found_high:
        print("  No highly correlated feature pairs found.")

    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(
        corr_ff, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1, square=True,
        linewidths=0.5, annot_kws={"size": 8}, ax=ax,
    )
    ax.set_title(
        f"VFA Feature Pearson Correlation Matrix  ({N_EPISODES} episodes × {EPISODE_DAYS}d)",
        fontsize=18,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    fig.tight_layout()
    heatmap_path = OUT_DIR / "feature_correlation_heatmap.png"
    fig.savefig(heatmap_path, dpi=300)
    plt.close(fig)
    print(f"\nHeatmap saved to '{heatmap_path}'")

    # ── 2. Feature–value correlation bar chart ────────────────────────────────
    if "expected_value_V" in df.columns:
        corr_fv = df[present_features].corrwith(df["expected_value_V"]).sort_values()

        print("\n--- FEATURE-VALUE CORRELATION (sorted by |r|) ---")
        for feat, r in corr_fv.sort_values(key=abs, ascending=False).items():
            print(f"  {feat:<42} r = {r:+.3f}")

        colors = ["#d73027" if v < 0 else "#1a9850" for v in corr_fv.values]
        fig, ax = plt.subplots(figsize=(10, max(6, len(corr_fv) * 0.4)))
        corr_fv.plot(kind="barh", color=colors, ax=ax)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Pearson r with V(S^x)")
        ax.set_title(
            f"Feature–Value Correlation  ({N_EPISODES} episodes × {EPISODE_DAYS}d)"
        )
        fig.tight_layout()
        fv_path = OUT_DIR / "feature_value_correlation.png"
        fig.savefig(fv_path, dpi=300)
        plt.close(fig)
        print(f"\nFeature–value chart saved to '{fv_path}'")

    # ── 3. Feature statistics table ───────────────────────────────────────────
    stats = df[present_features].agg(["mean", "std", "min", "max"]).T
    stats["variance"] = stats["std"] ** 2
    stats_path = OUT_DIR / "feature_statistics.csv"
    stats.to_csv(stats_path)
    print(f"Feature statistics saved to '{stats_path}'")


if __name__ == "__main__":
    run_analysis()
