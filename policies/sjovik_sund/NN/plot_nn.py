"""
plot_nn.py — Training curve visualiser for NNValueNetwork training logs.

Reads from the per-episode CSV log (preferred — all episodes, all metrics)
and falls back to a checkpoint .pt file if no CSV is found alongside it.

Usage:
    python policies/sjovik_sund/NN/plot_nn.py
    python policies/sjovik_sund/NN/plot_nn.py --csv path/to/training_log.csv
    python policies/sjovik_sund/NN/plot_nn.py --checkpoint path/to/model.pt

Panels:
  1. TD Loss          — raw + smoothed + ±1σ band + target-net update markers
  2. Service Level    — raw + smoothed + best-episode annotation
  3. Value Spread     — mean max−min across candidates (NN discrimination signal)
  4. Fallback Rate    — % PostDecisionState.apply() failures (data quality)
  5. Reward / Sparsity — mean raw reward and % zero-reward decisions
"""

import csv
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rolling(values, window, fn):
    arr = np.array(values, dtype=float)
    out = np.empty_like(arr)
    half = window // 2
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        out[i] = fn(arr[lo:hi])
    return out

def _smooth(values, window=10):
    return _rolling(values, window, np.mean)

def _rolling_std(values, window=10):
    return _rolling(values, window, np.std)

def _delta(values):
    arr = np.array(values, dtype=float)
    d = np.empty_like(arr)
    d[0] = 0.0
    d[1:] = arr[1:] - arr[:-1]
    return d

def _col(lc, key, default=None):
    """Extract a column from the learning curve if present."""
    if lc and key in lc[0]:
        return [row[key] for row in lc]
    return default


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_csv(csv_path: Path) -> list:
    rows = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def _load_checkpoint(pt_path: Path) -> tuple:
    """Returns (learning_curve, metadata_dict)."""
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    meta = {
        "station_feature_dim": ck.get("station_feature_dim", "?"),
        "vehicle_feature_dim": ck.get("vehicle_feature_dim", "?"),
        "global_feature_dim":  ck.get("global_feature_dim",  "?"),
        "episode":             ck.get("episode", "?"),
    }
    return ck.get("learning_curve", []), meta


def _find_csv_for_checkpoint(pt_path: Path) -> Path | None:
    """Look for a CSV in the same directory whose name contains the seed."""
    seed_tag = None
    for part in pt_path.stem.split("_"):
        if part.startswith("seed"):
            seed_tag = part
            break
    candidates = sorted(pt_path.parent.glob("training_log*.csv"), reverse=True)
    if seed_tag:
        for c in candidates:
            if seed_tag in c.name:
                return c
    return candidates[0] if candidates else None


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_training_results(csv_path: Path = None, checkpoint_path: Path = None):
    lc   = []
    meta = {}

    # Prefer CSV (has every episode); fall back to checkpoint learning_curve
    if csv_path and csv_path.exists():
        lc = _load_csv(csv_path)
        print(f"Loaded CSV  : {csv_path}  ({len(lc)} episodes)")
    elif checkpoint_path and checkpoint_path.exists():
        lc, meta = _load_checkpoint(checkpoint_path)
        print(f"Loaded checkpoint: {checkpoint_path}  ({len(lc)} episodes)")
        # Also try to find the paired CSV for richer data
        paired = _find_csv_for_checkpoint(checkpoint_path)
        if paired:
            lc_csv = _load_csv(paired)
            if len(lc_csv) >= len(lc):
                lc = lc_csv
                print(f"  Using paired CSV: {paired}  ({len(lc)} episodes)")
    else:
        print("[Error] No CSV or checkpoint path provided / found.")
        return

    if not lc:
        print("[Error] No learning curve data found.")
        return

    # Load checkpoint metadata if we have a path and didn't load it yet
    if checkpoint_path and checkpoint_path.exists() and not meta:
        _, meta = _load_checkpoint(checkpoint_path)

    episodes = np.array([int(row["episode"]) for row in lc])
    losses   = [row["mean_loss"]    for row in lc]
    taus     = _col(lc, "tau")
    sls      = _col(lc, "service_level")
    spreads  = _col(lc, "mean_value_spread")
    fallbacks = _col(lc, "fallback_rate")
    pct_zero = _col(lc, "pct_zero_reward")
    rewards  = _col(lc, "mean_reward")

    has_sl       = sls      is not None
    has_spread   = spreads  is not None
    has_fallback = fallbacks is not None
    has_reward   = rewards  is not None and pct_zero is not None

    target_update_freq = 10
    update_eps = [e for e in episodes if e % target_update_freq == 0]

    smoothed  = _smooth(losses)
    std_band  = _rolling_std(losses)
    upper     = np.array(smoothed) + np.array(std_band)
    lower     = np.array(smoothed) - np.array(std_band)

    # ── Layout ────────────────────────────────────────────────────────────────
    panel_specs = [
        ("loss",   True,       3.0),
        ("sl",     has_sl,     2.5),
        ("spread", has_spread, 1.8),
        ("reward", has_reward, 1.8),
    ]
    active = [(name, h) for name, show, h in panel_specs if show]
    n_panels      = len(active)
    panel_heights = [h for _, h in active]
    panel_names   = [n for n, _ in active]

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(13, sum(panel_heights) + 1.5),
        sharex=True,
        gridspec_kw={"height_ratios": panel_heights},
    )
    if n_panels == 1:
        axes = [axes]

    exploration = (f"Boltzmann τ {taus[0]:.2f}→{taus[-1]:.2f}"
                   if taus else "greedy")
    s_dim = meta.get("station_feature_dim", "?")
    v_dim = meta.get("vehicle_feature_dim", "?")
    g_dim = meta.get("global_feature_dim",  "?")
    ep_done = meta.get("episode", int(episodes[-1]))
    lr_start = lc[0].get("lr", "?")
    lr_end   = lc[-1].get("lr", "?")

    fig.suptitle(
        f"NN Value Network — Training Summary\n"
        f"Episodes: {ep_done}   |   "
        f"LR: {lr_start:.5f}→{lr_end:.5f}   |   "
        f"Exploration: {exploration}"
        + (f"   |   Input dims: S={s_dim} V={v_dim} G={g_dim}" if s_dim != "?" else ""),
        fontsize=9, fontweight="bold", y=0.998,
    )

    ax_map = {name: axes[i] for i, name in enumerate(panel_names)}

    # ── Panel: TD Loss ────────────────────────────────────────────────────────
    ax = ax_map["loss"]
    ax.fill_between(episodes, lower, upper,
                    color="tab:red", alpha=0.12, label="±1σ rolling window")
    ax.plot(episodes, losses,   color="tab:red", alpha=0.2,  linewidth=0.7)
    ax.plot(episodes, smoothed, color="tab:red", linewidth=2.0, label="smoothed (w=10)")

    for ue in update_eps:
        ax.axvline(ue, color="grey", linestyle=":", linewidth=0.7, alpha=0.5)
    update_patch = mpatches.Patch(
        facecolor="none", edgecolor="grey", linestyle=":",
        label=f"target net update (every {target_update_freq} ep)",
    )
    ax.annotate(f"{smoothed[0]:.4f}", xy=(episodes[0], smoothed[0]),
                xytext=(6, 4), textcoords="offset points", fontsize=7.5, color="tab:red")
    ax.annotate(f"{smoothed[-1]:.4f}", xy=(episodes[-1], smoothed[-1]),
                xytext=(-50, 10), textcoords="offset points", fontsize=7.5, color="tab:red",
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8))
    ax.set_ylabel("TD Loss (MSE)", fontsize=9)
    ax.set_title("TD Loss", fontsize=9, fontweight="bold")
    ax.legend(handles=[ax.lines[1], ax.collections[0], update_patch],
              fontsize=7.5, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(episodes[0], episodes[-1])

    # ── Panel: Service Level ──────────────────────────────────────────────────
    if "sl" in ax_map:
        ax = ax_map["sl"]
        sl_arr = np.array(sls)
        smoothed_sl = _smooth(sls)
        ax.plot(episodes, sls, color="tab:blue", alpha=0.2, linewidth=0.7)
        ax.plot(episodes, smoothed_sl, color="tab:blue", linewidth=2.0, label="smoothed (w=10)")
        ax.fill_between(episodes, smoothed_sl, color="tab:blue", alpha=0.07)

        best_idx = int(np.argmax(smoothed_sl))
        ax.axhline(smoothed_sl[best_idx], color="tab:blue",
                   linestyle="--", alpha=0.5, linewidth=1)
        ax.annotate(
            f"best: {smoothed_sl[best_idx]:.3f}  (ep {episodes[best_idx]})",
            xy=(episodes[best_idx], smoothed_sl[best_idx]),
            xytext=(8, -14), textcoords="offset points",
            fontsize=8, color="tab:blue",
        )
        # Annotate final value
        ax.annotate(f"{smoothed_sl[-1]:.3f}", xy=(episodes[-1], smoothed_sl[-1]),
                    xytext=(-45, 6), textcoords="offset points",
                    fontsize=7.5, color="tab:blue",
                    arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.8))
        # Zoom y-axis to the actual data range (±2% padding) so growth is visible
        sl_min = max(0.0, float(np.min(smoothed_sl)) - 0.02)
        sl_max = min(1.0, float(np.max(smoothed_sl)) + 0.02)
        ax.set_ylim(sl_min, sl_max)
        ax.set_ylabel("Service Level", fontsize=9)
        ax.set_title("Service Level", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7.5, loc="lower right")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.grid(True, linestyle="--", alpha=0.3)

    # ── Panel: Value Spread ───────────────────────────────────────────────────
    if "spread" in ax_map:
        ax = ax_map["spread"]
        smoothed_spread = _smooth(spreads)
        ax.plot(episodes, spreads, color="tab:purple", alpha=0.2, linewidth=0.7)
        ax.plot(episodes, smoothed_spread, color="tab:purple", linewidth=2.0,
                label="smoothed (w=10)")
        ax.axhline(0.01, color="tab:orange", linestyle="--", linewidth=1.0, alpha=0.8,
                   label="discrimination threshold (0.01)")
        ax.fill_between(episodes, smoothed_spread, 0.01,
                        where=np.array(smoothed_spread) < 0.01,
                        color="tab:orange", alpha=0.12)
        ax.set_ylabel("V spread (max−min)", fontsize=9)
        ax.set_title("Candidate Value Spread  (< 0.01 = NN not discriminating)",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper left")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.grid(True, linestyle="--", alpha=0.3)

    # ── Panel: Reward & Sparsity ──────────────────────────────────────────────
    if "reward" in ax_map:
        ax = ax_map["reward"]
        ax2r = ax.twinx()

        smoothed_rew = _smooth(rewards)
        ax.plot(episodes, rewards,       color="tab:green", alpha=0.2, linewidth=0.7)
        ax.plot(episodes, smoothed_rew,  color="tab:green", linewidth=2.0, label="mean reward (smoothed)")
        ax.set_ylabel("Mean step reward", fontsize=9, color="tab:green")
        ax.tick_params(axis="y", labelcolor="tab:green")

        ax2r.plot(episodes, pct_zero, color="tab:brown", alpha=0.2, linewidth=0.7)
        ax2r.plot(episodes, _smooth(pct_zero), color="tab:brown", linewidth=1.5,
                  linestyle="--", label="% zero reward (smoothed)")
        ax2r.set_ylabel("% zero-reward decisions", fontsize=9, color="tab:brown")
        ax2r.tick_params(axis="y", labelcolor="tab:brown")
        ax2r.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2r.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc="lower right")
        ax.set_title("Reward Signal  (mean & sparsity)",
                     fontsize=9, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Training Episode", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save next to the source file
    source = csv_path or checkpoint_path
    output_path = source.parent / (source.stem + "_training_plot.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved : {output_path}")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot NN training results")
    parser.add_argument("--csv",        type=str, default=None,
                        help="Path to training_log CSV (preferred)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint file")
    args = parser.parse_args()

    # Defaults when run without arguments
    _here = Path(__file__).parent
    csv_path = Path(args.csv) if args.csv else None
    ck_path  = Path(args.checkpoint) if args.checkpoint else None

    if csv_path is None and ck_path is None:
        # Auto-discover: newest CSV in models/ dir
        candidates = sorted((_here / "models").glob("training_log*.csv"), reverse=True)
        if candidates:
            csv_path = candidates[0]
            print(f"Auto-selected CSV: {csv_path}")
        else:
            candidates = sorted((_here / "models").glob("nn_model_final*.pt"), reverse=True)
            if candidates:
                ck_path = candidates[0]
                print(f"Auto-selected checkpoint: {ck_path}")

    plot_training_results(csv_path=csv_path, checkpoint_path=ck_path)
