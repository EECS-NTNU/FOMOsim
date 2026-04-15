"""
plot_nn.py — Training curve visualiser for NNValueNetwork checkpoints.

Usage (from FOMOsim root):
    python policies/sjovik_sund/NN/plot_nn.py

Panels:
  1. TD Loss  — raw trace, smoothed trend, ±1σ stability band,
                target-network update markers
  2. Convergence rate — rolling Δloss per episode (when is learning stalling?)
  3. Service Level — only rendered when present in the checkpoint
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rolling(values: list, window: int, fn):
    """Apply fn over a centred rolling window, shrinking at the edges."""
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
    """Episode-over-episode change in loss."""
    arr = np.array(values, dtype=float)
    d = np.empty_like(arr)
    d[0] = 0.0
    d[1:] = arr[1:] - arr[:-1]
    return d


# ── Main ──────────────────────────────────────────────────────────────────────

def plot_training_results(checkpoint_path: Path):
    checkpoint_path = Path(checkpoint_path)
    print(f"Loading checkpoint: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        print(f"[Error] File not found: {checkpoint_path}")
        return

    lc = checkpoint.get("learning_curve", [])
    if not lc:
        print("[Error] No 'learning_curve' key in this checkpoint.")
        return

    episodes = np.array([e["episode"]   for e in lc])
    losses   = [e["mean_loss"]           for e in lc]
    has_tau  = "tau"           in lc[0]
    has_sl   = "service_level" in lc[0]
    taus     = [e["tau"]           for e in lc] if has_tau else []
    sls      = [e["service_level"] for e in lc] if has_sl  else []

    # Target network update episodes (every 10 by convention)
    target_update_freq = 10
    update_eps = [e for e in episodes if e % target_update_freq == 0]

    smoothed     = _smooth(losses)
    std_band     = _rolling_std(losses)
    upper        = np.array(smoothed) + np.array(std_band)
    lower        = np.array(smoothed) - np.array(std_band)
    delta_smooth = _smooth(_delta(losses), window=5)

    smoothed_sl  = _smooth(sls) if has_sl else []

    # Metadata
    s_dim   = checkpoint.get("station_feature_dim", "?")
    v_dim   = checkpoint.get("vehicle_feature_dim", "?")
    g_dim   = checkpoint.get("global_feature_dim",  "?")
    ep_done = checkpoint.get("episode", int(episodes[-1]))
    lr_start = lc[0]["lr"]
    lr_end   = lc[-1]["lr"]

    # ── Layout ────────────────────────────────────────────────────────────────
    n_panels      = 2 + (1 if has_sl else 0)
    panel_heights = [3, 1.5] + ([2] if has_sl else [])
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(12, sum(panel_heights) + 2),
        sharex=True,
        gridspec_kw={"height_ratios": panel_heights},
    )
    if n_panels == 1:
        axes = [axes]

    exploration = (f"Boltzmann τ {taus[0]:.2f}→{taus[-1]:.2f}"
                   if has_tau else "greedy")
    fig.suptitle(
        f"NN Value Network — Training Summary\n"
        f"{checkpoint_path.name}   |   "
        f"Episodes: {ep_done}   |   "
        f"LR: {lr_start:.4f}→{lr_end:.5f}   |   "
        f"Exploration: {exploration}   |   "
        f"Input dims: S={s_dim} V={v_dim} G={g_dim}",
        fontsize=9, fontweight="bold", y=0.995,
    )

    # ── Panel 1: TD Loss with stability band ──────────────────────────────────
    ax1 = axes[0]

    # ±1σ stability band
    ax1.fill_between(episodes, lower, upper,
                     color="tab:red", alpha=0.12, label="±1σ rolling window")
    # Raw trace (very faint)
    ax1.plot(episodes, losses, color="tab:red", alpha=0.25, linewidth=0.8)
    # Smoothed trend
    ax1.plot(episodes, smoothed, color="tab:red", linewidth=2.5,
             label="smoothed mean (w=10)")

    # Target network update markers
    for ue in update_eps:
        ax1.axvline(ue, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    # Dummy handle for legend
    update_patch = mpatches.Patch(
        facecolor="none", edgecolor="grey", linestyle=":",
        label=f"target net update (every {target_update_freq} ep)",
    )

    # Annotate first and last smoothed value
    ax1.annotate(f"{smoothed[0]:.4f}",
                 xy=(episodes[0], smoothed[0]),
                 xytext=(6, 4), textcoords="offset points",
                 fontsize=8, color="tab:red")
    ax1.annotate(f"{smoothed[-1]:.4f}",
                 xy=(episodes[-1], smoothed[-1]),
                 xytext=(-50, 10), textcoords="offset points",
                 fontsize=8, color="tab:red",
                 arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8))

    ax1.set_ylabel("Mean TD Loss (MSE)", fontsize=10)
    ax1.set_title("TD Loss", fontsize=10, fontweight="bold")
    ax1.legend(handles=[ax1.lines[1], ax1.collections[0], update_patch],
               fontsize=8, loc="upper right")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.set_xlim(episodes[0], episodes[-1])

    # ── Panel 2: Convergence rate (Δloss) ────────────────────────────────────
    ax2 = axes[1]

    ax2.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax2.fill_between(episodes, delta_smooth, 0,
                     where=(np.array(delta_smooth) < 0),
                     color="tab:green", alpha=0.3, label="improving")
    ax2.fill_between(episodes, delta_smooth, 0,
                     where=(np.array(delta_smooth) >= 0),
                     color="tab:orange", alpha=0.3, label="worsening")
    ax2.plot(episodes, delta_smooth, color="dimgrey", linewidth=1.5)

    # Mark where smoothed delta first crosses 0 after initial drop
    crossings = [i for i in range(1, len(delta_smooth))
                 if delta_smooth[i-1] < 0 and delta_smooth[i] >= 0]
    if crossings:
        first_stall = episodes[crossings[0]]
        ax2.axvline(first_stall, color="tab:orange", linestyle="--",
                    linewidth=1.2, alpha=0.8)
        ax2.annotate(f"stall ep {first_stall}",
                     xy=(first_stall, 0),
                     xytext=(6, 8), textcoords="offset points",
                     fontsize=7.5, color="tab:orange")

    ax2.set_ylabel("Δ Loss / episode", fontsize=10)
    ax2.set_title("Convergence Rate  (negative = still improving)",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax2.grid(True, linestyle="--", alpha=0.3)

    # ── Panel 3: Service Level (only when present) ────────────────────────────
    if has_sl:
        ax3 = axes[2]
        sl_arr = np.array(sls)

        ax3.plot(episodes, sls, color="tab:blue", alpha=0.25, linewidth=0.8)
        ax3.plot(episodes, smoothed_sl, color="tab:blue", linewidth=2.5,
                 label="smoothed (w=10)")
        ax3.fill_between(episodes, smoothed_sl,
                         color="tab:blue", alpha=0.08)

        best_idx = int(np.argmax(smoothed_sl))
        ax3.axhline(smoothed_sl[best_idx], color="tab:blue",
                    linestyle="--", alpha=0.5, linewidth=1)
        ax3.annotate(
            f"best: {smoothed_sl[best_idx]:.3f}  (ep {episodes[best_idx]})",
            xy=(episodes[best_idx], smoothed_sl[best_idx]),
            xytext=(8, -14), textcoords="offset points",
            fontsize=8, color="tab:blue",
        )

        ax3.set_ylabel("Service Level", fontsize=10)
        ax3.set_title("Service Level  (learning phase, days 3–14)",
                      fontsize=10, fontweight="bold")
        ax3.legend(fontsize=8, loc="lower right")
        ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax3.grid(True, linestyle="--", alpha=0.3)
        ax3.set_xlabel("Training Episode", fontsize=10)
    else:
        axes[-1].set_xlabel("Training Episode", fontsize=10)
        axes[0].text(
            0.99, 0.05,
            "service_level not in checkpoint — retrain to get this panel",
            transform=axes[0].transAxes, fontsize=7.5,
            color="grey", ha="right", va="bottom", style="italic",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = checkpoint_path.parent / (checkpoint_path.stem + "_training_plot.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    model_path = (
        Path(__file__).parent
        / "models/training_log_seed1000_20260414_172707.csv"
    )
    plot_training_results(model_path)
