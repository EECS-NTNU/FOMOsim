"""
evaluate_checkpoints.py — Offline Checkpoint Evaluation

Auto-discovers all training runs from training_log_*.csv files in the models/
directory, finds the corresponding episode checkpoints for each run, and
evaluates them on a fixed validation set to produce smooth convergence curves.

One output CSV per run, written to models/validation_curves/.

Usage (from repo root):
    # Evaluate all discovered runs
    python policies/sjovik_sund/NN/evaluate_checkpoints.py

    # Evaluate only runs whose log filename contains a substring
    python policies/sjovik_sund/NN/evaluate_checkpoints.py --filter arch_128-64-32

    # Use full-maintenance MDP config
    python policies/sjovik_sund/NN/evaluate_checkpoints.py --maintenance
"""

import argparse
import csv
import logging
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure repo root is on the path when running the script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from policies.sjovik_sund.NN.NNGreedyPolicy import NNGreedyPolicy
from policies.sjovik_sund.NN.nn_model import NNValueNetwork
from policies.sjovik_sund.mdp.mdp_config import MDPConfig
from policies.sjovik_sund.run_simulation_ingvild import SimulationConfig, run_simulation

# ── Constants ────────────────────────────────────────────────────────────────

VALIDATION_SEEDS = [8001, 8002, 8003, 8004, 8005]

MODELS_DIR    = Path(__file__).parent / "models"
OUT_DIR       = MODELS_DIR / "validation_curves"
EPISODE_DAYS  = 14
NUM_VEHICLES  = 1
INSTANCE_NAME = "TD_W34_old"

# training_log files are named: training_log_<run_tag>.csv
# checkpoint files are named:   nn_model_ep<NNNN>_<run_tag>.pt
_LOG_PREFIX  = "training_log_"
_CKPT_PREFIX = "nn_model_ep"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── Data ─────────────────────────────────────────────────────────────────────

@dataclass
class Run:
    tag: str          # e.g. "seed1000_arch_128-64-32_20260425_120130"
    log: Path
    checkpoints: list[Path]   # sorted by episode number


# ── Discovery ────────────────────────────────────────────────────────────────

def _episode_from_name(path: Path) -> int:
    m = re.search(r"ep(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def _discover_runs(filter_str: str | None) -> list[Run]:
    """
    Group all nn_model_ep*.pt files found under MODELS_DIR by run tag.
    Run tag = filename with the episode prefix stripped, e.g.
        nn_model_ep0050_seed1000_arch_64_20260424_154535.pt
        → seed1000_arch_64_20260424_154535
    """
    all_pts = list(MODELS_DIR.rglob(f"{_CKPT_PREFIX}*.pt"))
    if filter_str:
        all_pts = [p for p in all_pts if filter_str in p.name]

    # Group by tag (strip leading "nn_model_ep????_")
    tag_to_ckpts: dict[str, list[Path]] = {}
    for pt in all_pts:
        # "nn_model_ep0050_seed1000_arch_64_..." → strip up to second underscore after prefix
        stem = pt.stem  # e.g. nn_model_ep0050_seed1000_arch_64_20260424_154535
        m = re.match(r"nn_model_ep\d+_(.*)", stem)
        if not m:
            continue
        tag = m.group(1)
        tag_to_ckpts.setdefault(tag, []).append(pt)

    runs = []
    for tag, ckpts in sorted(tag_to_ckpts.items()):
        # Find the corresponding training log (optional — may not exist)
        log_file = next(MODELS_DIR.rglob(f"{_LOG_PREFIX}{tag}.csv"), None)
        runs.append(Run(
            tag         = tag,
            log         = log_file,
            checkpoints = sorted(ckpts, key=_episode_from_name),
        ))

    return runs


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _infer_value_hidden_dims(state_dict: dict) -> list[int]:
    """Read value_hidden_dims from the saved weights (all linear layers except the last output layer)."""
    keys = sorted(
        (k for k in state_dict if k.startswith("value_mlp.net.") and k.endswith(".weight")),
        key=lambda k: int(k.split(".")[2]),
    )
    # All layers except the last are hidden; their output size = first dim of weight matrix.
    return [state_dict[k].shape[0] for k in keys[:-1]]


def _load_checkpoint(path: Path) -> tuple[NNValueNetwork, int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state"]
    value_hidden_dims = ckpt.get("value_hidden_dims") or _infer_value_hidden_dims(state_dict)
    model = NNValueNetwork(
        station_feature_dim = ckpt["station_feature_dim"],
        vehicle_feature_dim = ckpt["vehicle_feature_dim"],
        global_feature_dim  = ckpt["global_feature_dim"],
        value_hidden_dims   = value_hidden_dims,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt.get("episode", _episode_from_name(path))


def _service_level(simulator) -> float:
    m     = simulator.state.metrics
    trips = max(1, m.get_aggregate_value("trips") or 1)
    starv = m.get_aggregate_value("starvations") or 0
    cong  = m.get_aggregate_value("long congestions") or 0
    return 1.0 - (starv + cong) / trips


def _eval_checkpoint(model: NNValueNetwork, mdp_config: MDPConfig, seeds: list[int]) -> list[float]:
    sim_config = SimulationConfig()
    sls = []
    for seed in seeds:
        policy = NNGreedyPolicy(nn_model=model, config=mdp_config)
        sim = run_simulation(
            seed          = seed,
            policy        = policy,
            duration      = 24 * EPISODE_DAYS,
            num_vehicles  = NUM_VEHICLES,
            instance_name = INSTANCE_NAME,
            config        = sim_config,
        )
        sls.append(_service_level(sim))
    return sls


# ── Per-run evaluation ────────────────────────────────────────────────────────

def evaluate_run(run: Run, mdp_config: MDPConfig, seeds: list[int]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"validation_{run.tag}.csv"

    if not run.checkpoints:
        log.warning("  [%s] No checkpoints found — skipping.", run.tag)
        return

    total_sims = len(run.checkpoints) * len(seeds)
    log.info("  [%s] %d checkpoints × %d seeds = %d simulations → %s",
             run.tag, len(run.checkpoints), len(seeds), total_sims, out_path.name)

    write_header = not out_path.exists()
    already_done = set()
    if out_path.exists():
        with open(out_path, newline="") as f:
            for row in csv.DictReader(f):
                already_done.add(int(row["episode"]))

    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["episode", "avg_service_level", "std_service_level"])

        for i, ckpt_path in enumerate(run.checkpoints, 1):
            model, episode = _load_checkpoint(ckpt_path)

            if episode in already_done:
                log.info("    [%d/%d] ep%04d already evaluated — skipping.",
                         i, len(run.checkpoints), episode)
                continue

            log.info("    [%d/%d] ep%04d  %s", i, len(run.checkpoints), episode, ckpt_path.name)
            try:
                sls = _eval_checkpoint(model, mdp_config, seeds)
            except RuntimeError as e:
                log.warning("    ep%04d SKIPPED — incompatible checkpoint (shape mismatch): %s", episode, e)
                continue
            avg = statistics.mean(sls)
            std = statistics.stdev(sls) if len(sls) > 1 else 0.0

            writer.writerow([episode, f"{avg:.6f}", f"{std:.6f}"])
            f.flush()
            log.info("             avg_SL=%.4f  std=%.4f  per-seed=%s",
                     avg, std, [f"{s:.4f}" for s in sls])

    log.info("  [%s] Done → %s", run.tag, out_path)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all NN training runs on fixed validation seeds.")
    parser.add_argument("--filter",      default=None,   help="Only process runs whose log filename contains this substring (e.g. 'arch_128-64-32')")
    parser.add_argument("--maintenance", action="store_true", help="Use full_maintenance MDPConfig (default: no_maintenance)")
    args = parser.parse_args()

    mdp_config = MDPConfig.full_maintenance() if args.maintenance else MDPConfig.no_maintenance()

    runs = _discover_runs(args.filter)
    if not runs:
        log.error("No training log files found under %s", MODELS_DIR)
    else:
        log.info("Discovered %d run(s):", len(runs))
        for r in runs:
            log.info("  %s  (%d checkpoints)", r.tag, len(r.checkpoints))
        log.info("")

        for run in runs:
            evaluate_run(run, mdp_config, VALIDATION_SEEDS)
