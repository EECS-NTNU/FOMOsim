"""
train_nn_rollout.py  —  Offline Training Loop for the NN Value Network

Trains the NNValueNetwork using episodic simulation with TD(0) updates,
a target network for stability, and an experience replay buffer.

─────────────────────────────────────────────────────────────────────────────
EPISODE STRUCTURE  (mirrors train_vfa.py)
─────────────────────────────────────────────────────────────────────────────

  Days 1 – 7    Warm-up  : GreedyPolicy drives the system.
                           No TD updates — lets station inventories and repair
                           queues settle after the steady-state odometer draw.

  Days 8 – 21   Learning : NNLearningPolicy with Boltzmann exploration.
                           At each vehicle arrival:
                             • generate (MdpAction, sim.Action) pairs
                             • encode each post-decision state via encode_state()
                             • score via online NN  →  Boltzmann selection
                             • record transition (S^x_prev, r, S^x_cur) to buffer

  After each episode:  batch gradient updates from the replay buffer.
  Every TARGET_UPDATE_FREQ episodes: copy online weights → target network.

─────────────────────────────────────────────────────────────────────────────
HOW THIS DIFFERS FROM train_vfa.py
─────────────────────────────────────────────────────────────────────────────

  train_vfa.py                       train_nn_rollout.py
  ────────────────────────────────────────────────────────────────
  Weight update: θ ← θ + α δ φ      loss = δ²; optimizer.step()
  Value fn: θᵀ φ(S^x)               NN(encode_state(S^x))
  No target network needed           Frozen target network for TD stability
  No replay buffer needed            Replay buffer to break temporal correlation
  Feature vector: 28-dim φ          Raw tensor: station block + vehicle + global
  ────────────────────────────────────────────────────────────────

The warmup wrapper (NNEpisodeTrainingPolicy) is structurally identical to
EpisodeTrainingPolicy in LinearVFAPolicy.py, just wrapping the NN policy
instead of the linear VFA.
"""

import csv
import math
import sys
import copy
import random
from collections import deque
from pathlib import Path
from typing import Optional
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim

# ── Workspace root on sys.path ────────────────────────────────────────────────
WORKSPACE_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(WORKSPACE_ROOT))

from helpers import timeInMinutes
from policies.policy import Policy
from policies.greedy_policy import GreedyPolicy
from policies.greedy_policy_maintenance import GreedyMaintenancePolicy
# ── State encoding mode ───────────────────────────────────────────────────────
# USE_VFA_FEATURES=True  → 28-dim hand-crafted features (same as LinearVFAPolicy)
#                          Uses FlatNNValueNetwork (simple MLP, no Deep Sets).
# USE_VFA_FEATURES=False → raw per-station/vehicle blocks (Deep Sets architecture).
USE_VFA_FEATURES: bool = False

if USE_VFA_FEATURES:
    from policies.sjovik_sund.NN.nn_model import FlatNNValueNetwork, build_vfa_nn_value_network as build_nn_value_network
    from policies.sjovik_sund.NN.nn_state_encoder import encode_state_vfa as encode_state
else:
    from policies.sjovik_sund.NN.nn_model import NNValueNetwork, build_nn_value_network
    from policies.sjovik_sund.NN.nn_state_encoder import encode_state
from policies.sjovik_sund.NN.nn_state_encoder import (
    get_global_feature_dim,
    get_station_feature_dim,
    set_encoder_options,
)
from policies.sjovik_sund.mdp.candidate_generator_nn import generate_candidates
#from policies.sjovik_sund.mdp.candidate_generator_nn import generate_candidates
#from policies.sjovik_sund.mdp.candidate_generator_nn import generate_candidates
from policies.sjovik_sund.mdp.mdp_formulation import (
    extract_mdp_state,
    PostDecisionState,
    MDPState,
)
from policies.sjovik_sund.mdp.mdp_config import MDPConfig
from policies.sjovik_sund.mdp.reward_nn import RewardCalculator, RewardConfig
from policies.sjovik_sund.NN.nn_debug_logger import MaintenanceDebugLogger
from policies.sjovik_sund.run_simulation_ingvild import run_simulation, SimulationConfig
from settings import ENABLE_COMPONENT_FAILURES

# --- Device Selection ---
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

NUM_EPISODES         : int   = 800
WARMUP_DAYS          : int   = 7   # GreedyPolicy settling period after odometer warm-start
LEARNING_DAYS        : int   = 21 # NNLearningPolicy records transitions after warmup
EPISODE_DAYS         : int   = WARMUP_DAYS + LEARNING_DAYS

LR_START             : float = 5e-4  # Adam lr at episode 0
LR_END               : float = 5e-5  # Adam lr at episode N (linearly decayed)

GAMMA                : float = 0.97  # discount factor (matches linear VFA)

TARGET_UPDATE_FREQ   : int   = 25   # kept for CLI arg compatibility; not used when POLYAK > 0
POLYAK               : float = 0.0005 # soft target update rate: θ_target ← (1-τ)θ_target + τθ_online
                                     # applied every episode instead of hard copy every N episodes.
                                     # Set to 0.0 to fall back to hard copies (original behaviour).

N_STEP_RETURN        : int   = 3    # n-step TD return length.
                                     # 1 = standard TD(0).
                                     # Try n=3 now that normalize() scales by the n-step discount sum.
                                     # Previous attempt hurt greedy_sl (0.947→0.940) due to inflated
                                     # targets from dividing by the 1-step max regardless of n.

BUFFER_SIZE          : int   = 50_000  # max transitions in replay buffer; small buffer flushes stale exploration data faster
BATCH_SIZE           : int   = 128     # mini-batch size per gradient update
MIN_BUFFER_SIZE      : int   = 256     # start learning only after this many transitions

GRAD_CLIP_NORM       : float = 1.0   # raised from 0.5; reward clipping to [-5,5] keeps targets bounded
REWARD_NORM_EPS      : float = 1e-8  # avoid div-by-zero in reward normalizer

# mann exploration temperature schedule (linear anneal over all episodes)
# tau_start: high temperature early → broad exploration of action space
# tau_end:   near-zero  late        → essentially greedy exploitation
TAU_START          : float = 0.5    # Decreased from 0.5
TAU_END            : float = 0.001  # Decreased from 0.02
TAU_ANNEAL_EPISODES: int   = NUM_EPISODES    # anneal tau over this many eps, then hold at TAU_END

# ── Experiment switches (flip one at a time for ablations) ────────────────────
# All four are threaded through train_nn_rollout() and encoded in output filenames.
# Gemini-recommended fixes — change defaults here OR pass via CLI args.
#   N_STEP_RETURN = 1     → kill deadly triad (was 3)
#   BUFFER_SIZE   = 5_000 → force near-on-policy data (was 50_000)
#   POLYAK        = 0.0   → hard target copy every TARGET_UPDATE_FREQ eps (was 0.0005)
#   TAU_END       = 0.1   → constant exploration, no collapse (was 0.001)
USE_MAINTENANCE_SHAPING : bool  = False  # True=+10/+5/+5 shaping, False=base reward only

INSTANCE_NAME        : str   = "TD_W34_old"
NUM_VEHICLES         : int   = 1
START_HOUR           : int   = 5     # simulation clock starts at 05:00

SAVE_DIR = Path(__file__).parent / "models"

# ─────────────────────────────────────────────────────────────────────────────
# Debug verbosity
# ─────────────────────────────────────────────────────────────────────────────
# Set to the episode index (0-based) you want to trace in detail.
# That episode will print the full state representation on the first learning
# decision, and a brief candidate-score summary on every subsequent decision.
# Set to -1 to disable all debug output (normal training).
VERBOSE_EPISODE: int = 0

# ── Diagnostic probes ─────────────────────────────────────────────────────────
# DEBUG_EVERY   : print a full diagnostic summary every N episodes.
# EVAL_GREEDY_EVERY : run a tau=0 eval episode every N episodes (0 = disabled).
#                     Each eval adds ~1 episode worth of wall time.
DEBUG_EVERY       : int = 5   # set to 0 to silence all diagnostic output
EVAL_GREEDY_EVERY : int = 10   # set to 0 to skip greedy evaluation runs
EVAL_GREEDY_SEED  : int = 42 # fixed seed for all greedy evals — keeps the convergence curve comparable across episodes


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIENCE REPLAY BUFFER
# ═════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Fixed-size ring buffer storing TD transitions.

    Each entry is a tuple:
        (encoded_cur, reward, encoded_next, n_steps, done)

    where encoded_* are dicts {"station_block", "vehicle_block", "global_context"}
    produced by encode_state() on POST-DECISION states S^x, reward is the
    accumulated n-step return G_n = Σ_{i=0}^{n-1} γ^i · r_i, and n_steps is
    the number of steps accumulated (used as the bootstrap exponent γ^n).

    WHY POST-DECISION STATES?
    The value function V(S^x) is defined on post-decision states (after the
    action, before stochastic demand). Encoding S^x rather than the raw pre-
    decision state removes demand randomness from the input and makes the
    learning target more stable — this mirrors how the linear VFA evaluates
    V(S^x) = θᵀ φ(S^x) rather than φ of the pre-decision state.

    WHY REPLAY BUFFER?
    Consecutive transitions in a single vehicle route are highly correlated
    (same time of day, same vehicle load progression). Sampling random mini-
    batches breaks this correlation and stabilizes gradient estimates.
    """

    def __init__(self, max_size: int = BUFFER_SIZE):
        self._buffer  = deque(maxlen=max_size)
        self.max_size = max_size

    def push(self, encoded_cur: dict, reward: float, encoded_next: dict, n_steps: int, done: bool):
        """Add one (S^x_cur, G_n, S^x_{cur+n}, n_steps, done) transition."""
        self._buffer.append((encoded_cur, reward, encoded_next, n_steps, done))

    def sample(self, batch_size: int) -> list:
        """
        Sample a random mini-batch.

        Random sampling is the whole point: it removes temporal ordering
        from the gradient signal so the optimizer sees a diverse mix of
        system states rather than a recent trajectory.
        """
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)

    def ready(self, min_size: int = MIN_BUFFER_SIZE) -> bool:
        """True once enough transitions have been collected to start learning."""
        return len(self) >= min_size


# ═════════════════════════════════════════════════════════════════════════════
# REWARD NORMALIZER
# ═════════════════════════════════════════════════════════════════════════════

class RewardNormalizer:
    """
    EMA normalizer for step rewards.

    Rewards range from -76.7 to 0.0 (mean ≈ -2.7, std ≈ 5.6).
    Without normalization, a single bad step produces a TD target of -76.5,
    creating a gradient ≈5900× larger than a typical step.

    Uses exponential moving averages (EMA) for mean and variance so the
    normalizer tracks the *current policy's* reward distribution rather
    than the all-time average.

    alpha controls the adaptation speed (half-life ≈ ln(2)/alpha samples):
      alpha=0.005 → half-life ≈ 139 samples ≈ ~0.15 episodes  (fast adapt)
      alpha=0.001 → half-life ≈ 693 samples ≈ ~0.7 episodes   (moderate)
    """

    def __init__(
        self,
        alpha: float = 0.005,
        mode: str = "ema",
        fixed_scale: float = 10.0,
    ):
        if mode not in {"ema", "fixed", "none"}:
            raise ValueError(f"Unknown reward normalization mode: {mode!r}")
        if fixed_scale <= 0.0:
            raise ValueError("fixed_scale must be positive")

        self._alpha = alpha
        self._mode = mode
        self._fixed_scale = fixed_scale
        self._mean  = 0.0
        self._var   = 1.0    # start with unit variance (no scaling until first update)
        self._n     = 0      # counts updates; used only for the CSV log field

    def update(self, reward: float) -> None:
        self._n += 1
        # --- EMA Reward Tracking (Commented out to prevent Value Spread Explosion) ---
        if self._n == 1:
            self._mean = reward   # cold-start: seed mean at first reward
            return
        self._mean = (1.0 - self._alpha) * self._mean + self._alpha * reward
        self._var  = (1.0 - self._alpha) * self._var  + self._alpha * (reward - self._mean) ** 2

    @property
    def mean(self) -> float:
        return self._mean
        #return 0.0

    @property
    def std(self) -> float:
        return math.sqrt(max(self._var, REWARD_NORM_EPS))

    # ── Normalization call counter — used for periodic diagnostic prints ──────
    _norm_call_count: int = 0
    _NORM_PRINT_EVERY: int = 500   # print one line every N normalize() calls
    _norm_first_call: bool = True  # print a detailed one-shot comparison on first call

    '''def normalize(self, reward: float, n_steps: int = 1, gamma: float = 0.99) -> float:
        # EMA-based normalization using fitted single-step reward statistics.
        #
        # WHY: the old fixed-range approach divided by 76.7 * discount_sum (≈228
        # for n=3), crushing typical rewards (-3.5) to -0.015 — just 1.5% of the
        # target range. The bootstrap term γ³·V_next then dominated 98%+ of every
        # TD target, so the network was fitting its own prior predictions rather
        # than the reward signal. All architectures converged to the same flat SL.
        #
        # HOW: we scale the n-step EMA mean and std from single-step statistics
        # (under approximate i.i.d. reward assumption):
        #   E[G_n] ≈ μ · Σγ^i      Var[G_n] ≈ σ² · Σγ^{2i}
        # This keeps normalized G_n ≈ O(1) regardless of n_steps.
        discount_sum   = sum(gamma ** i for i in range(n_steps))
        sq_discount_sum = sum((gamma ** i) ** 2 for i in range(n_steps))

        expected_std  = math.sqrt(max(self._var * sq_discount_sum, REWARD_NORM_EPS))

        # Scale only — do NOT subtract expected mean.
        # Mean subtraction zeroed out the reward term (≈ 0 when mean ≈ actual reward),
        # leaving the bootstrap term γ^n·V_next as 100% of the TD signal and causing
        # V to spiral negative. Dividing by std only preserves the sign/magnitude so
        # the reward anchors V toward the true normalized value (≈ μ/σ / (1−γ^n)).
        normalized = reward / expected_std

        # ── One-shot first-call print ─────────────────────────────────────────
        if RewardNormalizer._norm_first_call:
            RewardNormalizer._norm_first_call = False
            old_norm = reward / (76.7 * discount_sum)
            print(
                f"\n  [REWARD NORM — FIRST CALL]"
                f"\n    raw G_{n_steps}          = {reward:.4f}"
                f"\n    EMA 1-step μ         = {self._mean:.4f}   σ={self.std:.4f}"
                f"\n    expected n-step σ    = {expected_std:.4f}  (mean NOT subtracted)"
                f"\n    normalized (÷σ only) = {normalized:.4f}   (O(1) ✓)"
                f"\n    OLD fixed-range      = {old_norm:.6f}  (was crushing signal)"
                f"\n    bootstrap γ^{n_steps}       = {gamma**n_steps:.4f}"
                f"\n    → V converges toward {self._mean/self.std:.2f}/(1-γ^n) ≈ {self._mean/self.std/(1-gamma**n_steps):.1f} (not 0)\n"
            )

        # ── Periodic diagnostic print ─────────────────────────────────────────
        RewardNormalizer._norm_call_count += 1
        if RewardNormalizer._norm_call_count % RewardNormalizer._NORM_PRINT_EVERY == 0:
            bootstrap_scale = gamma ** n_steps
            print(
                f"  [REWARD NORM #{RewardNormalizer._norm_call_count}]"
                f"  raw_G{n_steps}={reward:.4f}"
                f"  EMA_1step: μ={self._mean:.4f} σ={self.std:.4f}"
                f"  nstep_std={expected_std:.4f}"
                f"  normalized={normalized:.4f}"
                f"  bootstrap_γ^n={bootstrap_scale:.4f}"
                f"  → reward is {abs(normalized)/(abs(normalized)+abs(bootstrap_scale)*abs(self._mean/self.std)+1e-8)*100:.1f}% of signal"
            )

        return max(-5.0, min(5.0, normalized))'''
        
    def normalize(self, reward: float, n_steps: int = 1, gamma: float = 0.99) -> float:
        if self._mode == "none":
            return reward

        # Calculate N-step discounts
        discount_sum   = sum(gamma ** i for i in range(n_steps))
        sq_discount_sum = sum((gamma ** i) ** 2 for i in range(n_steps))

        if self._mode == "fixed":
            normalized = reward / max(self._fixed_scale * discount_sum, REWARD_NORM_EPS)
            return max(-5.0, min(5.0, normalized))

        expected_std  = math.sqrt(max(self._var * sq_discount_sum, REWARD_NORM_EPS))
        
        # 🔴 THE CURE: Subtract the expected N-step mean.
        # This makes the "average" reward exactly 0.0, so the infinite-horizon 
        # baseline V(S) stays anchored at 0.0. The network only learns the variance (good vs bad).
        expected_mean = self._mean * discount_sum
        
        # Divide by expected_std
        normalized = (reward - expected_mean) / expected_std

        # ── Periodic diagnostic print ─────────────────────────────────────────
        RewardNormalizer._norm_call_count += 1
        if RewardNormalizer._norm_call_count % RewardNormalizer._NORM_PRINT_EVERY == 0:
            print(
                f"  [REWARD NORM] raw={reward:.4f}  "
                f"expected_mean={expected_mean:.4f}  "
                f"normalized={normalized:.4f}"
            )

        return max(-5.0, min(5.0, normalized))
        
    '''def normalize(self, reward: float, n_steps: int = 1, gamma: float = 0.99) -> float:
        # Fixed-range normalization scaled to the n-step return window.
        # 1-step max penalty is 76.7. For n steps the worst-case accumulated
        # return is 76.7 × Σ_{i=0}^{n-1} γ^i, so we divide by that sum to
        # keep TD targets in [-1, 0] regardless of n.
        # n=1: divides by 76.7 (same as before).
        # n=3: divides by 76.7 × (1 + γ + γ²) ≈ 227.8 — prevents inflated targets.
        discount_sum = sum(gamma ** i for i in range(n_steps))
        return reward / (76.7 * discount_sum)'''


# ═════════════════════════════════════════════════════════════════════════════
# BOLTZMANN ACTION SELECTION
# ═════════════════════════════════════════════════════════════════════════════

def _greedy_select(values: list) -> int:
    """Return the index of the highest-value candidate (greedy argmax)."""
    return max(range(len(values)), key=lambda i: values[i])


def _boltzmann_select(values: list, tau: float) -> int:
    """
    Sample a candidate proportional to softmax(V / tau).

    tau → 0 : collapses to greedy (pure exploitation)
    tau → ∞ : uniform random (pure exploration)

    Values are shifted by their max before exponentiation to prevent
    overflow when tau is small and value differences are large.
    """
    if tau <= 0.0 or len(values) == 1:
        return _greedy_select(values)
    import torch.nn.functional as F
    v = torch.tensor(values, dtype=torch.float32)
    probs = F.softmax(v / tau, dim=0)
    return torch.multinomial(probs, num_samples=1).item()


# ═════════════════════════════════════════════════════════════════════════════
# TD LOSS
# ═════════════════════════════════════════════════════════════════════════════

_BATCH_FORWARD_VERIFIED = False   # printed once on first batched gradient step


def _compute_td_loss(online_model, target_model, batch, gamma,
                     reward_normalizer=None, collect_debug: bool = False):
    """
    Compute the mean Huber TD loss over a mini-batch in a single batched
    forward pass through each model.

    Previously: 128 separate forward() calls per gradient step (one per sample).
    Now: one forward_batch() call each for online and target model.

    The batch tuple layout is (enc_cur, reward, enc_next, n_steps, done) where
    reward is the accumulated n-step return G_n and n_steps is used to apply
    the correct γ^n discount when bootstrapping from the target network.
    """
    global _BATCH_FORWARD_VERIFIED

    # ── Unpack batch into column tensors ─────────────────────────────────────
    enc_curs   = [t[0] for t in batch]
    rewards    = [t[1] for t in batch]
    enc_nexts  = [t[2] for t in batch]
    n_steps_l  = [t[3] for t in batch]
    dones      = [t[4] for t in batch]

    # Stack encoded states: each field goes from [N, D] → [B, N, D]
    def _stack(enc_list, key):
        return torch.stack([e[key] for e in enc_list]).to(device)

    cur_station  = _stack(enc_curs,  "station_block")    # [B, N, station_feature_dim]
    cur_vehicle  = _stack(enc_curs,  "vehicle_block")    # [B, M, vehicle_feature_dim]
    cur_global   = _stack(enc_curs,  "global_context")   # [B, global_feature_dim]

    next_station = _stack(enc_nexts, "station_block")
    next_vehicle = _stack(enc_nexts, "vehicle_block")
    next_global  = _stack(enc_nexts, "global_context")

    if not _BATCH_FORWARD_VERIFIED:
        print(
            f"  [BATCH FORWARD] first call — "
            f"station {list(cur_station.shape)}  "
            f"vehicle {list(cur_vehicle.shape)}  "
            f"global {list(cur_global.shape)}  "
            f"| {len(batch)} samples in one forward pass"
        )
        _BATCH_FORWARD_VERIFIED = True

    # ── Online forward (needs grad) ───────────────────────────────────────────
    v_curs = online_model.forward_batch(cur_station, cur_vehicle, cur_global)  # [B, 1]

    # ── Target forward + TD targets (no grad) ────────────────────────────────
    with torch.no_grad():
        v_nexts = target_model.forward_batch(next_station, next_vehicle, next_global)  # [B, 1]

        norm_rewards = [
            reward_normalizer.normalize(r, n_steps=n, gamma=gamma) if reward_normalizer is not None else r
            for r, n in zip(rewards, n_steps_l)
        ]
        r_t    = torch.tensor(norm_rewards, dtype=torch.float32, device=device).unsqueeze(1)   # [B, 1]
        done_t = torch.tensor(dones,        dtype=torch.float32, device=device).unsqueeze(1)   # [B, 1]
        gn_t   = torch.tensor(
            [gamma ** n for n in n_steps_l], dtype=torch.float32, device=device
        ).unsqueeze(1)  # [B, 1]  — γ^n varies per sample (n=N_STEP_RETURN or shorter at episode end)

        # TD target: G_n + γ^n · V_target(S^x_next)  (bootstrap skipped when done)
        td_targets = r_t + (1.0 - done_t) * gn_t * v_nexts  # [B, 1]

    loss = torch.nn.functional.huber_loss(v_curs, td_targets, delta=1.0)

    debug_info = None
    if collect_debug:
        import statistics
        v_list       = v_curs.detach().cpu().squeeze(1).tolist()
        tgt_list     = td_targets.cpu().squeeze(1).tolist()
        norm_r_list  = r_t.cpu().squeeze(1).tolist()
        v_next_list   = v_nexts.cpu().squeeze(1).tolist()
        bootstrap_t   = (1.0 - done_t) * gn_t * v_nexts
        bootstrap_l   = bootstrap_t.cpu().squeeze(1).tolist()
        td_err_list   = (td_targets - v_curs.detach()).cpu().squeeze(1).tolist()
        terminal_tgts = [t for t, done in zip(tgt_list, dones) if done]
        nonterm_tgts  = [t for t, done in zip(tgt_list, dones) if not done]
        _mean = lambda xs: statistics.mean(xs) if xs else 0.0
        _std = lambda xs: statistics.stdev(xs) if len(xs) > 1 else 0.0
        debug_info = {
            "v_cur_mean":       _mean(v_list),
            "v_cur_std":        _std(v_list),
            "v_cur_min":        min(v_list),
            "v_cur_max":        max(v_list),
            "v_next_mean":      _mean(v_next_list),
            "v_next_std":       _std(v_next_list),
            "bootstrap_mean":   _mean(bootstrap_l),
            "bootstrap_std":    _std(bootstrap_l),
            "td_target_mean":   _mean(tgt_list),
            "td_target_std":    _std(tgt_list),
            "td_error_mean":    _mean(td_err_list),
            "td_error_std":     _std(td_err_list),
            "norm_reward_mean": _mean(norm_r_list),
            "norm_reward_std":  _std(norm_r_list),
            "reward_mean":      _mean(rewards),
            "reward_min":       min(rewards),
            "reward_max":       max(rewards),
            "n_done":           sum(dones),
            "fraction_terminal": sum(dones) / max(len(dones), 1),
            "terminal_target_mean": _mean(terminal_tgts),
            "nonterminal_target_mean": _mean(nonterm_tgts),
        }

    return loss, debug_info


# ═════════════════════════════════════════════════════════════════════════════
# OPTIONAL LINEAR-VFA TEACHER PREFIT
# ═════════════════════════════════════════════════════════════════════════════

def _linear_teacher_score_actions(teacher, state, vehicle, sim_actions: list) -> list[float]:
    """
    Score an arbitrary candidate set with a loaded LinearVFAPolicy.

    The LinearVFAPolicy normally scores candidates inside get_best_action().
    For teacher prefit we need the same scoring logic, but applied to the NN
    candidate pool so the NN learns a ranking over exactly the candidates it
    will later choose between.
    """
    if not getattr(teacher, "_initialized", False):
        teacher._lazy_init(state)

    base_func, base_onsite, base_depot = teacher._extract_inventories(state, vehicle)
    health_base = teacher._compute_health_base(state, vehicle) if teacher._use_health_features else None

    phis = []
    raw_bikes = getattr(vehicle.location, "bikes", [])
    station_bikes = (
        raw_bikes
        if isinstance(raw_bikes, dict)
        else {getattr(b, "bike_id", getattr(b, "id", None)): b for b in raw_bikes}
    )
    vehicle_bikes = {
        getattr(b, "bike_id", getattr(b, "id", None)): b
        for b in vehicle.get_bike_inventory()
    }

    for sim_action in sim_actions:
        functional_pickups = 0
        depot_pickups = 0

        if vehicle.is_at_depot():
            fixed_queue = getattr(vehicle.location, "fixed_queue", {})
            for bike_id in getattr(sim_action, "pick_ups", []) or []:
                if bike_id in fixed_queue:
                    functional_pickups += 1
            depot_dropoffs = sum(
                1 for bike_id in getattr(sim_action, "delivery_bikes", []) or []
                if getattr(vehicle_bikes.get(bike_id), "damage_status", None) == "depot"
            )
            delta_func = -functional_pickups
            delta_depot_cargo = -depot_dropoffs
        else:
            for bike_id in getattr(sim_action, "pick_ups", []) or []:
                bike = station_bikes.get(bike_id)
                if bike and getattr(bike, "damage_status", None) == "depot":
                    depot_pickups += 1
                elif bike:
                    functional_pickups += 1
            delta_func = len(getattr(sim_action, "delivery_bikes", []) or []) - functional_pickups
            delta_depot_cargo = depot_pickups

        delta_onsite_repairs = len(getattr(sim_action, "onsite_repairs", []) or [])
        dest_id = getattr(sim_action, "next_location", getattr(sim_action, "next_station", None))
        try:
            service_time = sim_action.get_action_time(0.0) if hasattr(sim_action, "get_action_time") else 0.0
        except Exception:
            service_time = 0.0
        try:
            travel_time = state.get_vehicle_travel_time(vehicle.location.id, dest_id) if dest_id else 0.0
        except Exception:
            travel_time = 0.0

        projected_time = state.time + service_time + travel_time
        phi = teacher.extract_features(
            state,
            vehicle,
            base_func,
            base_onsite,
            base_depot,
            delta_func,
            delta_depot_cargo,
            delta_onsite_repairs,
            time_remaining=teacher._get_time_remaining(state, vehicle),
            shift_length=teacher._get_shift_length(state, vehicle),
            next_station_id=dest_id,
            eval_time=projected_time,
            projected_time=projected_time,
            dist_to_next=travel_time,
            candidate_action=sim_action,
            health_base=health_base,
        )
        phis.append(phi)

    phis = teacher._prepare_candidate_features(phis)
    return [teacher.value(phi) for phi in phis]


class NNLinearTeacherPrefitPolicy(Policy):
    """Collect NN candidate encodings while a trained Linear VFA chooses actions."""

    def __init__(
        self,
        teacher,
        config: MDPConfig,
        depot_id: Optional[str],
        warmup_end_time: float,
        dataset: list,
        use_action_context: bool = False,
        candidate_wildcards: bool = True,
    ) -> None:
        super().__init__(maintenance_enabled=config.allow_onsite_repairs)
        self.teacher = teacher
        self.config = config
        self.depot_id = depot_id
        self.warmup_end_time = warmup_end_time
        self.dataset = dataset
        self.use_action_context = use_action_context
        self.candidate_wildcards = candidate_wildcards
        self.greedy_policy = (
            GreedyMaintenancePolicy() if self.maintenance_enabled else GreedyPolicy()
        )

    def init_sim(self, simulator) -> None:
        self.teacher.init_sim(simulator)
        self.greedy_policy.init_sim(simulator)

    def get_best_action(self, state, vehicle):
        if state.time < self.warmup_end_time:
            return self.greedy_policy.get_best_action(state, vehicle)

        vehicle_shift_end = getattr(vehicle, "shift_end_time", None)
        mdp_state = extract_mdp_state(
            sim_state=state,
            active_vehicle_id=vehicle.id,
            config=self.config,
            depot_id=self.depot_id,
            shift_end_time=vehicle_shift_end,
        )
        pairs = generate_candidates(
            state=state,
            vehicle=vehicle,
            maintenance_enabled=self.maintenance_enabled,
            return_pairs=True,
            wide_search=True,
            training_mode=self.candidate_wildcards,
        )

        encodings = []
        valid_pairs = []
        for mdp_action, sim_action in pairs:
            try:
                post_state, action_duration, _ = PostDecisionState.apply(mdp_state, mdp_action)
                dest = mdp_action.next_station
                dest_tt = {
                    sid: state.get_vehicle_travel_time(dest, sid)
                    for sid in mdp_state.stations
                }
                encodings.append(encode_state(
                    post_state,
                    dest_travel_times=dest_tt,
                    mdp_action=mdp_action if self.use_action_context else None,
                    action_duration=action_duration,
                ))
                valid_pairs.append((mdp_action, sim_action))
            except Exception:
                continue

        if not valid_pairs:
            return self.greedy_policy.get_best_action(state, vehicle)

        try:
            teacher_values = _linear_teacher_score_actions(
                self.teacher,
                state,
                vehicle,
                [sim_action for _, sim_action in valid_pairs],
            )
        except Exception as exc:
            print(f"  [TEACHER PREFIT] teacher scoring failed; greedy fallback: {str(exc)[:100]}")
            return self.greedy_policy.get_best_action(state, vehicle)

        if len(encodings) > 1:
            best_idx = int(max(range(len(teacher_values)), key=lambda i: teacher_values[i]))
            self.dataset.append((encodings, best_idx, [float(v) for v in teacher_values]))

        best_idx = int(max(range(len(teacher_values)), key=lambda i: teacher_values[i]))
        return valid_pairs[best_idx][1]


def _scores_for_encoded_candidates(model: nn.Module, encodings: list[dict]) -> torch.Tensor:
    station = torch.stack([e["station_block"] for e in encodings]).to(device)
    vehicle = torch.stack([e["vehicle_block"] for e in encodings]).to(device)
    global_context = torch.stack([e["global_context"] for e in encodings]).to(device)
    return model.forward_batch(station, vehicle, global_context).squeeze(1)


def _run_linear_teacher_prefit(
    online_model: nn.Module,
    teacher_path: str,
    config: MDPConfig,
    depot_id: Optional[str],
    warmup_end_time: float,
    seed_offset: int,
    instance_name: str,
    make_sim_config,
    episodes: int,
    steps: int,
    batch_sets: int,
    lr: float,
    use_action_context: bool,
    maintenance_enabled: bool,
    candidate_wildcards: bool,
) -> dict:
    """Pretrain NN candidate rankings from a trained LinearVFAPolicy teacher."""
    if episodes <= 0 or steps <= 0:
        return {"sets": 0, "loss": 0.0, "accuracy": 0.0}

    from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy

    teacher = LinearVFAPolicy.load(
        Path(teacher_path),
        maintenance_enabled=maintenance_enabled,
        config=config,
    )
    dataset = []

    print("\n" + "=" * 72)
    print("  LINEAR-VFA TEACHER PREFIT")
    print("=" * 72)
    print(f"  Teacher           : {teacher_path}")
    print(f"  Collection eps    : {episodes}")
    print(f"  Optimizer steps   : {steps}")
    print(f"  Batch candidate sets: {batch_sets}")
    print(f"  LR                : {lr:g}")
    print("=" * 72)

    for ep in range(episodes):
        policy = NNLinearTeacherPrefitPolicy(
            teacher=teacher,
            config=config,
            depot_id=depot_id,
            warmup_end_time=warmup_end_time,
            dataset=dataset,
            use_action_context=use_action_context,
            candidate_wildcards=candidate_wildcards,
        )
        simulator = run_simulation(
            seed=seed_offset + 50_000 + ep,
            policy=policy,
            duration=24 * EPISODE_DAYS,
            num_vehicles=NUM_VEHICLES,
            instance_name=instance_name,
            config=make_sim_config(),
        )
        sl = _service_level(simulator, policy)
        print(f"  [TEACHER PREFIT] collect ep {ep+1}/{episodes}: sets={len(dataset)} teacher_SL={sl:.4f}")

    if not dataset:
        print("  [TEACHER PREFIT] no candidate sets collected; skipping.")
        return {"sets": 0, "loss": 0.0, "accuracy": 0.0}

    optimizer = optim.Adam(online_model.parameters(), lr=lr, weight_decay=1e-4)
    online_model.train()
    last_loss = 0.0
    last_acc = 0.0

    for step in range(steps):
        batch = random.sample(dataset, min(batch_sets, len(dataset)))
        losses = []
        n_correct = 0
        for encodings, best_idx, _teacher_values in batch:
            scores = _scores_for_encoded_candidates(online_model, encodings)
            target = torch.tensor([best_idx], dtype=torch.long, device=device)
            losses.append(torch.nn.functional.cross_entropy(scores.unsqueeze(0), target))
            if int(torch.argmax(scores).item()) == int(best_idx):
                n_correct += 1
        loss = torch.stack(losses).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(online_model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        last_loss = float(loss.item())
        last_acc = n_correct / max(len(batch), 1)
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            print(
                f"  [TEACHER PREFIT] step {step+1:4d}/{steps} "
                f"loss={last_loss:.4f} top1_acc={last_acc:.3f}"
            )

    return {"sets": len(dataset), "loss": last_loss, "accuracy": last_acc}


# ═════════════════════════════════════════════════════════════════════════════
# NN LEARNING POLICY  (active during the learning phase only)
# ═════════════════════════════════════════════════════════════════════════════

class NNLearningPolicy(Policy):
    """
    Policy that scores candidates with the online NN and records transitions
    to the replay buffer for offline gradient updates.

    This is the LEARNING-PHASE policy. It is never used directly — it is
    always wrapped by NNEpisodeTrainingPolicy, which routes warmup decisions
    to GreedyPolicy and delegates here only once state.time >= warmup_end_time.

    ── Per-decision flow ────────────────────────────────────────────────────

      get_best_action(state, vehicle):
        1. Measure reward r_k since last decision (compute_step_reward Δ)
        2. Extract MDPState from live simulator
        3. Call _generate_candidates(return_pairs=True) → [(MdpAction, sim.Action)]
        4. For each candidate, apply PostDecisionState.apply(mdp_state, mdp_action)
              → deterministic post-decision state S^x
              → encode_state(S^x)  →  {station_block, vehicle_block, global}
              → online_model(S^x)  →  V(S^x) [no grad]
        5. Boltzmann-select based on V values
        6. Push (S^x_prev, r_k, S^x_chosen) to replay buffer
        7. Store S^x_chosen as the new "previous" for the next epoch
        8. Return the chosen sim.Action

    ── Why post-decision states? ────────────────────────────────────────────

      The value function V(S^x) should estimate the future cost of a state
      AFTER the action has been applied, before stochastic demand events.
      Evaluating on the pre-decision state would mix the value of the state
      with the value of the action, and would make the TD target noisier.
      PostDecisionState.apply() computes this deterministically from the
      MDPState and MdpAction without touching the live simulator.
    """

    def __init__(
        self,
        online_model:      nn.Module,
        reward_calc_config,               # reward.RewardConfig — passed from the training loop
        replay_buffer:     ReplayBuffer,
        gamma:             float,
        config:            MDPConfig,
        depot_id:          Optional[str],
        tau:               float = 0.0,   # Boltzmann temperature; 0 = greedy
        verbose:           bool = False,
        reward_normalizer: Optional[RewardNormalizer] = None,
        debug_logger=None,                # MaintenanceDebugLogger — None disables logging
        training_mode:     bool = False,  # True enables training-only wildcard candidate injection
        n_step_return:     int  = N_STEP_RETURN,  # n-step return length; 1=TD(0), 3=default
        use_action_context: bool = False,
    ):
        super().__init__(maintenance_enabled=config.allow_onsite_repairs)

        self.online_model       = online_model
        self.reward_calc_config = reward_calc_config
        self.replay_buffer      = replay_buffer
        self.gamma              = gamma
        self.config             = config
        self.depot_id           = depot_id
        self.tau                = tau
        self.verbose            = verbose
        self._debug_logger      = debug_logger
        self.training_mode      = training_mode
        self.n_step_return      = n_step_return
        self.use_action_context = use_action_context

        # RewardCalculator: initialized lazily on the first get_best_action call
        # because we need the simulator's initial metrics to set the baseline.
        self._reward_calc:    Optional[RewardCalculator] = None

        # Stores the encoded POST-DECISION state from the previous decision epoch.
        # None until the first learning-phase decision.
        self._prev_post_encoded: Optional[dict] = None
        self._prev_time: Optional[float] = None

        # Decision counter — used for verbose debug output.
        self._decision_count: int = 0


        # ── Per-episode diagnostic accumulators ───────────────────────────────
        # V-spread tracker: records (max - min) over candidates at each decision.
        # Used post-episode to diagnose whether the NN distinguishes candidates.
        # _value_spreads   : max(V) - min(V) across candidates, per decision.
        #                    If this is consistently near 0, the NN cannot
        #                    distinguish candidates → Boltzmann collapses to random.
        # _fallback_count    : how many times PostDecisionState.apply() failed
        #                      and fell back to encoding the pre-decision state.
        # _fallback_reasons  : Counter of exception message prefixes — tells you
        #                      which validation check fires most often.
        # _reward_values     : raw step rewards pushed to the buffer (for scale check).
        self._value_spreads:         list = []
        #self._shift_remaining_values: list = []
        self._fallback_count:         int  = 0
        self._fallback_reasons:       dict = {}   # {reason_str: count}
        self._candidate_count:        int  = 0    # total _encode_post_decision calls (denominator for fallback %)
        self._reward_values:          list = []
        self._shift_checked:          bool = False   # print shift_remaining once per episode
        self._maintenance_chosen:     int  = 0    # decisions where chosen action had maintenance op

        # Reward normalizer — shared across episodes, passed in from training loop.
        self.reward_normalizer = reward_normalizer

        # Candidate index tracking: which pool slot (0=nearest, last=most critical) did NN pick?
        # Persistent clustering at 0 → NN collapsed to "always go nearest".
        # Distributed picks → NN is making state-dependent choices.
        self._chosen_indices: list = []

        # Max-deficit diagnostics:
        # _bypass_deficit_values : deficit_ratio of the max-deficit station each decision.
        # _bypass_same_as_dest   : how often max-deficit station == chosen destination.
        self._bypass_deficit_values: list = []
        self._bypass_same_as_dest:   int  = 0
        self._v_pred_values:         list = []   # mean V across candidates, per decision

        # Candidate pool quality audit
        # _pool_sizes         : valid candidate count per decision (after PDS filter)
        # _nn_top1_chosen     : decisions where Boltzmann chose the NN argmax
        # _forced_decisions   : decisions with only 1 valid candidate (no real choice)
        # _pool_has_both      : decisions where pool contained ≥1 maintenance AND ≥1 rebalancing action
        self._pool_sizes:       list = []
        self._nn_top1_chosen:   int  = 0
        self._forced_decisions: int  = 0
        self._pool_has_both:    int  = 0
        self._pool_has_maintenance: int = 0
        self._nn_top_maintenance:   int = 0
        self._nn_top_rebalancing:   int = 0
        self._chosen_maintenance_when_available: int = 0
        self._best_maintenance_values: list = []
        self._best_rebalancing_values: list = []
        self._maintenance_minus_rebalancing_gaps: list = []
        self._maintenance_beats_rebalancing: int = 0
        self._action_type_counts: dict = {}
        self._candidate_type_counts: dict = {}

        # n-step return buffer: sliding window of (enc_state, scaled_reward) pairs.
        # Transitions are not pushed to the replay buffer immediately; instead they
        # accumulate here until N_STEP_RETURN entries are ready, then folded into
        # one n-step transition: G_n = Σ_{i=0}^{n-1} γ^i r_i, pushed as
        # (S^x_0, G_n, S^x_n, n, done=False). Remaining entries at episode end are
        # drained with done=True (no bootstrap) in flush_terminal_transition().
        self._nstep_pending: deque = deque()
        self._nstep_pushed: int = 0       # full n-step transitions pushed this episode
        self._nstep_terminal: int = 0     # short terminal transitions pushed at flush

    # init_sim is intentionally not overridden: generate_candidates() is a
    # standalone function that reads directly from sim.State and sim.Vehicle,
    # so no policy-level initialisation is needed before calling it.

    @staticmethod
    def _action_type(mdp_action) -> str:
        """Coarse action label for entropy diagnostics."""
        has_depot_visit = mdp_action.depot_dropoffs > 0 or mdp_action.load_from_queue > 0
        has_maintenance = mdp_action.depot_removals > 0 or mdp_action.onsite_repairs > 0
        has_rebalancing = mdp_action.rebalancing != 0
        if has_depot_visit:
            return "depot"
        if has_maintenance and has_rebalancing:
            return "mixed"
        if has_maintenance:
            return "maintenance"
        if mdp_action.rebalancing > 0:
            return "deliver"
        if mdp_action.rebalancing < 0:
            return "pickup"
        return "move"

    @staticmethod
    def _is_maintenance_like(mdp_action) -> bool:
        """True for actions that repair, collect, drop off, or reload maintenance-related bikes."""
        return (
            getattr(mdp_action, "depot_removals", 0) > 0
            or getattr(mdp_action, "onsite_repairs", 0) > 0
            or getattr(mdp_action, "depot_dropoffs", 0) > 0
            or getattr(mdp_action, "load_from_queue", 0) > 0
        )

    @staticmethod
    def _is_rebalancing_like(mdp_action) -> bool:
        """True when the action changes functional-bike balance at a normal station."""
        return getattr(mdp_action, "rebalancing", 0) != 0

    @staticmethod
    def _inc_count(counter: dict, key: str, amount: int = 1) -> None:
        counter[key] = counter.get(key, 0) + amount

    def _lazy_init_reward_calc(self, sim_state) -> None:
        """
        Build the RewardCalculator on the first call, synced to current metrics.

        Syncing at first call means we measure rewards RELATIVE to the system
        state at the start of the learning phase, not from episode start.
        """
        if self._reward_calc is None:
            self._reward_calc = RewardCalculator(
                config=copy.deepcopy(self.reward_calc_config),
                gamma=self.gamma,
            )
            # Baseline sync: any starvations during warmup are not counted
            self._reward_calc.compute_step_reward(sim_state.metrics)

    def _encode_post_decision(self, mdp_state: MDPState, mdp_action, sim_state=None) -> dict:
        """
        Compute and encode the post-decision state S^x for a given MdpAction.

        PostDecisionState.apply() is a pure function — it returns a NEW
        MDPState without modifying the live simulator or the original mdp_state.

        Falls back to encoding the current pre-decision state if apply() raises
        a validation error (e.g., action is infeasible due to state rounding).
        """
        self._candidate_count += 1
        try:
            post_state, action_duration, _ = PostDecisionState.apply(mdp_state, mdp_action)
            dest_tt = None
            if sim_state is not None:
                dest = mdp_action.next_station
                dest_tt = {sid: sim_state.get_travel_time(dest, sid) for sid in mdp_state.stations}
            return encode_state(
                post_state,
                dest_travel_times=dest_tt,
                mdp_action=mdp_action if self.use_action_context else None,
                action_duration=action_duration,
            )
        except Exception as _exc:
            self._fallback_count += 1
            reason = str(_exc)[:80]
            self._fallback_reasons[reason] = self._fallback_reasons.get(reason, 0) + 1
            return encode_state(mdp_state, mdp_action=mdp_action if self.use_action_context else None)

    def get_best_action(self, state, vehicle):
        """
        Score candidates with the online NN and record the transition.

        Called by NNEpisodeTrainingPolicy once state.time >= warmup_end_time.
        """
        # Initialize reward calculator on first learning-phase call
        self._lazy_init_reward_calc(state)

        # --- Step 1: measure reward since last decision ---
        # compute_step_reward computes the Δ in (starvations + 0.7·congestions)
        # since the last time it was called, giving r_k for the interval [t_{k-1}, t_k].
        #r_k = (self._reward_calc.compute_step_reward(state.metrics)
               #+ self._reward_calc.compute_fleet_penalty(state))
               
        r_k = (self._reward_calc.compute_step_reward(
                   state.metrics, 
                   executed_action=getattr(self, '_prev_executed_action', None)
               )
               + self._reward_calc.compute_fleet_penalty(state)
               + self._reward_calc.compute_late_shift_penalty(vehicle, state))

        # --- Step 2: snapshot MDP state from live simulator ---
        # Read shift_end_time from the vehicle object — same pattern as
        # LinearVFAPolicy._get_time_remaining(). Without this, shift_remaining
        # in the global context is always 1.0 (a dead constant feature).
        vehicle_shift_end = getattr(vehicle, "shift_end_time", None)
        mdp_state = extract_mdp_state(
            sim_state=state,
            active_vehicle_id=vehicle.id,
            config=self.config,
            depot_id=self.depot_id,
            shift_end_time=vehicle_shift_end,
        )
        # --- Step 3: generate (MdpAction, sim.Action) pairs ---
        # return_pairs=True gives us the MdpAction objects needed for
        # PostDecisionState.apply() below. Imported directly from mdp/candidate_generator.py
        # so NNLearningPolicy has no dependency on LinearVFAPolicy for this step.
        pairs = generate_candidates(
            state=state,
            vehicle=vehicle,
            maintenance_enabled=self.maintenance_enabled,
            return_pairs=True,
            wide_search=True if self.tau == 0.0 else False,
            training_mode=self.training_mode,
        )

        # --- Step 4: score each candidate's post-decision state ---
        '''values         = []
        post_encodings = []
        with torch.no_grad():
            for mdp_action, _ in pairs:
                post_encoded = self._encode_post_decision(mdp_state, mdp_action)
                v = self.online_model(
                    post_encoded["station_block"].to(device),
                    post_encoded["vehicle_block"].to(device),
                    post_encoded["global_context"].to(device),
                )
                values.append(v.item())
                post_encodings.append(post_encoded)'''
                
                
        # --- Step 4: score each candidate's post-decision state ---
        values         = []
        post_encodings = []
        valid_pairs    = [] # Keep track of which sim_actions correspond to valid PDS
        
        with torch.no_grad():
            for mdp_action, sim_action in pairs: # <-- Unpack sim_action here too
                self._candidate_count += 1
                try:
                    post_state, action_duration, _ = PostDecisionState.apply(mdp_state, mdp_action)
                    dest = mdp_action.next_station
                    dest_tt = {
                        sid: state.get_vehicle_travel_time(dest, sid)
                        for sid in mdp_state.stations
                    }
                    post_encoded = encode_state(
                        post_state,
                        dest_travel_times=dest_tt,
                        mdp_action=mdp_action if self.use_action_context else None,
                        action_duration=action_duration,
                    )
                    
                    v = self.online_model(
                        post_encoded["station_block"].to(device),
                        post_encoded["vehicle_block"].to(device),
                        post_encoded["global_context"].to(device),
                    )
                    
                    values.append(v.item())
                    post_encodings.append(post_encoded)
                    valid_pairs.append((mdp_action, sim_action))
                    
                except Exception as _exc:
                    # Log the failure for your diagnostics, but DO NOT add it to the pool
                    self._fallback_count += 1
                    reason = str(_exc)[:80]
                    self._fallback_reasons[reason] = self._fallback_reasons.get(reason, 0) + 1
                    print(f"    [SKIP] Killed invalid action: {reason}")
                    continue
        # --- NEW YIELD DEBUG ---
        if self.verbose:
            print(f"  [DEBUG] PDS Yield: {len(valid_pairs)} valid states out of {len(pairs)} proposed candidates.")
            
        if not valid_pairs:
            print(f"  [WARNING] ZERO valid actions this epoch at t={mdp_state.time:.0f} — falling back to greedy.")
            from policies.greedy_policy_maintenance import GreedyMaintenancePolicy
            return GreedyMaintenancePolicy().get_best_action(state, vehicle)

        # --- Debug: state representation and candidate scores ---
        if self.verbose:
            # make it print more regularly than just episode 0 — every 2nd decision epoch (since some episodes have very few decisions)
            if self._decision_count % 50 == 0:
                # Full state encoding dump on the very first learning decision
                #enc = post_encodings[0] if post_encodings else self._encode_post_decision(mdp_state, pairs[0][0])
                enc = post_encodings[0] if post_encodings else (self._encode_post_decision(mdp_state, valid_pairs[0][0]) if valid_pairs else None)
                if enc is None:
                    return GreedyPolicy().get_best_action(state, vehicle)
                sb = enc["station_block"]   # [N, 4]
                vb = enc["vehicle_block"]   # [M, 5]
                gc = enc["global_context"]  # [8]
                print("\n" + "-" * 60)
                print("  [DEBUG] Learning-phase decision - state encoding")
                print("-" * 60)
                print(f"  station_block   shape : {list(sb.shape)}  (N_stations x 5)")
                print(f"  vehicle_block   shape : {list(vb.shape)}  (M_vehicles x 5)")
                print(f"  global_context  shape : {list(gc.shape)}  (8 features)")
                print()
                print("  station_block  [func | onsite | depot | time_sin | time_cos]")
                for i, row in enumerate(sb.tolist()):
                    sid = sorted(mdp_state.stations.keys())[i]
                    print(f"    station {sid:>4s}: {['%.3f'%x for x in row]}")
                print()
                print("  vehicle_block  [func_cargo | depot_cargo | dest_func | eta | dest_id]")
                for i, row in enumerate(vb.tolist()):
                    vid = sorted(mdp_state.vehicles.keys())[i]
                    print(f"    vehicle {vid:>4s}: {['%.3f'%x for x in row]}")
                print()
                gc_labels = ["time_sin", "time_cos", "starved_ratio", "low_ratio",
                             "broken_ratio", "depot_queue", "shift_remaining", "mean_load"]
                print("  global_context:")
                for label, val in zip(gc_labels, gc.tolist()):
                    print(f"    {label:<15s}: {val:.4f}")
                print("-" * 60)
            # Every decision: brief candidate-value summary
            if not values:
                return GreedyPolicy().get_best_action(state, vehicle)
            v_min, v_max = min(values), max(values)
            print(
                f"  [DEBUG] decision {self._decision_count:4d} | "
                f"t={mdp_state.time:.0f}min | "
                f"{len(values)} candidates | "
                f"V min={v_min:.4f} max={v_max:.4f} | "
                f"r_k={r_k:.4f}"
            )

        # Accumulate per-decision spread and mean V for episode-level diagnostics.
        if len(values) > 1:
            self._value_spreads.append(max(values) - min(values))
        if values:
            self._v_pred_values.append(sum(values) / len(values))

        self._decision_count += 1

        # --- Step 5: Action selection (Boltzmann if tau > 0, else greedy) ---
        idx = _boltzmann_select(values, self.tau)
        chosen_sim_action    = valid_pairs[idx][1]
        chosen_post_encoded  = post_encodings[idx]

        if self.verbose:
            sorted_vals = sorted(values, reverse=True)
            chosen_rank = sorted_vals.index(values[idx]) if values else 0
            print(f"             chosen idx={idx} rank={chosen_rank+1}/{len(values)} | V={values[idx]:.4f} spread={max(values)-min(values):.4f}")

        # Track which pool slot was chosen (0=nearest station, last=most critical).
        self._chosen_indices.append(idx)

        # --- Candidate pool quality audit ---
        n_pool = len(valid_pairs)
        self._pool_sizes.append(n_pool)
        if n_pool == 1:
            self._forced_decisions += 1
        nn_top1_idx = int(max(range(len(values)), key=lambda i: values[i]))
        if idx == nn_top1_idx:
            self._nn_top1_chosen += 1
        top_mdp = valid_pairs[nn_top1_idx][0]
        chosen_mdp = valid_pairs[idx][0]
        has_maint = any(self._is_maintenance_like(a) for a, _ in valid_pairs)
        has_rebal = any(self._is_rebalancing_like(a) for a, _ in valid_pairs)
        if has_maint:
            self._pool_has_maintenance += 1
        if self._is_maintenance_like(top_mdp):
            self._nn_top_maintenance += 1
        if self._is_rebalancing_like(top_mdp):
            self._nn_top_rebalancing += 1
        if has_maint and self._is_maintenance_like(chosen_mdp):
            self._chosen_maintenance_when_available += 1
        maint_values = [
            v for v, (a, _) in zip(values, valid_pairs)
            if self._is_maintenance_like(a)
        ]
        rebal_values = [
            v for v, (a, _) in zip(values, valid_pairs)
            if self._is_rebalancing_like(a)
        ]
        if maint_values:
            self._best_maintenance_values.append(max(maint_values))
        if rebal_values:
            self._best_rebalancing_values.append(max(rebal_values))
        if maint_values and rebal_values:
            gap = max(maint_values) - max(rebal_values)
            self._maintenance_minus_rebalancing_gaps.append(gap)
            if gap > 0.0:
                self._maintenance_beats_rebalancing += 1
        if has_maint and has_rebal:
            self._pool_has_both += 1

        for cand_mdp, _ in valid_pairs:
            self._inc_count(self._candidate_type_counts, self._action_type(cand_mdp))

        # --- Max-deficit diagnostics ---
        best_def_val, best_def_sid = -float('inf'), None
        chosen_dest = valid_pairs[idx][0].next_station if valid_pairs else None
        for sid, inv in mdp_state.stations.items():
            d = (inv.target - inv.functional) / max(inv.capacity, 1)
            if d > best_def_val:
                best_def_val, best_def_sid = d, sid
        if best_def_sid is not None:
            self._bypass_deficit_values.append(best_def_val)
            if chosen_dest == best_def_sid:
                self._bypass_same_as_dest += 1

        # Track maintenance action frequency
        self._inc_count(self._action_type_counts, self._action_type(chosen_mdp))
        if self._is_maintenance_like(chosen_mdp):
            self._maintenance_chosen += 1

        # Maintenance debug logger — log per-decision stats
        if self._debug_logger is not None:
            _all_cands = list(zip([p[0] for p in valid_pairs], values))
            self._debug_logger.log_training_decision(
                sim_time=mdp_state.time,
                chosen_mdp_action=valid_pairs[idx][0],
                all_candidates=_all_cands,
                reward=r_k,
                mdp_state=mdp_state,
            )

        # --- Step 6: push transition to replay buffer ---
        # We push (S^x_{k-1}, r_k, S^x_k) where:
        #   S^x_{k-1} = post-decision state from the PREVIOUS decision epoch
        #   r_k       = reward observed in the interval [t_{k-1}, t_k]
        #   S^x_k     = post-decision state of the CHOSEN action at t_k
        #
        # On the very first learning-phase call, _prev_post_encoded is None
        # (no previous learning-phase decision), so we skip this push.
        if self._prev_post_encoded is not None:
            # Reward shaping: division by elapsed step time to yield a rate.
            elapsed_time = mdp_state.time - (self._prev_time if self._prev_time is not None else mdp_state.time)
            elapsed_time = max(1.0, elapsed_time)

            #scaled_r = r_k / elapsed_time
            scaled_r = r_k

            self._reward_values.append(scaled_r)   # track raw (but scaled) reward for episode-level stats
            if self.reward_normalizer is not None:
                self.reward_normalizer.update(scaled_r)

            # --- n-step return accumulation (sliding window) ---
            # Append (S^x_{k-1}, r_k) to the pending window.
            # Once the window holds N_STEP_RETURN entries, fold the oldest n
            # transitions into one n-step transition and push to the replay buffer:
            #   G_n = r_0 + γ·r_1 + ... + γ^{n-1}·r_{n-1}
            #   push (S^x_0, G_n, S^x_n, n, done=False)
            # Then slide the window by popping the oldest entry.
            self._nstep_pending.append((self._prev_post_encoded, scaled_r))
            if len(self._nstep_pending) >= self.n_step_return:
                accum_r = sum(
                    self.gamma ** i * self._nstep_pending[i][1]
                    for i in range(self.n_step_return)
                )
                enc_0 = self._nstep_pending[0][0]
                self.replay_buffer.push(
                    encoded_cur=enc_0,
                    reward=accum_r,
                    encoded_next=chosen_post_encoded,
                    n_steps=self.n_step_return,
                    done=False,
                )
                self._nstep_pushed += 1
                self._nstep_pending.popleft()

        # --- Step 7: store chosen post-decision state for next epoch ---
        self._prev_post_encoded = chosen_post_encoded
        self._prev_time         = mdp_state.time
        
        # 🔴 NEW: Save what action we actually took, so we can get rewarded for it next epoch
        # PostDecisionState.apply returns (new_state, duration, executed_action)
        _, _, executed = PostDecisionState.apply(mdp_state, valid_pairs[idx][0])
        self._prev_executed_action = executed

        return chosen_sim_action

    def flush_terminal_transition(self, sim_state) -> None:
        """
        Push the final transition at episode end (done=True).

        At the very last decision epoch of an episode, there is no "next"
        decision to form the S^x_next of the transition. We close the episode
        by pushing a terminal transition with done=True so the TD target is
        just r (no bootstrap). Call this after run_simulation returns.
        """
        if self._prev_post_encoded is not None and self._reward_calc is not None:
            final_r = (self._reward_calc.compute_step_reward(
                           sim_state.metrics,
                           executed_action=getattr(self, "_prev_executed_action", None),
                       )
                       + self._reward_calc.compute_fleet_penalty(sim_state))

            elapsed_time = sim_state.time - (self._prev_time if self._prev_time is not None else sim_state.time)
            elapsed_time = max(1.0, elapsed_time)

            scaled_r = final_r

            if self.reward_normalizer is not None:
                self.reward_normalizer.update(scaled_r)

            # Append the terminal step to the pending window.
            self._nstep_pending.append((self._prev_post_encoded, scaled_r))

            # Drain all remaining pending entries as terminal transitions (done=True).
            # Each entry at position i gets an accumulated return from i to the end
            # of the window — a short return of length (len - i) steps.  Since the
            # episode has ended there is no bootstrap term, so the TD target is just
            # the accumulated return with no V_next added.
            pending = list(self._nstep_pending)
            for i in range(len(pending)):
                n_remaining = len(pending) - i
                accum_r = sum(
                    self.gamma ** j * pending[i + j][1]
                    for j in range(n_remaining)
                )
                enc_0 = pending[i][0]
                self.replay_buffer.push(
                    encoded_cur=enc_0,
                    reward=accum_r,
                    encoded_next=enc_0,   # not used; done=True suppresses bootstrap
                    n_steps=n_remaining,
                    done=True,
                )
                self._nstep_terminal += 1

            print(
                f"  [N-STEP] n={self.n_step_return} | "
                f"full pushes={self._nstep_pushed} | "
                f"terminal flush={self._nstep_terminal} | "
                f"total={self._nstep_pushed + self._nstep_terminal}"
            )

        # Reset for the next episode
        self._prev_post_encoded = None
        self._prev_time         = None
        self._reward_calc       = None
        self._nstep_pending.clear()
        self._prev_executed_action = None  # 🔴 NEW: Clear the stored action


# ═════════════════════════════════════════════════════════════════════════════
# EPISODE TRAINING POLICY  (warmup + learning wrapper)
# ═════════════════════════════════════════════════════════════════════════════

class NNEpisodeTrainingPolicy(Policy):
    """
    Episodic wrapper that routes vehicle decisions to:

      - GreedyPolicy      during the warm-up phase  (state.time < warmup_end_time)
      - NNLearningPolicy  during the learning phase (state.time >= warmup_end_time)

    This mirrors EpisodeTrainingPolicy in LinearVFAPolicy.py exactly,
    just substituting NNLearningPolicy for LinearVFAPolicy.

    A new instance is created for every episode (since NNLearningPolicy
    resets per-episode state: _prev_post_encoded, _reward_calc). The shared
    online_model weights and replay_buffer persist across episodes.
    """

    def __init__(
        self,
        nn_learning_policy: NNLearningPolicy,
        greedy_policy:      GreedyPolicy,
        warmup_end_time:    float,
    ) -> None:
        super().__init__(maintenance_enabled=True)

        self.nn_policy       = nn_learning_policy
        self.greedy_policy   = greedy_policy
        self.warmup_end_time = warmup_end_time

    def init_sim(self, simulator) -> None:
        """Forward simulator reference to both inner policies."""
        self.nn_policy.init_sim(simulator)
        self.greedy_policy.init_sim(simulator)

    def get_best_action(self, state, vehicle):
        """
        Route to the appropriate policy based on current simulation time.

        Before warmup_end_time: GreedyPolicy (no recording, no gradient updates).
        After warmup_end_time:  NNLearningPolicy (Boltzmann + replay recording).
        """
        if state.time < self.warmup_end_time:
            return self.greedy_policy.get_best_action(state, vehicle)
        else:
            # --- NEW: Snapshot metrics at the exact moment warmup ends ---
            if not hasattr(self, 'warmup_trips_snapshot'):
                self.warmup_trips_snapshot = state.metrics.get_aggregate_value("trips") or 1
                self.warmup_starvations_snapshot = state.metrics.get_aggregate_value("starvations") or 0
                self.warmup_congestions_snapshot = state.metrics.get_aggregate_value("long congestions") or 0
            # -------------------------------------------------------------
            
            # Learning phase: NN scoring + Boltzmann exploration
            return self.nn_policy.get_best_action(state, vehicle)

    def __repr__(self) -> str:
        return "NNEpisodeTrainingPolicy(greedy)"
    
    
def _service_level(simulator, episode_policy) -> float:
    """Computes the service level strictly for the LEARNING phase (Days 3-14)."""
    m = simulator.state.metrics
    
    # Total metrics at the end of the episode
    total_trips = m.get_aggregate_value("trips") or 1
    total_starv = m.get_aggregate_value("starvations") or 0
    total_cong  = m.get_aggregate_value("long congestions") or 0

    # Retrieve the snapshots taken at the end of warmup
    warmup_trips = getattr(episode_policy, 'warmup_trips_snapshot', 0)
    warmup_starv = getattr(episode_policy, 'warmup_starvations_snapshot', 0)
    warmup_cong  = getattr(episode_policy, 'warmup_congestions_snapshot', 0)

    # Isolate the Neural Network's true performance
    nn_trips = max(1, total_trips - warmup_trips)
    nn_starv = max(0, total_starv - warmup_starv)
    nn_cong  = max(0, total_cong - warmup_cong)

    return 1.0 - ((nn_starv + nn_cong) / nn_trips)


# ─────────────────────────────────────────────────────────────────────────────
# Greedy evaluation episode
# ─────────────────────────────────────────────────────────────────────────────

def _run_greedy_eval(
    online_model:    nn.Module,
    reward_calc_config,
    config:          MDPConfig,
    depot_id:        Optional[str],
    seed:            int,
    instance_name:   str,
    gamma:           float,
    warmup_end_time: float,
    odometer_stats_path: Optional[str] = None,
    odometer_sampling_method: str = "triangular",
    odometer_sampling_bounds: str = "p05-p95",
    use_action_context: bool = False,
) -> float:
    """
    Run one evaluation episode with tau=0 (pure greedy NN) and no buffer writes.

    This is the key check for whether the NN has learned anything useful.
    During training episodes, Boltzmann noise masks the NN quality.
    Here tau=0 forces the NN to pick its argmax action every time,
    so the returned service level purely reflects the NN's value estimates.

    Returns the learning-phase service level (same formula as _service_level).
    """
    eval_learning = NNLearningPolicy(
        online_model=online_model,
        reward_calc_config=reward_calc_config,
        replay_buffer=ReplayBuffer(max_size=1),   # dummy — nothing pushed with tau=0 eval
        gamma=gamma,
        config=config,
        depot_id=depot_id,
        tau=0.0,       # <-- pure greedy: NN argmax only
        verbose=False,
        training_mode=False,
        use_action_context=use_action_context,
    )
    greedy_policy  = GreedyPolicy()
    eval_episode   = NNEpisodeTrainingPolicy(
        nn_learning_policy=eval_learning,
        greedy_policy=greedy_policy,
        warmup_end_time=warmup_end_time,
    )
    sim_config = SimulationConfig()
    if odometer_stats_path is not None:
        sim_config.odometer_stats_path = odometer_stats_path
    sim_config.odometer_sampling_method = odometer_sampling_method
    sim_config.odometer_sampling_bounds = odometer_sampling_bounds
    eval_sim   = run_simulation(
        seed          = seed,
        policy        = eval_episode,
        duration      = 24 * EPISODE_DAYS,
        num_vehicles  = NUM_VEHICLES,
        instance_name = instance_name,
        config        = sim_config,
    )
    return _service_level(eval_sim, eval_episode)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def train_nn_rollout(
    num_episodes:           int   = NUM_EPISODES,
    save_path:              Path  = None,
    seed_offset:            int   = 0,
    instance_name:          str   = INSTANCE_NAME,
    gamma:                  float = GAMMA,
    lr_start:               float = LR_START,
    lr_end:                 float = LR_END,
    tau_start:              float = TAU_START,
    tau_end:                float = TAU_END,
    tau_anneal_episodes:    int   = TAU_ANNEAL_EPISODES,
    target_update_freq:     int   = TARGET_UPDATE_FREQ,
    polyak:                 float = POLYAK,
    batch_size:             int   = BATCH_SIZE,
    buffer_size:            int   = BUFFER_SIZE,
    n_step_return:          int   = N_STEP_RETURN,           # 1=TD(0) kills deadly triad
    use_maintenance_shaping: bool = USE_MAINTENANCE_SHAPING, # False=base reward only
    depot_id:               Optional[str] = None,
    reward_calc_config=None,
    run_label:              Optional[str] = None,
    value_hidden_dims:      list = None,
    enable_logging:         bool = False,
    maintenance_enabled:    bool = None,
    odometer_stats_path:    Optional[str] = None,
    odometer_sampling_method: str = "triangular",
    odometer_sampling_bounds: str = "p05-p95",
    reward_norm_mode:       str = "ema",
    fixed_reward_scale:     float = 10.0,
    use_station_id_embedding: bool = False,
    station_id_embed_dim:    int = 8,
    use_action_context:      bool = False,
    use_station_spotlights:  bool = False,
    use_demand_horizon:      bool = False,
    use_global_health:       bool = False,
    linear_teacher_prefit_path: Optional[str] = None,
    teacher_prefit_episodes: int = 0,
    teacher_prefit_steps:    int = 500,
    teacher_prefit_batch_sets: int = 16,
    teacher_prefit_lr:       float = 1e-4,
    candidate_wildcards:     bool = True,
    max_updates_per_episode: int = 400,
) -> nn.Module:
    """
    Run the full episodic NN training loop.

    Each episode:
      1. Create NNEpisodeTrainingPolicy (wraps GreedyPolicy + NNLearningPolicy)
      2. run_simulation() for EPISODE_DAYS days
         • Days 1–WARMUP_DAYS:  GreedyPolicy (no recording)
         • Days (WARMUP_DAYS+1)–EPISODE_DAYS: NNLearningPolicy records transitions
      3. After simulation: batch gradient updates from replay buffer
      4. Every TARGET_UPDATE_FREQ episodes: copy online → target network

    Args:
        num_episodes        : total training episodes
        save_path           : output .pt path (auto-generated if None)
        seed_offset         : episode i uses seed = seed_offset + i
        instance_name       : simulator instance (e.g. "TD_W34_old")
        gamma               : TD discount factor
        lr_start / lr_end   : Adam learning rate bounds (linear decay)
        target_update_freq  : online → target copy frequency (episodes)
        batch_size          : gradient update mini-batch size
        buffer_size         : replay buffer capacity
        depot_id            : depot station ID
        reward_calc_config  : RewardConfig for building the step-reward calculator.
                              If None, a default config is created from a
                              temporary LinearVFAPolicy instance.

    Returns:
        Trained NNValueNetwork (online model, eval mode).
    """
    random.seed(seed_offset)
    torch.manual_seed(seed_offset)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    set_encoder_options(
        use_action_context=use_action_context,
        use_demand_horizon=use_demand_horizon,
        use_global_health=use_global_health,
    )

    # ── Build online model + frozen target ────────────────────────────────────
    # The target model starts as an exact copy of the online model.
    # It is NEVER updated by backprop; only by hard weight copies every
    # TARGET_UPDATE_FREQ episodes.
    #online_model = build_nn_value_network().to(device)
    online_model = build_nn_value_network(
        value_hidden_dims=value_hidden_dims,
        station_feature_dim=get_station_feature_dim(use_demand_horizon),
        global_feature_dim=get_global_feature_dim(use_action_context, use_global_health),
        station_id_embed_dim=station_id_embed_dim if use_station_id_embedding else 0,
        use_station_spotlights=use_station_spotlights,
    ).to(device)
    target_model = copy.deepcopy(online_model).to(device)
    for param in target_model.parameters():
        param.requires_grad = False   # no gradient tracking in target

    # ── Optimizer and shared replay buffer ───────────────────────────────────
    # The replay buffer is shared across ALL episodes: transitions from earlier
    # episodes (when the policy was exploratory) stay in the buffer and continue
    # to contribute to gradient updates in later episodes.
    optimizer         = None
    replay_buffer     = ReplayBuffer(max_size=buffer_size)
    reward_normalizer = RewardNormalizer(
        mode=reward_norm_mode,
        fixed_scale=fixed_reward_scale,
    )   # shared across all episodes

    # ── Warmup end time (absolute simulation minutes) ─────────────────────────
    # Mirrors the calculation in train_vfa.py:
    #   sim_start_min = 5h × 60 = 300 min
    #   warmup_end_time = 300 + WARMUP_DAYS × 1440
    sim_start_min   = timeInMinutes(hours=START_HOUR)
    warmup_end_time = sim_start_min + WARMUP_DAYS * 24 * 60

    _maintenance = ENABLE_COMPONENT_FAILURES if maintenance_enabled is None else maintenance_enabled
    import settings as _settings_mod
    _settings_mod.ENABLE_COMPONENT_FAILURES = _maintenance
    config = MDPConfig.full_maintenance() if _maintenance else MDPConfig.no_maintenance()

    def _make_sim_config() -> SimulationConfig:
        sim_config = SimulationConfig()
        if odometer_stats_path is not None:
            sim_config.odometer_stats_path = odometer_stats_path
        sim_config.odometer_sampling_method = odometer_sampling_method
        sim_config.odometer_sampling_bounds = odometer_sampling_bounds
        return sim_config

    # ── Reward calculator config ──────────────────────────────────────────────
    # NNLearningPolicy needs a RewardConfig to build its step-reward calculator.
    # If none is provided, borrow one from a throw-away LinearVFAPolicy instance
    # (its weights are never used; only the reward config matters).
    if reward_calc_config is None:
        from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
        _tmp_vfa = LinearVFAPolicy(
            learning_mode=False,
            maintenance_enabled=_maintenance,
        )
        reward_calc_config = _tmp_vfa.reward_calc.config

    # Apply experiment switch: override shaping flag regardless of how config was created.
    reward_calc_config.use_maintenance_shaping = use_maintenance_shaping

    # ── Resolve depot_id from instance if not supplied ────────────────────────
    # LinearVFAPolicy resolves this in init_sim via state.get_closest_depot().
    # We do the same here with a 1-step probe so every episode gets the correct
    # depot_id — without it, is_at_depot() always returns False and every depot
    # visit falls through to apply_at_normal_station(), causing a KeyError.
    if depot_id is None:
        _probe = run_simulation(
            seed=seed_offset,
            policy=GreedyPolicy(),
            duration=1,
            num_vehicles=NUM_VEHICLES,
            instance_name=instance_name,
            config=_make_sim_config(),
        )
        _probe_vehicles = _probe.state.get_vehicles()
        depot_id = _probe.state.get_closest_depot(_probe_vehicles[0]) if _probe_vehicles else None
        print(f"  Depot ID          : {depot_id!r}  (auto-resolved from instance)")

    prefit_info = {"sets": 0, "loss": 0.0, "accuracy": 0.0}
    if linear_teacher_prefit_path and teacher_prefit_episodes > 0 and teacher_prefit_steps > 0:
        prefit_info = _run_linear_teacher_prefit(
            online_model=online_model,
            teacher_path=linear_teacher_prefit_path,
            config=config,
            depot_id=depot_id,
            warmup_end_time=warmup_end_time,
            seed_offset=seed_offset,
            instance_name=instance_name,
            make_sim_config=_make_sim_config,
            episodes=teacher_prefit_episodes,
            steps=teacher_prefit_steps,
            batch_sets=teacher_prefit_batch_sets,
            lr=teacher_prefit_lr,
            use_action_context=use_action_context,
            maintenance_enabled=_maintenance,
            candidate_wildcards=candidate_wildcards,
        )
        target_model.load_state_dict(online_model.state_dict())
        print("  [TEACHER PREFIT] target network reset to prefitted online weights")

    optimizer = optim.Adam(online_model.parameters(), lr=lr_start, weight_decay=1e-4)

    # ── Metrics tracking ──────────────────────────────────────────────────────
    learning_curve = []   # written to checkpoint; also mirrored to CSV below
    best_greedy_sl  = -float("inf")   # track best greedy SL for model saving
    best_greedy_path: Path = None
    t0 = time.time()

    # Open CSV log — one row per episode, written incrementally so a partial
    # run is still readable if training is interrupted on the cluster.
    ts_run    = datetime.now().strftime("%Y%m%d_%H%M%S")
    def _tag_float(value: float) -> str:
        return f"{value:g}".replace("-", "m").replace(".", "p")

    _reward_norm_suffix = f"rnorm{reward_norm_mode}"
    if reward_norm_mode == "fixed":
        _reward_norm_suffix += f"_rs{_tag_float(fixed_reward_scale)}"
    _encoder_suffix = ""
    if use_station_id_embedding:
        _encoder_suffix += f"_sid{station_id_embed_dim}"
    if use_action_context:
        _encoder_suffix += "_actctx"
    if use_station_spotlights:
        _encoder_suffix += "_spots"
    if use_demand_horizon:
        _encoder_suffix += "_dh"
    if use_global_health:
        _encoder_suffix += "_ghealth"
    if not candidate_wildcards:
        _encoder_suffix += "_nowild"
    if max_updates_per_episode != 400:
        _encoder_suffix += f"_upd{max_updates_per_episode}"
    if linear_teacher_prefit_path and teacher_prefit_episodes > 0 and teacher_prefit_steps > 0:
        _encoder_suffix += f"_ltpref{teacher_prefit_episodes}e{teacher_prefit_steps}s"
    _base_label = (
        run_label if run_label else
        f"n{n_step_return}_buf{buffer_size}_poly{polyak}"
        f"_taue{tau_end}_maint{'1' if use_maintenance_shaping else '0'}"
    )
    _label = f"_{_base_label}{_encoder_suffix}_{_reward_norm_suffix}"
    csv_path  = SAVE_DIR / f"training_log_seed{seed_offset}{_label}_{ts_run}.csv"
    CSV_FIELDS = [
        "episode", "mean_loss", "lr", "tau", "polyak",
        "reward_norm_mode", "fixed_reward_scale",
        "station_id_embedding", "station_id_embed_dim", "action_context",
        "station_spotlights", "demand_horizon", "global_health",
        "candidate_wildcards",
        "max_updates_per_episode",
        "linear_teacher_prefit", "teacher_prefit_sets", "teacher_prefit_acc",
        "service_level", "buffer_size", "n_updates", "elapsed_s",
        "mean_value_spread",  # max(V)-min(V) per decision; near 0 = NN not discriminating
        "mean_reward",        # raw step reward mean (un-normalized); scale check
        "pct_zero_reward",    # % decisions with r=0; high = too sparse
        "fallback_rate",      # % PostDecisionState.apply() failures; > 5% = data quality issue
        "reward_norm_std",    # running std of all rewards seen so far; tracks normalization scale
        "pct_idx0",           # % decisions where NN picked pool slot 0 (nearest); ~100% = collapsed
        "mean_chosen_idx",    # mean pool index chosen; 0=always nearest, ~3.5=uniform
        "pct_maintenance",    # % decisions where chosen action had depot_removals>0 or onsite_repairs>0
        "greedy_sl",          # tau=0 eval SL — true NN quality, unconfounded by exploration
        "bypass_mean_deficit",  # mean deficit_ratio of worst station; >0 = real starvation signal
        "bypass_pct_same",      # % decisions where chosen dest == max-deficit station
        "mean_v_pred",        # mean V(S^x) across all candidate evaluations; should stabilize
        "n_step_terminal",    # short terminal transitions at episode end; high = many partial n-steps
        "reward_ema_mean",    # EMA running mean of step rewards; confirm it updates each episode
        "prediction_mean",    # V(Sx) mean on final sampled training batch
        "prediction_std",
        "prediction_min",
        "prediction_max",
        "target_mean",        # TD target mean/std on final sampled training batch
        "target_std",
        "v_next_mean",
        "v_next_std",
        "bootstrap_mean",
        "bootstrap_std",
        "td_error_mean",
        "td_error_std",
        "fraction_terminal_transitions",
        "terminal_target_mean",
        "nonterminal_target_mean",
        "grad_norm",          # kept for compatibility; same as grad_norm_after_clip
        "grad_norm_before_clip",
        "grad_norm_after_clip",
        "mean_pool_size",     # avg valid candidates per decision; <2 = NN has no real choice
        "pct_forced",         # % decisions with exactly 1 valid candidate (forced)
        "pct_nn_top1_chosen", # % decisions where Boltzmann picked NN argmax (agreement w/ greedy)
        "pct_pool_mixed",     # % decisions where pool had ≥1 maintenance AND ≥1 rebalancing action
        "pct_pool_has_maintenance",
        "pct_nn_top_maintenance",
        "pct_nn_top_rebalancing",
        "pct_chosen_maintenance_when_available",
        "best_maintenance_value_mean",
        "best_rebalancing_value_mean",
        "maint_minus_rebal_gap_mean",
        "pct_maintenance_beats_rebalancing",
        "action_entropy",
        "candidate_entropy",
    ]
    csv_file   = csv_path.open("w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    csv_writer.writeheader()
    print(f"  CSV log           : {csv_path}")

    log_path     = csv_path.with_suffix(".log")
    debug_logger = MaintenanceDebugLogger(log_path, log_detail_every=1) if enable_logging else None
    print(f"  Maintenance log   : {log_path}")

    print("=" * 72)
    print("  NN ROLLOUT TRAINING")
    print("=" * 72)
    print(f"  Episodes          : {num_episodes}")
    print(f"  Episode duration  : {EPISODE_DAYS} days  "
          f"(warm-up = {WARMUP_DAYS}d,  learning = {LEARNING_DAYS}d)")
    print(f"  Selection         : Boltzmann  tau {tau_start:.3f} -> {tau_end:.3f}  (linear anneal)")
    print(f"  LR schedule       : {lr_start:.5f}  ->  {lr_end:.5f}  (linear)")
    print(f"  gamma                 : {gamma}")
    print(f"  Target update     : every {target_update_freq} episodes")
    print(f"  Polyak τ          : {polyak:.3f}  (for soft target updates, if enabled)")
    print(f"  Replay buffer     : {buffer_size:,}  |  batch: {batch_size}")
    print(f"  Max updates/ep    : {max_updates_per_episode}")
    print(f"  Reward norm       : {reward_norm_mode}"
          f"{f'  scale={fixed_reward_scale:g}' if reward_norm_mode == 'fixed' else ''}")
    print(f"  Station ID embed  : {use_station_id_embedding}"
          f"{f'  dim={station_id_embed_dim}' if use_station_id_embedding else ''}")
    print(f"  Action context    : {use_action_context}")
    print(f"  Station spotlights: {use_station_spotlights}")
    print(f"  Demand horizon    : {use_demand_horizon}")
    print(f"  Global health     : {use_global_health}")
    print(f"  Candidate wildcards: {candidate_wildcards}")
    print(f"  Warmup policy     : {'GreedyMaintenancePolicy' if _maintenance else 'GreedyPolicy'}")
    if linear_teacher_prefit_path and teacher_prefit_episodes > 0 and teacher_prefit_steps > 0:
        print(f"  Linear teacher    : {linear_teacher_prefit_path}")
        print(
            f"  Teacher prefit    : {teacher_prefit_episodes} collect eps, "
            f"{teacher_prefit_steps} steps, batch_sets={teacher_prefit_batch_sets}, "
            f"lr={teacher_prefit_lr:g}"
        )
    print(f"  Instance          : {instance_name}")
    n_params = sum(p.numel() for p in online_model.parameters())
    print(f"  Model parameters  : {n_params:,}")
    print("=" * 72 + "\n")

    # ── Episode loop ──────────────────────────────────────────────────────────
    _buffer_ready_logged = False   # print once when buffer first becomes ready
    for ep in range(num_episodes):

        # --- Decay lr and tau (both linear) for this episode ---
        frac        = ep / max(num_episodes - 1, 1)
        current_lr  = lr_start  + (lr_end  - lr_start)  * frac
        tau_frac    = min(ep, tau_anneal_episodes - 1) / max(tau_anneal_episodes - 1, 1)
        current_tau = tau_start + (tau_end  - tau_start) * tau_frac
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # --- Build per-episode policies ---
        # NNLearningPolicy is created fresh each episode to reset:
        #   _prev_post_encoded and _reward_calc (per-episode state).
        # The shared online_model weights and replay_buffer persist.
        nn_learning = NNLearningPolicy(
            online_model=online_model,
            reward_calc_config=reward_calc_config,
            replay_buffer=replay_buffer,
            gamma=gamma,
            config=config,
            depot_id=depot_id,
            tau=current_tau,
            verbose=(ep == VERBOSE_EPISODE),
            reward_normalizer=reward_normalizer,
            debug_logger=debug_logger,
            training_mode=candidate_wildcards,
            n_step_return=n_step_return,
            use_action_context=use_action_context,
        )
        greedy_policy = GreedyMaintenancePolicy() if _maintenance else GreedyPolicy()
        episode_policy = NNEpisodeTrainingPolicy(
            nn_learning_policy=nn_learning,
            greedy_policy=greedy_policy,
            warmup_end_time=warmup_end_time,
        )

        print(f"\n{'='*50}")
        print(f"EPISODE {ep+1}/{num_episodes} | lr={current_lr:.5f}")
        print(f"{'='*50}")
        if debug_logger is not None:
            debug_logger.log_episode_start(ep + 1, num_episodes, current_tau, current_lr)
    

        # --- Run the episode ---
        sim_config = _make_sim_config()
        simulator  = run_simulation(
            seed          = seed_offset + ep,
            policy        = episode_policy,
            duration      = 24 * EPISODE_DAYS,
            num_vehicles  = NUM_VEHICLES,
            instance_name = instance_name,
            config        = sim_config,
        )

        # Push the terminal transition for the last learning-phase decision
        nn_learning.flush_terminal_transition(simulator.state)

        # NEW: Calculate service level for this episode
        sl = _service_level(simulator, episode_policy)

        # --- Gradient updates from replay buffer (post-episode) ---
        # We do one gradient pass per episode rather than per transition.
        # This keeps training fast while still leveraging the diversity of the buffer.
        episode_losses     = []
        episode_grad_norms_before = []
        episode_grad_norms_after  = []
        n_updates          = 0   # tracked for CSV logging

        buffer_ready = replay_buffer.ready(min_size=MIN_BUFFER_SIZE)
        if not buffer_ready:
            print(f"  [DEBUG] Buffer filling: {len(replay_buffer)}/{MIN_BUFFER_SIZE} transitions — skipping gradient update")
        elif not _buffer_ready_logged:
            print(f"  [DEBUG] Buffer ready at episode {ep+1} ({len(replay_buffer)} transitions) — gradient updates begin")
            _buffer_ready_logged = True

        # Decide whether to collect detailed debug stats this episode.
        # Collected on the LAST gradient step to avoid the overhead on every step.
        collect_debug_this_ep = (
            DEBUG_EVERY > 0 and (ep + 1) % DEBUG_EVERY == 0
        )
        last_debug_info = None

        if buffer_ready:
            # Number of gradient steps: proportional to episode length relative to batch size,
            # capped to prevent overfitting on a single episode's transitions.
            n_updates = min(
                max(1, len(replay_buffer) // batch_size) * 4,
                max_updates_per_episode,
            )

            if ep == VERBOSE_EPISODE:
                print(f"  [DEBUG] gradient update: {n_updates} steps x batch={batch_size}")

            for update_i in range(n_updates):
                batch = replay_buffer.sample(batch_size)

                # Collect debug info on the final update step of a diagnostic episode.
                want_debug = (update_i == n_updates - 1)

                # Forward: online model; backward: clip gradients; step
                loss, debug_info = _compute_td_loss(
                    online_model=online_model,
                    target_model=target_model,
                    batch=batch,
                    gamma=gamma,
                    reward_normalizer=reward_normalizer,
                    collect_debug=want_debug,
                )
                if debug_info is not None:
                    last_debug_info = debug_info

                optimizer.zero_grad()
                loss.backward()
                _gn_before = nn.utils.clip_grad_norm_(online_model.parameters(), GRAD_CLIP_NORM)
                _gn_before = float(_gn_before.item() if hasattr(_gn_before, "item") else _gn_before)
                _gn_after = sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in online_model.parameters() if p.grad is not None
                ) ** 0.5
                episode_grad_norms_before.append(_gn_before)
                episode_grad_norms_after.append(_gn_after)
                optimizer.step()
                episode_losses.append(loss.item())

                # Polyak update every gradient step (standard SAC/TD3 schedule).
                # Per-step τ=0.005 keeps the target responsive without losing
                # the stabilising lag; the effective target lag depends on
                # --max_updates_per_episode.
                # Previously this ran once per episode (τ per episode), leaving
                # the target 80% at random initialisation after 43 episodes.
                # Polyak update every gradient step
                if polyak > 0.0:
                    with torch.no_grad():
                        for _op, _tp in zip(online_model.parameters(), target_model.parameters()):
                            _tp.data.mul_(1.0 - polyak).add_(polyak * _op.data)
                            
        mean_loss = sum(episode_losses) / max(len(episode_losses), 1)
        mean_grad_norm_before = (
            round(sum(episode_grad_norms_before) / max(len(episode_grad_norms_before), 1), 4)
            if episode_grad_norms_before else 0.0
        )
        mean_grad_norm_after = (
            round(sum(episode_grad_norms_after) / max(len(episode_grad_norms_after), 1), 4)
            if episode_grad_norms_after else 0.0
        )

        # --- Target network update ---
        # Polyak updates now run inside the gradient loop (per gradient step).
        # This block handles the fallback hard-copy when POLYAK=0, and prints
        # a periodic diagnostic showing how far online and target have drifted.
        if polyak > 0.0:
            if ep == 0 or (DEBUG_EVERY > 0 and (ep + 1) % DEBUG_EVERY == 0):
                sample_p = next(online_model.parameters())
                sample_t = next(target_model.parameters())
                diff = (sample_p - sample_t).norm().item()
                print(f"  [POLYAK] ep={ep+1}  τ={polyak}/step  |online-target|={diff:.5f}")
        else:
            if (ep + 1) % target_update_freq == 0:
                target_model.load_state_dict(online_model.state_dict())
                print(f"  -> Target network updated (episode {ep+1})")

        # ── Periodic diagnostic summary ───────────────────────────────────────
        # Printed every DEBUG_EVERY episodes to help diagnose flat service level.
        if collect_debug_this_ep:
            import statistics as _stat
            spreads = nn_learning._value_spreads
            rewards = nn_learning._reward_values
            n_dec   = nn_learning._decision_count
            n_fall  = nn_learning._fallback_count

            print(f"\n  {'─'*60}")
            print(f"  [DIAGNOSTIC] Episode {ep+1}")
            print(f"  {'─'*60}")

            # 1. Candidate value spread — is the NN discriminating?
            if spreads:
                mean_sp = _stat.mean(spreads)
                med_sp  = _stat.median(spreads)
                max_sp  = max(spreads)
                pct_tiny = sum(1 for s in spreads if s < 0.001) / len(spreads) * 100
                print(
                    f"  Value spread (max-min across candidates):"
                    f"  mean={mean_sp:.5f}  median={med_sp:.5f}  max={max_sp:.5f}"
                    f"  | {pct_tiny:.0f}% decisions < 0.001"
                    f"\n  -> If mean spread < 0.01: NN not discriminating candidates"
                )
            else:
                print("  Value spread: no data (no multi-candidate decisions?)")

            # 2. Reward distribution — is the signal non-trivial?
            if rewards:
                mean_r = _stat.mean(rewards)
                min_r  = min(rewards)
                max_r  = max(rewards)
                pct_z  = sum(1 for r in rewards if r == 0.0) / len(rewards) * 100
                print(
                    f"  Raw rewards pushed to buffer:"
                    f"  mean={mean_r:.4f}  min={min_r:.4f}  max={max_r:.4f}"
                    f"  | {pct_z:.0f}% are exactly 0"
                    f"\n  -> If >80% zero: reward is too sparse to drive learning"
                )
            else:
                print("  Rewards: no transitions pushed this episode")

            # 3. Max-deficit bypass diagnostics
            bypass_vals = nn_learning._bypass_deficit_values
            n_dec_total = max(nn_learning._decision_count, 1)
            if bypass_vals:
                mean_def = _stat.mean(bypass_vals)
                pct_same = nn_learning._bypass_same_as_dest / n_dec_total * 100
                pct_pos  = sum(1 for v in bypass_vals if v > 0) / len(bypass_vals) * 100
                print(
                    f"  Max-deficit bypass:"
                    f"  mean_deficit={mean_def:.3f}  {pct_pos:.0f}% decisions had a starving max-deficit station"
                    f"  | {pct_same:.0f}% times chosen dest == max-deficit (bypass redundant those times)"
                    f"\n  -> If pct_same is low: routing is often not selecting the most starved station"
                    f"\n  -> If mean_deficit < 0: no station is starving — bypass signal is weak"
                )

            '''# 4. TD loss internals — full reward chain breakdown
            if last_debug_info is not None:
                d = last_debug_info
                # -- raw n-step returns in buffer
                print(
                    f"  RAW n-step returns (buffer sample):"
                    f"  mean={d['reward_mean']:.4f}  min={d['reward_min']:.4f}  max={d['reward_max']:.4f}"
                )
                # -- after normalize(): should now be O(1), not O(0.01)
                print(
                    f"  NORMALIZED rewards (after normalize()):"
                    f"  mean={d['norm_reward_mean']:.4f}  std={d['norm_reward_std']:.4f}"
                    f"  [{d['norm_reward_min']:.4f}, {d['norm_reward_max']:.4f}]"
                    f"\n  -> TARGET: should be O(1). If still <0.1: normalization still too aggressive."
                )
                # -- bootstrap contribution: γ^n * V_target(S_next)
                print(
                    f"  BOOTSTRAP term γ^n·V_next:"
                    f"  mean={d['bootstrap_mean']:.4f}  std={d['bootstrap_std']:.4f}"
                    f"  | V_next mean={d['v_next_mean']:.4f}"
                )
                # -- signal balance: reward vs bootstrap
                r_abs  = abs(d['norm_reward_mean'])
                b_abs  = abs(d['bootstrap_mean'])
                total  = r_abs + b_abs + 1e-8
                print(
                    f"  SIGNAL BALANCE: reward={r_abs/total*100:.1f}%  bootstrap={b_abs/total*100:.1f}%"
                    f"\n  -> TARGET: reward should be >30% of signal. <5% = bootstrap domination."
                )
                # -- what the network currently predicts
                print(
                    f"  V(S^x) online  (last batch):"
                    f"  mean={d['v_cur_mean']:.4f}  std={d['v_cur_std']:.4f}"
                    f"  [{d['v_cur_min']:.4f}, {d['v_cur_max']:.4f}]"
                )
                print(
                    f"  TD targets     (last batch):"
                    f"  mean={d['td_target_mean']:.4f}  std={d['td_target_std']:.4f}"
                    f"  | done={d['n_done']}/{batch_size}"
                )
                if abs(d['v_cur_mean'] - d['td_target_mean']) < 0.001 and d['v_cur_std'] < 0.01:
                    print(f"  !! V and targets near-identical with tiny std — network may be collapsed !!")
            else:
                print("  TD internals: no gradient updates this episode")'''
            # 3. TD loss internals — are V and targets at similar scales?
            if last_debug_info is not None:
                d = last_debug_info
                print(
                    f"  V(S^x) online  (last batch):  "
                    f"mean={d['v_cur_mean']:.4f}  std={d['v_cur_std']:.4f}"
                    f"  [{d['v_cur_min']:.4f}, {d['v_cur_max']:.4f}]"
                )
                print(
                    f"  TD targets     (last batch):  "
                    f"mean={d['td_target_mean']:.4f}  std={d['td_target_std']:.4f}"
                )
                print(
                    f"  Normalized rewards (last batch):  "
                    f"mean={d['reward_mean']:.5f}  "
                    f"min={d['reward_min']:.5f}  max={d['reward_max']:.5f}"
                    f"  | done transitions: {d['n_done']}/{batch_size}"
                    f"\n  -> If V and TD targets both near 0 with tiny std:"
                    f" reward normalization may be over-squashing the signal"
                )
            else:
                print("  TD internals: no gradient updates this episode")

            # 3b. Candidate index distribution — is NN collapsed to "always nearest"?
            idxs = nn_learning._chosen_indices
            if idxs:
                pct0 = sum(1 for i in idxs if i == 0) / len(idxs) * 100
                mean_idx = sum(idxs) / len(idxs)
                print(
                    f"  Candidate index: pct_idx0={pct0:.1f}%  mean_idx={mean_idx:.2f}"
                    f"  (uniform random → mean≈3.5, collapsed → mean≈0)"
                    f"\n  -> >70% idx0: NN is collapsing to 'always go nearest'"
                )

            # 3c. Maintenance-vs-rebalancing ranking diagnostics.
            if nn_learning._pool_sizes:
                n_decisions = max(len(nn_learning._pool_sizes), 1)
                maint_gap = (
                    _stat.mean(nn_learning._maintenance_minus_rebalancing_gaps)
                    if nn_learning._maintenance_minus_rebalancing_gaps else 0.0
                )
                maint_beats = (
                    nn_learning._maintenance_beats_rebalancing
                    / max(len(nn_learning._maintenance_minus_rebalancing_gaps), 1) * 100
                )
                print(
                    f"  Maintenance ranking:"
                    f" pool_has_maint={nn_learning._pool_has_maintenance / n_decisions * 100:.1f}%"
                    f" | NN-top maint={nn_learning._nn_top_maintenance / n_decisions * 100:.1f}%"
                    f" | NN-top rebal={nn_learning._nn_top_rebalancing / n_decisions * 100:.1f}%"
                    f" | chosen maint when available="
                    f"{nn_learning._chosen_maintenance_when_available / max(nn_learning._pool_has_maintenance, 1) * 100:.1f}%"
                    f"\n  Best-value gap: V(best maint)-V(best rebal)={maint_gap:+.4f}"
                    f" | maintenance beats rebal={maint_beats:.1f}%"
                    f"\n  -> If pool_has_maint is high but NN-top maint is low/negative gap:"
                    f" the NN is actively ranking maintenance below rebalancing."
                )

            # 3d. Reward normalizer stats
            print(
                f"  Reward normalizer: mean={reward_normalizer.mean:.3f}"
                f"  std={reward_normalizer.std:.3f}"
                f"  n={reward_normalizer._n}"
            )

            # 4. PostDecisionState fallback rate + breakdown by reason
            if n_dec > 0:
                fall_pct = n_fall / n_dec * 100
                print(
                    f"  PostDecisionState fallbacks: {n_fall}/{n_dec} ({fall_pct:.1f}%)"
                    f"\n  -> If >5%: corrupted S^x inputs in replay buffer"
                )
                reasons = nn_learning._fallback_reasons
                if reasons:
                    print("  Fallback reasons (top 5):")
                    for msg, cnt in sorted(reasons.items(), key=lambda x: -x[1])[:5]:
                        print(f"    [{cnt:4d}x]  {msg}")

            print(f"  {'─'*60}\n")

        # ── Optional greedy evaluation run ────────────────────────────────────
        # Runs a separate tau=0 episode to measure true NN quality,
        # unconfounded by Boltzmann exploration noise.
        eval_sl = None
        if (EVAL_GREEDY_EVERY > 0
                and (ep + 1) % EVAL_GREEDY_EVERY == 0
                and reward_calc_config is not None):
            online_model.eval()
            eval_sl = _run_greedy_eval(
                online_model=online_model,
                reward_calc_config=reward_calc_config,
                config=config,
                depot_id=depot_id,
                seed=EVAL_GREEDY_SEED,
                instance_name=instance_name,
                gamma=gamma,
                warmup_end_time=warmup_end_time,
                odometer_stats_path=odometer_stats_path,
                odometer_sampling_method=odometer_sampling_method,
                odometer_sampling_bounds=odometer_sampling_bounds,
                use_action_context=use_action_context,
            )
            online_model.train()
            print(
                f"  [GREEDY EVAL ep {ep+1}]  SL={eval_sl:.4f}"
                f"  (training SL this ep={sl:.4f})"
                f"\n  -> If greedy eval SL >> training SL: NN learned but"
                f" exploration is masking it in training metrics"
            )
            # Save best-greedy-SL checkpoint whenever a new high is reached
            if eval_sl > best_greedy_sl:
                best_greedy_sl = eval_sl
                best_greedy_path = SAVE_DIR / f"nn_model_best_greedy_seed{seed_offset}{_label}_{ts_run}.pt"
                torch.save({
                    "episode":              ep + 1,
                    "model_state":          online_model.state_dict(),
                    "target_state":         target_model.state_dict(),
                    "optimizer_state":      optimizer.state_dict(),
                    "learning_curve":       learning_curve,
                    "station_feature_dim":  online_model.station_feature_dim,
                    "vehicle_feature_dim":  online_model.vehicle_feature_dim,
                    "global_feature_dim":   online_model.global_feature_dim,
                    "reward_norm_mode":     reward_norm_mode,
                    "fixed_reward_scale":   fixed_reward_scale,
                    "use_station_id_embedding": use_station_id_embedding,
                    "station_id_embed_dim":  station_id_embed_dim if use_station_id_embedding else 0,
                    "use_action_context":    use_action_context,
                    "use_station_spotlights": use_station_spotlights,
                    "use_demand_horizon":    use_demand_horizon,
                    "use_global_health":     use_global_health,
                    "linear_teacher_prefit_path": linear_teacher_prefit_path,
                    "teacher_prefit_episodes": teacher_prefit_episodes,
                    "teacher_prefit_steps":    teacher_prefit_steps,
                    "teacher_prefit_sets":     int(prefit_info.get("sets", 0)),
                    "teacher_prefit_accuracy": float(prefit_info.get("accuracy", 0.0)),
                    "best_greedy_sl":       best_greedy_sl,
                }, best_greedy_path)
                print(f"  -> New best greedy SL={best_greedy_sl:.4f} — saved to {best_greedy_path.name}")

        # Maintenance debug log — episode summary
        if debug_logger is not None:
            debug_logger.log_episode_summary(
                sl=sl,
                mean_loss=mean_loss,
                buffer_size=len(replay_buffer),
                greedy_sl=eval_sl,
            )

        # --- Logging ---
        learning_curve.append({
            "episode":       ep + 1,
            "mean_loss":     mean_loss,
            "lr":            current_lr,
            "tau":           current_tau,
            "buffer_size":   len(replay_buffer),
            "service_level": sl,
        })
        elapsed = time.time() - t0
        print(
            f"  Ep {ep+1:3d}/{num_episodes} | "
            f"lr={current_lr:.5f} | tau={current_tau:.3f} | "
            f"loss={mean_loss:.4f} | SL={sl:.4f} | "
            f"buffer={len(replay_buffer):,} | "
            f"t={elapsed:.0f}s"
        )

        # Compute per-episode encoding diagnostics from nn_learning accumulators.
        import statistics as _stat
        _spreads  = nn_learning._value_spreads
        _rewards  = nn_learning._reward_values
        #_shifts   = nn_learning._shift_remaining_values
        _n_fall   = nn_learning._fallback_count
        _n_cand   = nn_learning._candidate_count   # true denominator for fallback %

        #mean_shift_remaining = round(_stat.mean(_shifts),  4) if _shifts  else 1.0
        mean_value_spread = round(_stat.mean(_spreads), 6) if _spreads else 0.0
        mean_reward       = round(_stat.mean(_rewards), 5) if _rewards else 0.0
        pct_zero_reward   = round(
            sum(1 for r in _rewards if r == 0.0) / max(len(_rewards), 1) * 100, 1
        )
        fallback_rate     = round(_n_fall / max(_n_cand, 1) * 100, 2)
        _idxs             = nn_learning._chosen_indices
        pct_idx0          = round(sum(1 for i in _idxs if i == 0) / max(len(_idxs), 1) * 100, 1)
        mean_chosen_idx   = round(sum(_idxs) / max(len(_idxs), 1), 2)
        pct_maintenance   = round(nn_learning._maintenance_chosen / max(len(_idxs), 1) * 100, 1)
        _bypass_vals      = nn_learning._bypass_deficit_values
        bypass_mean_deficit = round(_stat.mean(_bypass_vals), 4) if _bypass_vals else 0.0
        bypass_pct_same   = round(nn_learning._bypass_same_as_dest / max(len(_idxs), 1) * 100, 1)
        _vpreds           = nn_learning._v_pred_values
        mean_v_pred       = round(_stat.mean(_vpreds), 5) if _vpreds else 0.0

        _psizes           = nn_learning._pool_sizes
        _n_dec            = max(len(_psizes), 1)
        mean_pool_size    = round(_stat.mean(_psizes), 2) if _psizes else 0.0
        pct_forced        = round(nn_learning._forced_decisions / _n_dec * 100, 1)
        pct_nn_top1       = round(nn_learning._nn_top1_chosen   / _n_dec * 100, 1)
        pct_pool_mixed    = round(nn_learning._pool_has_both     / _n_dec * 100, 1)
        pct_pool_has_maintenance = round(nn_learning._pool_has_maintenance / _n_dec * 100, 1)
        pct_nn_top_maintenance = round(nn_learning._nn_top_maintenance / _n_dec * 100, 1)
        pct_nn_top_rebalancing = round(nn_learning._nn_top_rebalancing / _n_dec * 100, 1)
        pct_chosen_maintenance_when_available = round(
            nn_learning._chosen_maintenance_when_available
            / max(nn_learning._pool_has_maintenance, 1) * 100,
            1,
        )
        best_maintenance_value_mean = round(
            _stat.mean(nn_learning._best_maintenance_values), 6
        ) if nn_learning._best_maintenance_values else 0.0
        best_rebalancing_value_mean = round(
            _stat.mean(nn_learning._best_rebalancing_values), 6
        ) if nn_learning._best_rebalancing_values else 0.0
        maint_minus_rebal_gap_mean = round(
            _stat.mean(nn_learning._maintenance_minus_rebalancing_gaps), 6
        ) if nn_learning._maintenance_minus_rebalancing_gaps else 0.0
        pct_maintenance_beats_rebalancing = round(
            nn_learning._maintenance_beats_rebalancing
            / max(len(nn_learning._maintenance_minus_rebalancing_gaps), 1) * 100,
            1,
        )

        def _entropy(counts: dict) -> float:
            total = sum(counts.values())
            if total <= 0:
                return 0.0
            ent = 0.0
            for count in counts.values():
                p = count / total
                if p > 0.0:
                    ent -= p * math.log(p)
            return round(ent, 4)

        action_entropy = _entropy(nn_learning._action_type_counts)
        candidate_entropy = _entropy(nn_learning._candidate_type_counts)
        _td = last_debug_info or {}
        _tdv = lambda key: round(float(_td.get(key, 0.0)), 6) if _td else ""

        # Write one CSV row per episode (flushed immediately so partial runs are readable)
        csv_writer.writerow({
            "episode":          ep + 1,
            "mean_loss":        round(mean_loss,    6),
            "lr":               round(current_lr,   6),
            "tau":              round(current_tau,  4),
            "polyak":           polyak,
            "reward_norm_mode":  reward_norm_mode,
            "fixed_reward_scale": fixed_reward_scale if reward_norm_mode == "fixed" else "",
            "station_id_embedding": int(use_station_id_embedding),
            "station_id_embed_dim": station_id_embed_dim if use_station_id_embedding else 0,
            "action_context":    int(use_action_context),
            "station_spotlights": int(use_station_spotlights),
            "demand_horizon":    int(use_demand_horizon),
            "global_health":     int(use_global_health),
            "candidate_wildcards": int(candidate_wildcards),
            "max_updates_per_episode": max_updates_per_episode,
            "linear_teacher_prefit": int(bool(linear_teacher_prefit_path and teacher_prefit_episodes > 0 and teacher_prefit_steps > 0)),
            "teacher_prefit_sets": int(prefit_info.get("sets", 0)),
            "teacher_prefit_acc": round(float(prefit_info.get("accuracy", 0.0)), 4),
            "service_level":    round(sl,           4),
            "buffer_size":      len(replay_buffer),
            "n_updates":        n_updates,
            "elapsed_s":        round(elapsed,      1),
            "mean_value_spread": mean_value_spread,
            "mean_reward":       mean_reward,
            "pct_zero_reward":   pct_zero_reward,
            "fallback_rate":     fallback_rate,
            "reward_norm_std":   round(reward_normalizer.std, 4),
            "pct_idx0":          pct_idx0,
            "mean_chosen_idx":   mean_chosen_idx,
            "pct_maintenance":   pct_maintenance,
            "greedy_sl":         round(eval_sl, 4) if eval_sl is not None else "",
            "bypass_mean_deficit": bypass_mean_deficit,
            "bypass_pct_same":     bypass_pct_same,
            "mean_v_pred":       mean_v_pred,
            "n_step_terminal":   nn_learning._nstep_terminal,
            "reward_ema_mean":   round(reward_normalizer.mean, 4),
            "prediction_mean":   _tdv("v_cur_mean"),
            "prediction_std":    _tdv("v_cur_std"),
            "prediction_min":    _tdv("v_cur_min"),
            "prediction_max":    _tdv("v_cur_max"),
            "target_mean":       _tdv("td_target_mean"),
            "target_std":        _tdv("td_target_std"),
            "v_next_mean":       _tdv("v_next_mean"),
            "v_next_std":        _tdv("v_next_std"),
            "bootstrap_mean":    _tdv("bootstrap_mean"),
            "bootstrap_std":     _tdv("bootstrap_std"),
            "td_error_mean":     _tdv("td_error_mean"),
            "td_error_std":      _tdv("td_error_std"),
            "fraction_terminal_transitions": _tdv("fraction_terminal"),
            "terminal_target_mean": _tdv("terminal_target_mean"),
            "nonterminal_target_mean": _tdv("nonterminal_target_mean"),
            "grad_norm":         mean_grad_norm_after,
            "grad_norm_before_clip": mean_grad_norm_before,
            "grad_norm_after_clip":  mean_grad_norm_after,
            "mean_pool_size":    mean_pool_size,
            "pct_forced":        pct_forced,
            "pct_nn_top1_chosen": pct_nn_top1,
            "pct_pool_mixed":    pct_pool_mixed,
            "pct_pool_has_maintenance": pct_pool_has_maintenance,
            "pct_nn_top_maintenance": pct_nn_top_maintenance,
            "pct_nn_top_rebalancing": pct_nn_top_rebalancing,
            "pct_chosen_maintenance_when_available": pct_chosen_maintenance_when_available,
            "best_maintenance_value_mean": best_maintenance_value_mean,
            "best_rebalancing_value_mean": best_rebalancing_value_mean,
            "maint_minus_rebal_gap_mean": maint_minus_rebal_gap_mean,
            "pct_maintenance_beats_rebalancing": pct_maintenance_beats_rebalancing,
            "action_entropy":    action_entropy,
            "candidate_entropy": candidate_entropy,
        })
        csv_file.flush()

        # --- Periodic checkpoint ---
        if (ep + 1) % 50 == 0:
            ck_path = SAVE_DIR / f"nn_model_ep{ep+1:04d}_seed{seed_offset}{_label}_{ts_run}.pt"
            torch.save({
                "episode":              ep + 1,
                "model_state":          online_model.state_dict(),
                "target_state":         target_model.state_dict(),
                "optimizer_state":      optimizer.state_dict(),
                "learning_curve":       learning_curve,
                "station_feature_dim":  online_model.station_feature_dim,
                "vehicle_feature_dim":  online_model.vehicle_feature_dim,
                "global_feature_dim":   online_model.global_feature_dim,
                "reward_norm_mode":     reward_norm_mode,
                "fixed_reward_scale":   fixed_reward_scale,
                "use_station_id_embedding": use_station_id_embedding,
                "station_id_embed_dim":  station_id_embed_dim if use_station_id_embedding else 0,
                "use_action_context":    use_action_context,
                "use_station_spotlights": use_station_spotlights,
                "use_demand_horizon":    use_demand_horizon,
                "use_global_health":     use_global_health,
                "candidate_wildcards":   candidate_wildcards,
                "max_updates_per_episode": max_updates_per_episode,
                "linear_teacher_prefit_path": linear_teacher_prefit_path,
                "teacher_prefit_episodes": teacher_prefit_episodes,
                "teacher_prefit_steps":    teacher_prefit_steps,
                "teacher_prefit_sets":     int(prefit_info.get("sets", 0)),
                "teacher_prefit_accuracy": float(prefit_info.get("accuracy", 0.0)),
            }, ck_path)
            print(f"  -> Checkpoint saved -> {ck_path}")

    if debug_logger is not None:
        debug_logger.close()

    # ── Final save ────────────────────────────────────────────────────────────
    if save_path is None:
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = SAVE_DIR / f"nn_model_final_seed{seed_offset}{_label}_{ts}.pt"

    torch.save({
        "episode":             num_episodes,
        "model_state":         online_model.state_dict(),
        "learning_curve":      learning_curve,
        "station_feature_dim": online_model.station_feature_dim,
        "vehicle_feature_dim": online_model.vehicle_feature_dim,
        "global_feature_dim":  online_model.global_feature_dim,
        "value_hidden_dims":   online_model.value_hidden_dims,
        "reward_norm_mode":    reward_norm_mode,
        "fixed_reward_scale":  fixed_reward_scale,
        "use_station_id_embedding": use_station_id_embedding,
        "station_id_embed_dim": station_id_embed_dim if use_station_id_embedding else 0,
        "use_action_context":   use_action_context,
        "use_station_spotlights": use_station_spotlights,
        "use_demand_horizon":   use_demand_horizon,
        "use_global_health":    use_global_health,
        "candidate_wildcards":  candidate_wildcards,
        "max_updates_per_episode": max_updates_per_episode,
        "linear_teacher_prefit_path": linear_teacher_prefit_path,
        "teacher_prefit_episodes": teacher_prefit_episodes,
        "teacher_prefit_steps": teacher_prefit_steps,
        "teacher_prefit_sets": int(prefit_info.get("sets", 0)),
        "teacher_prefit_accuracy": float(prefit_info.get("accuracy", 0.0)),
    }, save_path)

    csv_file.close()

    total_elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print("  NN TRAINING COMPLETE")
    print("=" * 72)
    print(f"  Total time    : {total_elapsed / 60:.1f} min")
    print(f"  Final model   : {save_path}")
    if best_greedy_path is not None:
        print(f"  Best greedy   : {best_greedy_path.name}  (SL={best_greedy_sl:.4f})")
    print(f"  CSV log saved : {csv_path}")
    print("=" * 72 + "\n")

    online_model.eval()
    return online_model


# ═════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_nn_model(checkpoint_path: str) -> nn.Module:
    """
    Reconstruct and load a trained NNValueNetwork from a checkpoint file.

    Checkpoints saved by train_nn_rollout() always include the architecture
    dimensions alongside the state dict, so the model can be reconstructed
    without knowing the hyperparameters upfront.

    Args:
        checkpoint_path : path to a .pt file saved by train_nn_rollout().

    Returns:
        NNValueNetwork with loaded weights, set to eval mode.

    Usage (in NNRolloutPolicy):
        nn = load_nn_model("NN/models/nn_model_final_seed0_20250414_1200.pt")
        policy = NNRolloutPolicy(nn_model=nn, candidate_vfa=vfa, ...)
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state"]
    use_action_context = bool(checkpoint.get("use_action_context", False))
    use_demand_horizon = bool(checkpoint.get("use_demand_horizon", False))
    use_global_health = bool(checkpoint.get("use_global_health", False))
    use_station_spotlights = bool(checkpoint.get("use_station_spotlights", False))
    station_id_embed_dim = int(checkpoint.get("station_id_embed_dim", 0) or 0)
    set_encoder_options(
        use_action_context=use_action_context,
        use_demand_horizon=use_demand_horizon,
        use_global_health=use_global_health,
    )

    # Infer value_hidden_dims from weights when the key is absent or None
    stored_dims = checkpoint.get("value_hidden_dims")
    if not stored_dims:
        keys = sorted(
            (k for k in state_dict if k.startswith("value_mlp.net.") and k.endswith(".weight")),
            key=lambda k: int(k.split(".")[2]),
        )
        stored_dims = [state_dict[k].shape[0] for k in keys[:-1]]

    if USE_VFA_FEATURES:
        from policies.sjovik_sund.NN.nn_model import FlatNNValueNetwork
        model = FlatNNValueNetwork(
            input_dim=checkpoint["global_feature_dim"],
            hidden_dims=stored_dims,
        )
    else:
        from policies.sjovik_sund.NN.nn_model import NNValueNetwork
        model = NNValueNetwork(
            station_feature_dim=checkpoint["station_feature_dim"],
            vehicle_feature_dim=checkpoint["vehicle_feature_dim"],
            global_feature_dim=checkpoint["global_feature_dim"],
            value_hidden_dims=stored_dims,
            station_id_embed_dim=station_id_embed_dim,
            use_station_spotlights=use_station_spotlights,
        )
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded NN model from {checkpoint_path} "
          f"(episode {checkpoint.get('episode', '?')})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# GREEDY BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_greedy_baseline(
    n_seeds: int = 5,
    seed_offset: int = 1000,
    instance_name: str = INSTANCE_NAME,
) -> float:
    """
    Run pure GreedyPolicy for EPISODE_DAYS days on n_seeds seeds.
    Reports full-episode SL (all 14 days, not warmup-subtracted).
    Use this to judge whether the NN greedy_sl is above the greedy ceiling.
    If NN greedy_sl ≈ this number, the NN has learned nothing useful.
    """
    sls = []
    for seed in range(seed_offset, seed_offset + n_seeds):
        sim = run_simulation(
            seed=seed,
            policy=GreedyPolicy(),
            duration=24 * EPISODE_DAYS,
            num_vehicles=NUM_VEHICLES,
            instance_name=instance_name,
            config=SimulationConfig(),
        )
        m = sim.state.metrics
        trips = m.get_aggregate_value("trips") or 1
        starv = m.get_aggregate_value("starvations") or 0
        cong  = m.get_aggregate_value("long congestions") or 0
        sl = 1.0 - (starv + cong) / trips
        sls.append(sl)
        print(f"  seed={seed}  greedy_sl={sl:.4f}")

    mean_sl = sum(sls) / len(sls)
    print(f"\n  GreedyPolicy baseline (n={n_seeds}): mean SL = {mean_sl:.4f}")
    print(f"  NN greedy_sl of ~0.888 → delta vs greedy = {0.888 - mean_sl:+.4f}")
    return mean_sl


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Offline episodic TD(0) training of a neural network value function",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes",              type=int,   default=NUM_EPISODES)
    parser.add_argument("--seed",                  type=int,   default=1000)
    parser.add_argument("--instance",              type=str,   default=INSTANCE_NAME)
    parser.add_argument("--gamma",                 type=float, default=GAMMA)
    parser.add_argument("--lr_start",              type=float, default=LR_START)
    parser.add_argument("--lr_end",                type=float, default=LR_END)
    parser.add_argument("--tau_start",             type=float, default=TAU_START)
    parser.add_argument("--tau_end",               type=float, default=TAU_END)
    parser.add_argument("--tau_anneal_episodes",   type=int,   default=TAU_ANNEAL_EPISODES)
    parser.add_argument("--target_update_freq",    type=int,   default=TARGET_UPDATE_FREQ)
    parser.add_argument("--polyak",                type=float, default=POLYAK)
    parser.add_argument("--batch_size",            type=int,   default=BATCH_SIZE)
    parser.add_argument("--buffer_size",           type=int,   default=BUFFER_SIZE)
    parser.add_argument("--n_step",                type=int,   default=N_STEP_RETURN,
                        help="n-step return length. 1=TD(0) kills deadly triad.")
    parser.add_argument("--maintenance_shaping",   action=argparse.BooleanOptionalAction,
                        default=USE_MAINTENANCE_SHAPING,
                        help="Enable +10/+5/+5 maintenance reward shaping.")
    parser.add_argument("--run_label",             type=str,   default=None)
    parser.add_argument("--save",                  type=str,   default=None)
    parser.add_argument("--depot_id",              type=str,   default='D0')
    parser.add_argument("--congestion_weight",     type=float, default=-1.0)
    parser.add_argument("--fleet_degradation_weight", type=float, default=-0.5,
                        help="Penalty weight for broken_bikes / total_fleet at each decision.")
    parser.add_argument("--maintenance",           action=argparse.BooleanOptionalAction,
                        default=ENABLE_COMPONENT_FAILURES)
    parser.add_argument("--valuehead_hidden_dims", type=int,   nargs="+", default=None,
                        help="Single arch, e.g. --valuehead_hidden_dims 128 64 32. Omit to run all.")
    parser.add_argument("--odometer_stats",        type=str,   default=None,
                        help="Aggregated steady-state odometer CSV used to initialize every episode.")
    parser.add_argument("--odometer_sampling_method", type=str,
                        choices=["triangular", "uniform", "truncated-normal"],
                        default="triangular")
    parser.add_argument("--odometer_sampling_bounds", type=str,
                        choices=["p05-p95", "min-max"],
                        default="p05-p95")
    parser.add_argument("--reward_norm", type=str,
                        choices=["ema", "fixed", "none"],
                        default="ema",
                        help=(
                            "Reward normalization for TD targets. "
                            "ema=current moving mean/std normalizer; "
                            "fixed=reward/(fixed_reward_scale*discount_sum); "
                            "none=raw reward."
                        ))
    parser.add_argument("--fixed_reward_scale", type=float, default=10.0,
                        help="Scale used when --reward_norm fixed.")
    parser.add_argument("--station_id_embedding", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Enable learned station-ID embeddings in the Deep Sets station branch.")
    parser.add_argument("--station_id_embed_dim", type=int, default=8,
                        help="Embedding dimension used when --station_id_embedding is enabled.")
    parser.add_argument("--action_context", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Append proposed-action features to the global context.")
    parser.add_argument("--station_spotlights", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Append raw top starving/congested/depot-broken/onsite-broken station spotlights.")
    parser.add_argument("--demand_horizon", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Append per-station projected flow/risk horizon features.")
    parser.add_argument("--global_health", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Append global inventory-health features to the global context.")
    parser.add_argument("--candidate_wildcards", action=argparse.BooleanOptionalAction,
                        default=True,
                        help=(
                            "Inject random wildcard routing candidates during NN training. "
                            "Evaluation/greedy policies always disable this for deterministic candidate pools."
                        ))
    parser.add_argument("--max_updates_per_episode", type=int, default=400,
                        help="Hard cap on gradient updates after each episode.")
    parser.add_argument("--linear_teacher_prefit", type=str, default=None,
                        help=(
                            "Path to a trained LinearVFAPolicy model. When set with "
                            "--teacher_prefit_episodes > 0, pretrain the NN to imitate "
                            "the Linear VFA's candidate rankings from raw NN encodings."
                        ))
    parser.add_argument("--teacher_prefit_episodes", type=int, default=0,
                        help="Teacher-driven episodes used to collect candidate-ranking data before TD training.")
    parser.add_argument("--teacher_prefit_steps", type=int, default=500,
                        help="Supervised ranking optimizer steps before TD training.")
    parser.add_argument("--teacher_prefit_batch_sets", type=int, default=16,
                        help="Candidate sets per supervised prefit batch.")
    parser.add_argument("--teacher_prefit_lr", type=float, default=1e-4,
                        help="Learning rate for Linear-VFA teacher prefit.")
    args = parser.parse_args()

    custom_reward = RewardConfig()
    custom_reward.weight_congestion = args.congestion_weight
    custom_reward.weight_fleet_degradation = args.fleet_degradation_weight

    architectures = [
        [64],
        [104, 52],
        [128, 64, 32],
        [256, 128, 64, 32],
    ]
    if args.valuehead_hidden_dims is not None:
        architectures = [args.valuehead_hidden_dims]

    for arch in architectures:
        arch_str   = "-".join(map(str, arch))
        arch_label = f"{args.run_label}_arch_{arch_str}" if args.run_label else f"arch_{arch_str}"

        print("\n" + "="*50)
        print(f"LAUNCHING NEURAL NETWORK TRAINING")
        print("="*50)
        print(f"  Architecture:        {arch}")
        print(f"  Instance:            {args.instance}")
        print(f"  Episodes:            {args.episodes}")
        print(f"  n_step:              {args.n_step}")
        print(f"  buffer_size:         {args.buffer_size}")
        print(f"  polyak:              {args.polyak}")
        print(f"  tau_end:             {args.tau_end}")
        print(f"  maintenance_shaping: {args.maintenance_shaping}")
        print(f"  fleet_degradation:   {args.fleet_degradation_weight}")
        print(f"  reward_norm:         {args.reward_norm}")
        if args.reward_norm == "fixed":
            print(f"  fixed_reward_scale:  {args.fixed_reward_scale:g}")
        print(f"  station_id_embed:    {args.station_id_embedding}"
              f"{f' dim={args.station_id_embed_dim}' if args.station_id_embedding else ''}")
        print(f"  action_context:      {args.action_context}")
        print(f"  station_spotlights:  {args.station_spotlights}")
        print(f"  demand_horizon:      {args.demand_horizon}")
        print(f"  global_health:       {args.global_health}")
        print(f"  candidate_wildcards: {args.candidate_wildcards}")
        print(f"  max_updates/ep:      {args.max_updates_per_episode}")
        print(f"  linear_teacher:      {args.linear_teacher_prefit}")
        print(
            f"  teacher_prefit:      eps={args.teacher_prefit_episodes} "
            f"steps={args.teacher_prefit_steps} "
            f"batch_sets={args.teacher_prefit_batch_sets} "
            f"lr={args.teacher_prefit_lr:g}"
        )
        print(f"  odometer_stats:      {args.odometer_stats}")
        print("="*50 + "\n")

        train_nn_rollout(
            num_episodes             = args.episodes,
            seed_offset              = args.seed,
            instance_name            = args.instance,
            gamma                    = args.gamma,
            lr_start                 = args.lr_start,
            lr_end                   = args.lr_end,
            tau_start                = args.tau_start,
            tau_end                  = args.tau_end,
            tau_anneal_episodes      = args.tau_anneal_episodes,
            target_update_freq       = args.target_update_freq,
            polyak                   = args.polyak,
            batch_size               = args.batch_size,
            buffer_size              = args.buffer_size,
            n_step_return            = args.n_step,
            use_maintenance_shaping  = args.maintenance_shaping,
            save_path                = None,
            depot_id                 = args.depot_id,
            reward_calc_config       = custom_reward,
            run_label                = arch_label,
            value_hidden_dims        = arch,
            enable_logging           = False,
            maintenance_enabled      = args.maintenance,
            odometer_stats_path      = args.odometer_stats,
            odometer_sampling_method = args.odometer_sampling_method,
            odometer_sampling_bounds = args.odometer_sampling_bounds,
            reward_norm_mode         = args.reward_norm,
            fixed_reward_scale       = args.fixed_reward_scale,
            use_station_id_embedding = args.station_id_embedding,
            station_id_embed_dim     = args.station_id_embed_dim,
            use_action_context       = args.action_context,
            use_station_spotlights   = args.station_spotlights,
            use_demand_horizon       = args.demand_horizon,
            use_global_health        = args.global_health,
            candidate_wildcards      = args.candidate_wildcards,
            max_updates_per_episode  = args.max_updates_per_episode,
            linear_teacher_prefit_path = args.linear_teacher_prefit,
            teacher_prefit_episodes  = args.teacher_prefit_episodes,
            teacher_prefit_steps     = args.teacher_prefit_steps,
            teacher_prefit_batch_sets = args.teacher_prefit_batch_sets,
            teacher_prefit_lr        = args.teacher_prefit_lr,
        )
