"""
train_nn_rollout.py  —  Offline Training Loop for the NN Value Network

Trains the NNValueNetwork using episodic simulation with TD(0) updates,
a target network for stability, and an experience replay buffer.

─────────────────────────────────────────────────────────────────────────────
EPISODE STRUCTURE  (mirrors train_vfa.py)
─────────────────────────────────────────────────────────────────────────────

  Days 1 – 2    Warm-up  : GreedyPolicy drives the system.
                           No TD updates — builds up a realistic "messy"
                           state without biasing the NN weights early on.

  Days 3 – 14   Learning : NNLearningPolicy with Boltzmann exploration.
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
from policies.sjovik_sund.NN.nn_model import NNValueNetwork, build_nn_value_network
from policies.sjovik_sund.NN.nn_state_encoder import encode_state
from policies.sjovik_sund.mdp.candidate_generator import generate_candidates
#from policies.sjovik_sund.mdp.candidate_generator_nn import generate_candidates
from policies.sjovik_sund.mdp.mdp_formulation import (
    extract_mdp_state,
    PostDecisionState,
    MDPState,
)
from policies.sjovik_sund.mdp.mdp_config import MDPConfig
from policies.sjovik_sund.mdp.reward import RewardCalculator
from policies.sjovik_sund.run_simulation_ingvild import run_simulation, SimulationConfig
from settings import ENABLE_COMPONENT_FAILURES

# --- Device Selection ---
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

NUM_EPISODES         : int   = 800
EPISODE_DAYS         : int   = 14
WARMUP_DAYS          : int   = 2     # GreedyPolicy for days 1–2
LEARNING_DAYS        : int   = 12    # NNLearningPolicy for days 3–14

LR_START             : float = 5e-4  # Adam lr at episode 0
LR_END               : float = 5e-5  # Adam lr at episode N (linearly decayed)

GAMMA                : float = 0.99  # discount factor (matches linear VFA)

TARGET_UPDATE_FREQ   : int   = 10    # kept for CLI arg compatibility; not used when POLYAK > 0
POLYAK               : float = 0.0005 # soft target update rate: θ_target ← (1-τ)θ_target + τθ_online
                                     # applied every episode instead of hard copy every N episodes.
                                     # Set to 0.0 to fall back to hard copies (original behaviour).

N_STEP_RETURN        : int   = 3    # n-step TD return length.
                                     # 1 = standard TD(0).
                                     # Try n=3 now that normalize() scales by the n-step discount sum.
                                     # Previous attempt hurt greedy_sl (0.947→0.940) due to inflated
                                     # targets from dividing by the 1-step max regardless of n.

BUFFER_SIZE          : int   = 50_000  # max transitions in replay buffer
BATCH_SIZE           : int   = 128     # mini-batch size per gradient update
MIN_BUFFER_SIZE      : int   = 256     # start learning only after this many transitions

GRAD_CLIP_NORM       : float = 0.5   # max gradient norm (prevents large TD spikes)
REWARD_NORM_EPS      : float = 1e-8  # avoid div-by-zero in reward normalizer

# Boltzmann exploration temperature schedule (linear anneal over all episodes)
# tau_start: high temperature early → broad exploration of action space
# tau_end:   near-zero  late        → essentially greedy exploitation
TAU_START : float = 0.05    # Decreased from 0.5
TAU_END   : float = 0.001   # Decreased from 0.02

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
    than the all-time average. The Welford approach froze after ~5 episodes
    because the infinite window made new samples negligible — by ep 800 the
    statistics still reflected episode-1 behavior regardless of policy improvement.

    alpha controls the adaptation speed (half-life ≈ ln(2)/alpha samples):
      alpha=0.005 → half-life ≈ 139 samples ≈ ~0.15 episodes  (fast adapt)
      alpha=0.001 → half-life ≈ 693 samples ≈ ~0.7 episodes   (moderate)

    The first update seeds mean=reward and var=1 to avoid a cold-start at 0.
    """

    def __init__(self, alpha: float = 0.005):
        self._alpha = alpha
        self._mean  = 0.0
        self._var   = 1.0    # start with unit variance (no scaling until first update)
        self._n     = 0      # counts updates; used only for the CSV log field

    def update(self, reward: float) -> None:
        self._n += 1
        if self._n == 1:
            self._mean = reward   # cold-start: seed mean at first reward
            return
        self._mean = (1.0 - self._alpha) * self._mean + self._alpha * reward
        self._var  = (1.0 - self._alpha) * self._var  + self._alpha * (reward - self._mean) ** 2

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        return math.sqrt(max(self._var, REWARD_NORM_EPS))

    def normalize(self, reward: float, n_steps: int = 1, gamma: float = 0.99) -> float:
        # Fixed-range normalization scaled to the n-step return window.
        # 1-step max penalty is 76.7. For n steps the worst-case accumulated
        # return is 76.7 × Σ_{i=0}^{n-1} γ^i, so we divide by that sum to
        # keep TD targets in [-1, 0] regardless of n.
        # n=1: divides by 76.7 (same as before).
        # n=3: divides by 76.7 × (1 + γ + γ²) ≈ 227.8 — prevents inflated targets.
        discount_sum = sum(gamma ** i for i in range(n_steps))
        return reward / (76.7 * discount_sum)


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
        v_list  = v_curs.detach().cpu().squeeze(1).tolist()
        tgt_list = td_targets.cpu().squeeze(1).tolist()
        debug_info = {
            "v_cur_mean":     statistics.mean(v_list),
            "v_cur_std":      statistics.stdev(v_list) if len(v_list) > 1 else 0.0,
            "v_cur_min":      min(v_list),
            "v_cur_max":      max(v_list),
            "td_target_mean": statistics.mean(tgt_list),
            "td_target_std":  statistics.stdev(tgt_list) if len(tgt_list) > 1 else 0.0,
            "reward_mean":    statistics.mean(rewards),
            "reward_min":     min(rewards),
            "reward_max":     max(rewards),
            "n_done":         sum(dones),
        }

    return loss, debug_info


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
        online_model:      NNValueNetwork,
        reward_calc_config,               # reward.RewardConfig — passed from the training loop
        replay_buffer:     ReplayBuffer,
        gamma:             float,
        config:            MDPConfig,
        depot_id:          Optional[str],
        tau:               float = 0.0,   # Boltzmann temperature; 0 = greedy
        verbose:           bool = False,
        reward_normalizer: Optional[RewardNormalizer] = None,
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

        # Reward normalizer — shared across episodes, passed in from training loop.
        self.reward_normalizer = reward_normalizer

        # Candidate index tracking: which pool slot (0=nearest, last=most critical) did NN pick?
        # Persistent clustering at 0 → NN collapsed to "always go nearest".
        # Distributed picks → NN is making state-dependent choices.
        self._chosen_indices: list = []

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

    def _encode_post_decision(self, mdp_state: MDPState, mdp_action) -> dict:
        """
        Compute and encode the post-decision state S^x for a given MdpAction.

        PostDecisionState.apply() is a pure function — it returns a NEW
        MDPState without modifying the live simulator or the original mdp_state.

        Falls back to encoding the current pre-decision state if apply() raises
        a validation error (e.g., action is infeasible due to state rounding).
        """
        self._candidate_count += 1
        try:
            post_state, _, _ = PostDecisionState.apply(mdp_state, mdp_action)
            return encode_state(post_state)
        except Exception as _exc:
            # Fallback: encode current state; slightly less accurate but safe.
            # Tally the reason so we can diagnose which validation check fires.
            self._fallback_count += 1
            reason = str(_exc)[:80]   # first 80 chars is enough to identify the check
            self._fallback_reasons[reason] = self._fallback_reasons.get(reason, 0) + 1
            return encode_state(mdp_state)

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
        r_k = self._reward_calc.compute_step_reward(state.metrics)

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
            wide_search=True,
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
                try:
                    post_state, _, _ = PostDecisionState.apply(mdp_state, mdp_action)
                    post_encoded = encode_state(post_state)
                    
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
            if self.verbose:
                print("  [WARNING] ZERO valid actions this epoch! Forcing DoNothing.")
            
        # --- Debug: state representation and candidate scores ---
        if self.verbose:
            # make it print more regularly than just episode 0 — every 2nd decision epoch (since some episodes have very few decisions)
            if self._decision_count % 2 == 0: 
                # Full state encoding dump on the very first learning decision
                #enc = post_encodings[0] if post_encodings else self._encode_post_decision(mdp_state, pairs[0][0])
                enc = post_encodings[0] if post_encodings else self._encode_post_decision(mdp_state, valid_pairs[0][0])
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
            v_min, v_max = min(values), max(values)
            print(
                f"  [DEBUG] decision {self._decision_count:4d} | "
                f"t={mdp_state.time:.0f}min | "
                f"{len(values)} candidates | "
                f"V min={v_min:.4f} max={v_max:.4f} | "
                f"r_k={r_k:.4f}"
            )

        # Accumulate per-decision spread for episode-level diagnostics.
        # A spread near 0 means all candidates look equally good to the NN
        # → Boltzmann selection degenerates to uniform random.
        if len(values) > 1:
            self._value_spreads.append(max(values) - min(values))

        self._decision_count += 1

        # --- Step 5: Action selection (Boltzmann if tau > 0, else greedy) ---
        idx = _boltzmann_select(values, self.tau)
        chosen_sim_action    = pairs[idx][1]
        chosen_post_encoded  = post_encodings[idx]

        if self.verbose:
            print(f"             chosen candidate idx={idx} | V={values[idx]:.4f}")

        # Track which pool slot was chosen (0=nearest station, last=most critical).
        self._chosen_indices.append(idx)

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
            if len(self._nstep_pending) >= N_STEP_RETURN:
                accum_r = sum(
                    self.gamma ** i * self._nstep_pending[i][1]
                    for i in range(N_STEP_RETURN)
                )
                enc_0 = self._nstep_pending[0][0]
                self.replay_buffer.push(
                    encoded_cur=enc_0,
                    reward=accum_r,
                    encoded_next=chosen_post_encoded,
                    n_steps=N_STEP_RETURN,
                    done=False,
                )
                self._nstep_pushed += 1
                self._nstep_pending.popleft()

        # --- Step 7: store chosen post-decision state for next epoch ---
        self._prev_post_encoded = chosen_post_encoded
        self._prev_time         = mdp_state.time

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
            final_r = self._reward_calc.compute_step_reward(sim_state.metrics)

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
                f"  [N-STEP] n={N_STEP_RETURN} | "
                f"full pushes={self._nstep_pushed} | "
                f"terminal flush={self._nstep_terminal} | "
                f"total={self._nstep_pushed + self._nstep_terminal}"
            )

        # Reset for the next episode
        self._prev_post_encoded = None
        self._prev_time         = None
        self._reward_calc       = None
        self._nstep_pending.clear()
        self._nstep_pushed   = 0
        self._nstep_terminal = 0


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
    online_model:    NNValueNetwork,
    reward_calc_config,
    config:          MDPConfig,
    depot_id:        Optional[str],
    seed:            int,
    instance_name:   str,
    gamma:           float,
    warmup_end_time: float,
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
    )
    greedy_policy  = GreedyPolicy()
    eval_episode   = NNEpisodeTrainingPolicy(
        nn_learning_policy=eval_learning,
        greedy_policy=greedy_policy,
        warmup_end_time=warmup_end_time,
    )
    sim_config = SimulationConfig()
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
    num_episodes:       int   = NUM_EPISODES,
    save_path:          Path  = None,
    seed_offset:        int   = 0,
    instance_name:      str   = INSTANCE_NAME,
    gamma:              float = GAMMA,
    lr_start:           float = LR_START,
    lr_end:             float = LR_END,
    tau_start:          float = TAU_START,
    tau_end:            float = TAU_END,
    target_update_freq: int   = TARGET_UPDATE_FREQ,
    polyak:             float = POLYAK,
    batch_size:         int   = BATCH_SIZE,
    buffer_size:        int   = BUFFER_SIZE,
    depot_id:           Optional[str] = None,
    reward_calc_config=None,  # RewardConfig used to build the step-reward calculator
    run_label:          Optional[str] = None,  # extra tag injected into all output filenames
) -> NNValueNetwork:
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

    # ── Build online model + frozen target ────────────────────────────────────
    # The target model starts as an exact copy of the online model.
    # It is NEVER updated by backprop; only by hard weight copies every
    # TARGET_UPDATE_FREQ episodes.
    online_model = build_nn_value_network().to(device)
    target_model = copy.deepcopy(online_model).to(device)
    for param in target_model.parameters():
        param.requires_grad = False   # no gradient tracking in target

    # ── Optimizer and shared replay buffer ───────────────────────────────────
    # The replay buffer is shared across ALL episodes: transitions from earlier
    # episodes (when the policy was exploratory) stay in the buffer and continue
    # to contribute to gradient updates in later episodes.
    optimizer         = optim.Adam(online_model.parameters(), lr=lr_start, weight_decay=1e-4)
    replay_buffer     = ReplayBuffer(max_size=buffer_size)
    reward_normalizer = RewardNormalizer()   # shared across all episodes

    # ── Warmup end time (absolute simulation minutes) ─────────────────────────
    # Mirrors the calculation in train_vfa.py:
    #   sim_start_min = 5h × 60 = 300 min
    #   warmup_end_time = 300 + 2 × 1440 = 3180 min (end of day 2)
    sim_start_min   = timeInMinutes(hours=START_HOUR)
    warmup_end_time = sim_start_min + WARMUP_DAYS * 24 * 60

    config = MDPConfig.full_maintenance() if ENABLE_COMPONENT_FAILURES else MDPConfig.no_maintenance()

    # ── Reward calculator config ──────────────────────────────────────────────
    # NNLearningPolicy needs a RewardConfig to build its step-reward calculator.
    # If none is provided, borrow one from a throw-away LinearVFAPolicy instance
    # (its weights are never used; only the reward config matters).
    if reward_calc_config is None:
        from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
        _tmp_vfa = LinearVFAPolicy(
            learning_mode=False,
            maintenance_enabled=ENABLE_COMPONENT_FAILURES,
        )
        reward_calc_config = _tmp_vfa.reward_calc.config

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
            config=SimulationConfig(),
        )
        _probe_vehicles = _probe.state.get_vehicles()
        depot_id = _probe.state.get_closest_depot(_probe_vehicles[0]) if _probe_vehicles else None
        print(f"  Depot ID          : {depot_id!r}  (auto-resolved from instance)")

    # ── Metrics tracking ──────────────────────────────────────────────────────
    learning_curve = []   # written to checkpoint; also mirrored to CSV below
    best_greedy_sl  = -float("inf")   # track best greedy SL for model saving
    best_greedy_path: Path = None
    t0 = time.time()

    # Open CSV log — one row per episode, written incrementally so a partial
    # run is still readable if training is interrupted on the cluster.
    ts_run    = datetime.now().strftime("%Y%m%d_%H%M%S")
    _label    = f"_{run_label}" if run_label else f"_tau{tau_start}_freq{target_update_freq}_lr{lr_start}_poly{polyak}"
    csv_path  = SAVE_DIR / f"training_log_seed{seed_offset}{_label}_{ts_run}.csv"
    CSV_FIELDS = [
        "episode", "mean_loss", "lr", "tau", "polyak",
        "service_level", "buffer_size", "n_updates", "elapsed_s",
        "mean_value_spread",  # max(V)-min(V) per decision; near 0 = NN not discriminating
        "mean_reward",        # raw step reward mean (un-normalized); scale check
        "pct_zero_reward",    # % decisions with r=0; high = too sparse
        "fallback_rate",      # % PostDecisionState.apply() failures; > 5% = data quality issue
        "reward_norm_std",    # running std of all rewards seen so far; tracks normalization scale
        "pct_idx0",           # % decisions where NN picked pool slot 0 (nearest); ~100% = collapsed
        "mean_chosen_idx",    # mean pool index chosen; 0=always nearest, ~3.5=uniform
        "greedy_sl",          # tau=0 eval SL — true NN quality, unconfounded by exploration
    ]
    csv_file   = csv_path.open("w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    csv_writer.writeheader()
    print(f"  CSV log           : {csv_path}")

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
        current_tau = tau_start + (tau_end  - tau_start) * frac
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
        )
        greedy_policy  = GreedyPolicy()
        episode_policy = NNEpisodeTrainingPolicy(
            nn_learning_policy=nn_learning,
            greedy_policy=greedy_policy,
            warmup_end_time=warmup_end_time,
        )

        print(f"\n{'='*50}")
        print(f"EPISODE {ep+1}/{num_episodes} | lr={current_lr:.5f}")
        print(f"{'='*50}")

        # --- Run the episode ---
        sim_config = SimulationConfig()
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
        episode_losses = []
        n_updates      = 0   # tracked for CSV logging

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
            '''n_updates = min(
                max(1, len(replay_buffer) // batch_size) * 4,
                150,  # hard cap: at most 400 gradient steps per episode
            )'''
            # Change the cap from 150 to 400 to squeeze more learning out of the data
            n_updates = min(max(1, len(replay_buffer) // batch_size) * 4, 400)

            if ep == VERBOSE_EPISODE:
                print(f"  [DEBUG] gradient update: {n_updates} steps x batch={batch_size}")

            for update_i in range(n_updates):
                batch = replay_buffer.sample(batch_size)

                # Collect debug info on the final update step of a diagnostic episode.
                want_debug = collect_debug_this_ep and (update_i == n_updates - 1)

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
                # Gradient clipping prevents a rare large TD spike from
                # destabilizing the weights — standard practice in neural TD.
                nn.utils.clip_grad_norm_(online_model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                episode_losses.append(loss.item())

                # Polyak update every gradient step (standard SAC/TD3 schedule).
                # Per-step τ=0.005 keeps the target responsive without losing
                # the stabilising lag: after 400 steps the target is ~87% online.
                # Previously this ran once per episode (τ per episode), leaving
                # the target 80% at random initialisation after 43 episodes.
                # Polyak update every gradient step
                if polyak > 0.0:
                    with torch.no_grad():
                        for _op, _tp in zip(online_model.parameters(), target_model.parameters()):
                            _tp.data.mul_(1.0 - polyak).add_(polyak * _op.data)
                            
        mean_loss = sum(episode_losses) / max(len(episode_losses), 1)

        # --- Target network update ---
        # Polyak updates now run inside the gradient loop (per gradient step).
        # This block handles the fallback hard-copy when POLYAK=0, and prints
        # a periodic diagnostic showing how far online and target have drifted.
        if POLYAK > 0.0:
            if ep == 0 or (DEBUG_EVERY > 0 and (ep + 1) % DEBUG_EVERY == 0):
                sample_p = next(online_model.parameters())
                sample_t = next(target_model.parameters())
                diff = (sample_p - sample_t).norm().item()
                print(f"  [POLYAK] ep={ep+1}  τ={POLYAK}/step  |online-target|={diff:.5f}")
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
            # 3c. Reward normalizer stats
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
                seed=seed_offset + num_episodes + ep,   # unseen seed
                instance_name=instance_name,
                gamma=gamma,
                warmup_end_time=warmup_end_time,
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
                    "best_greedy_sl":       best_greedy_sl,
                }, best_greedy_path)
                print(f"  -> New best greedy SL={best_greedy_sl:.4f} — saved to {best_greedy_path.name}")

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

        # Write one CSV row per episode (flushed immediately so partial runs are readable)
        csv_writer.writerow({
            "episode":          ep + 1,
            "mean_loss":        round(mean_loss,    6),
            "lr":               round(current_lr,   6),
            "tau":              round(current_tau,  4),
            "polyak":           polyak,
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
            "greedy_sl":         round(eval_sl, 4) if eval_sl is not None else "",
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
            }, ck_path)
            print(f"  -> Checkpoint saved -> {ck_path}")

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

def load_nn_model(checkpoint_path: str) -> NNValueNetwork:
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

    model = NNValueNetwork(
        station_feature_dim=checkpoint["station_feature_dim"],
        vehicle_feature_dim=checkpoint["vehicle_feature_dim"],
        global_feature_dim=checkpoint["global_feature_dim"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Loaded NN model from {checkpoint_path} "
          f"(episode {checkpoint.get('episode', '?')})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Offline episodic TD(0) training of a neural network value function",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes",           type=int,   default=NUM_EPISODES)
    parser.add_argument("--seed",               type=int,   default=0)
    parser.add_argument("--instance",           type=str,   default=INSTANCE_NAME)
    parser.add_argument("--gamma",              type=float, default=GAMMA)
    parser.add_argument("--lr_start",           type=float, default=LR_START)
    parser.add_argument("--lr_end",             type=float, default=LR_END)
    parser.add_argument("--tau_start",          type=float, default=TAU_START)
    parser.add_argument("--tau_end",            type=float, default=TAU_END)
    parser.add_argument("--target_update_freq", type=int,   default=TARGET_UPDATE_FREQ)
    parser.add_argument("--polyak", type=float, default=POLYAK, help="Soft target update rate (tau)")
    parser.add_argument("--batch_size",         type=int,   default=BATCH_SIZE)
    parser.add_argument("--buffer_size",        type=int,   default=BUFFER_SIZE)
    parser.add_argument("--run_label",           type=str,   default=None,
                        help="Tag injected into output filenames (auto-generated from tau/freq if omitted).")
    parser.add_argument("--save",               type=str,   default=None)
    parser.add_argument("--depot_id",           type=str,   default='D0',
                        help="Depot station ID (e.g. 'D0'). Auto-resolved from instance if omitted.")
    args = parser.parse_args()

    train_nn_rollout(
        num_episodes        = args.episodes,
        seed_offset         = args.seed,
        instance_name       = args.instance,
        gamma               = args.gamma,
        lr_start            = args.lr_start,
        lr_end              = args.lr_end,
        tau_start           = args.tau_start,
        tau_end             = args.tau_end,
        target_update_freq  = args.target_update_freq,
        polyak              = args.polyak,
        batch_size          = args.batch_size,
        buffer_size         = args.buffer_size,
        save_path           = Path(args.save) if args.save else None,
        depot_id            = args.depot_id,
        run_label           = args.run_label,
    )
