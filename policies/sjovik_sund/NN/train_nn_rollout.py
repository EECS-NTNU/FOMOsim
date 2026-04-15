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

NUM_EPISODES         : int   = 300
EPISODE_DAYS         : int   = 14
WARMUP_DAYS          : int   = 2     # GreedyPolicy for days 1–2
LEARNING_DAYS        : int   = 12    # NNLearningPolicy for days 3–14

LR_START             : float = 1e-3  # Adam lr at episode 0
LR_END               : float = 1e-4  # Adam lr at episode N (linearly decayed)

GAMMA                : float = 0.99  # discount factor (matches linear VFA)

TARGET_UPDATE_FREQ   : int   = 10    # copy online → target every N episodes

BUFFER_SIZE          : int   = 10_000  # max transitions in replay buffer
BATCH_SIZE           : int   = 64      # mini-batch size per gradient update
MIN_BUFFER_SIZE      : int   = 256     # start learning only after this many transitions

GRAD_CLIP_NORM       : float = 1.0   # max gradient norm (prevents large TD spikes)

# Boltzmann exploration temperature schedule (linear anneal over all episodes)
# tau_start: high temperature early → broad exploration of action space
# tau_end:   near-zero  late        → essentially greedy exploitation
TAU_START            : float = 0.5
TAU_END              : float = 0.02

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


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIENCE REPLAY BUFFER
# ═════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Fixed-size ring buffer storing TD transitions.

    Each entry is a tuple:
        (encoded_cur, reward, encoded_next, done)

    where encoded_* are dicts {"station_block", "vehicle_block", "global_context"}
    produced by encode_state() on POST-DECISION states S^x.

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

    def push(self, encoded_cur: dict, reward: float, encoded_next: dict, done: bool):
        """Add one (S^x_cur, r, S^x_next, done) transition."""
        self._buffer.append((encoded_cur, reward, encoded_next, done))

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

def _compute_td_loss(online_model, target_model, batch, gamma, n_stations):
    losses = []
    for (enc_cur, reward, enc_next, done) in batch:
        # Move input blocks to M1 GPU
        v_cur = online_model(
            enc_cur["station_block"].to(device),
            enc_cur["vehicle_block"].to(device),
            enc_cur["global_context"].to(device),
        )

        with torch.no_grad():
            # Reward and target must also be on the same device
            normalized_r = torch.tensor([reward / max(n_stations, 1)], 
                                        dtype=torch.float32, device=device)
            if done:
                td_target = normalized_r
            else:
                # Move next-state blocks to M1 GPU
                v_next = target_model(
                    enc_next["station_block"].to(device),
                    enc_next["vehicle_block"].to(device),
                    enc_next["global_context"].to(device),
                )
                td_target = normalized_r + gamma * v_next

        losses.append((td_target - v_cur) ** 2)
    return torch.stack(losses).mean()


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

        # Decision counter — used for verbose debug output.
        self._decision_count: int = 0

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
        try:
            post_state, _, _ = PostDecisionState.apply(mdp_state, mdp_action)
            return encode_state(post_state)
        except Exception:
            # Fallback: encode current state; slightly less accurate but safe
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
        mdp_state = extract_mdp_state(
            sim_state=state,
            active_vehicle_id=vehicle.id,
            config=self.config,
            depot_id=self.depot_id,
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
        )

        # --- Step 4: score each candidate's post-decision state ---
        values         = []
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
                post_encodings.append(post_encoded)

        # --- Debug: state representation and candidate scores ---
        if self.verbose:
            if self._decision_count == 0:
                # Full state encoding dump on the very first learning decision
                enc = post_encodings[0] if post_encodings else self._encode_post_decision(mdp_state, pairs[0][0])
                sb = enc["station_block"]   # [N, 4]
                vb = enc["vehicle_block"]   # [M, 5]
                gc = enc["global_context"]  # [7]
                print("\n" + "-" * 60)
                print("  [DEBUG] First learning-phase decision - state encoding")
                print("-" * 60)
                print(f"  station_block   shape : {list(sb.shape)}  (N_stations x 4)")
                print(f"  vehicle_block   shape : {list(vb.shape)}  (M_vehicles x 5)")
                print(f"  global_context  shape : {list(gc.shape)}  (7 features)")
                print()
                print("  station_block  [func | onsite | depot | free_docks]")
                for i, row in enumerate(sb.tolist()):
                    sid = sorted(mdp_state.stations.keys())[i]
                    print(f"    station {sid:>4s}: {['%.3f'%x for x in row]}")
                print()
                print("  vehicle_block  [func_cargo | depot_cargo | free_cap | dest_func | eta]")
                for i, row in enumerate(vb.tolist()):
                    vid = sorted(mdp_state.vehicles.keys())[i]
                    print(f"    vehicle {vid:>4s}: {['%.3f'%x for x in row]}")
                print()
                gc_labels = ["time_sin", "time_cos", "starvation", "broken_ratio",
                             "depot_queue", "shift_remain", "mean_load"]
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

        self._decision_count += 1

        # --- Step 5: Action selection (Boltzmann if tau > 0, else greedy) ---
        idx = _boltzmann_select(values, self.tau)
        chosen_sim_action    = pairs[idx][1]
        chosen_post_encoded  = post_encodings[idx]

        if self.verbose:
            print(f"             chosen candidate idx={idx} | V={values[idx]:.4f}")

        # --- Step 6: push transition to replay buffer ---
        # We push (S^x_{k-1}, r_k, S^x_k) where:
        #   S^x_{k-1} = post-decision state from the PREVIOUS decision epoch
        #   r_k       = reward observed in the interval [t_{k-1}, t_k]
        #   S^x_k     = post-decision state of the CHOSEN action at t_k
        #
        # On the very first learning-phase call, _prev_post_encoded is None
        # (no previous learning-phase decision), so we skip this push.
        if self._prev_post_encoded is not None:
            self.replay_buffer.push(
                encoded_cur=self._prev_post_encoded,
                reward=r_k,
                encoded_next=chosen_post_encoded,
                done=False,
            )

        # --- Step 7: store chosen post-decision state for next epoch ---
        self._prev_post_encoded = chosen_post_encoded

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
            # For the terminal transition, S^x_next is a copy of S^x_cur
            # (the bootstrapped value will be multiplied by 0 since done=True
            # means _compute_td_loss uses only r, not r + γ V_next).
            self.replay_buffer.push(
                encoded_cur=self._prev_post_encoded,
                reward=final_r,
                encoded_next=self._prev_post_encoded,  # not used; done=True
                done=True,
            )
        # Reset for the next episode
        self._prev_post_encoded = None
        self._reward_calc       = None


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
    batch_size:         int   = BATCH_SIZE,
    buffer_size:        int   = BUFFER_SIZE,
    depot_id:           Optional[str] = None,
    reward_calc_config=None,  # RewardConfig used to build the step-reward calculator
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
    optimizer     = optim.Adam(online_model.parameters(), lr=lr_start)
    replay_buffer = ReplayBuffer(max_size=buffer_size)

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

    # ── Metrics tracking ──────────────────────────────────────────────────────
    learning_curve = []   # written to checkpoint; also mirrored to CSV below
    t0 = time.time()

    # Open CSV log — one row per episode, written incrementally so a partial
    # run is still readable if training is interrupted on the cluster.
    ts_run   = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = SAVE_DIR / f"training_log_seed{seed_offset}_{ts_run}.csv"
    CSV_FIELDS = ["episode", "mean_loss", "lr", "tau",
                  "service_level", "buffer_size", "n_updates", "elapsed_s"]
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

        if buffer_ready:
            # Number of gradient steps: proportional to episode length relative to batch size,
            # capped to prevent overfitting on a single episode's transitions.
            n_updates = min(
                max(1, len(replay_buffer) // batch_size),
                50,   # hard cap: at most 50 gradient steps per episode
            )

            # Infer n_stations from the first sample for reward normalization
            sample_batch = replay_buffer.sample(min(batch_size, len(replay_buffer)))
            n_stations   = len(sample_batch[0][0]["station_block"])

            if ep == VERBOSE_EPISODE:
                print(f"  [DEBUG] gradient update: {n_updates} steps x batch={batch_size} | n_stations={n_stations}")

            for _ in range(n_updates):
                batch = replay_buffer.sample(batch_size)

                # Forward: online model; backward: clip gradients; step
                loss = _compute_td_loss(
                    online_model=online_model,
                    target_model=target_model,
                    batch=batch,
                    gamma=gamma,
                    n_stations=n_stations,
                )

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping prevents a rare large TD spike from
                # destabilizing the weights — standard practice in neural TD.
                nn.utils.clip_grad_norm_(online_model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                episode_losses.append(loss.item())

        mean_loss = sum(episode_losses) / max(len(episode_losses), 1)

        # --- Target network update ---
        # Hard copy online → target every TARGET_UPDATE_FREQ episodes.
        # Between updates the target is frozen, giving a stable TD bootstrap
        # reference. Copying too often → instability; too rarely → stale target.
        if (ep + 1) % target_update_freq == 0:
            target_model.load_state_dict(online_model.state_dict())
            print(f"  -> Target network updated (episode {ep+1})")

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
            f"loss={mean_loss:.4f} | SL={sl:.4f} | buffer={len(replay_buffer):,} | "
            f"t={elapsed:.0f}s"
        )

        # Write one CSV row per episode (flushed immediately so partial runs are readable)
        csv_writer.writerow({
            "episode":       ep + 1,
            "mean_loss":     round(mean_loss,   6),
            "lr":            round(current_lr,  6),
            "tau":           round(current_tau, 4),
            "service_level": round(sl,          4),
            "buffer_size":   len(replay_buffer),
            "n_updates":     n_updates,
            "elapsed_s":     round(elapsed,     1),
        })
        csv_file.flush()

        # --- Periodic checkpoint ---
        if (ep + 1) % 50 == 0:
            ck_path = SAVE_DIR / f"nn_model_ep{ep+1:04d}_seed{seed_offset}.pt"
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
        save_path = SAVE_DIR / f"nn_model_final_seed{seed_offset}_{ts}.pt"

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
    print(f"  Model saved   : {save_path}")
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
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--instance", type=str, default=INSTANCE_NAME)
    parser.add_argument("--gamma",    type=float, default=GAMMA)
    parser.add_argument("--lr_start",  type=float, default=LR_START)
    parser.add_argument("--lr_end",    type=float, default=LR_END)
    parser.add_argument("--save",     type=str, default=None)
    args = parser.parse_args()

    train_nn_rollout(
        num_episodes  = args.episodes,
        seed_offset   = args.seed,
        instance_name = args.instance,
        gamma         = args.gamma,
        lr_start      = args.lr_start,
        lr_end        = args.lr_end,
        save_path     = Path(args.save) if args.save else None,
    )
