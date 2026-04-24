# Neural Network Value Function — Full Technical Documentation

**System:** FOMOsim bike-sharing rebalancing  
**Branch:** `sjovik_vfa`  
**Last updated:** 2026-04-21  

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture and Code Structure](#2-architecture-and-code-structure)
3. [Data Flow: From Simulator State to NN Prediction](#3-data-flow-from-simulator-state-to-nn-prediction)
4. [Neural Network Architecture](#4-neural-network-architecture)
5. [The NN's Role in the Rollout](#5-the-nns-role-in-the-rollout)
6. [Training Pipeline](#6-training-pipeline)
7. [Loss Function, Optimizer, and Regularization](#7-loss-function-optimizer-and-regularization)
8. [Hyperparameters: Full Reference](#8-hyperparameters-full-reference)
9. [Experiments and Tuning History](#9-experiments-and-tuning-history)
10. [Diagnostics, Metrics, and Training Curve Interpretation](#10-diagnostics-metrics-and-training-curve-interpretation)
11. [Known Issues and Potential Improvements](#11-known-issues-and-potential-improvements)

---

## 1. System Overview

The system solves an online **bike-sharing rebalancing MDP**: a service vehicle visits stations, picks up or drops off bikes, and attempts to minimize starvations (empty stations) and congestions (full stations) over a planning horizon.

The neural network replaces (or augments) a **linear value function approximation (VFA)** as the terminal value estimator inside a **rollout policy**. Both approaches estimate the same quantity:

> **V(S^x)**: the expected future cost (penalty) from a post-decision state S^x.

The key difference:
- **Linear VFA**: `V(S^x) = θᵀ φ(S^x)` — hand-crafted 28-dim feature vector, linear weights.
- **NN**: `V(S^x) = NNValueNetwork(encode_state(S^x))` — raw normalized tensor inputs, learned non-linear representation via Deep Sets.

---

## 2. Architecture and Code Structure

### 2.1 File Map

```
policies/sjovik_sund/NN/
├── nn_model.py              # Neural network definition (Deep Sets)
├── nn_state_encoder.py      # MDPState → tensor encoder (data boundary)
├── train_nn_rollout.py      # Full TD(0) training loop
├── NNRolloutPolicy.py       # Inference: rollout + NN terminal value
├── NNGreedyPolicy.py        # Inference: greedy NN scoring, no rollout
└── evaluate_nn_rollout.py   # Batch/single model evaluation harness
```

### 2.2 Component Responsibilities

| File | Role | Depends On |
|------|------|------------|
| `nn_model.py` | Defines `NNValueNetwork`, sub-encoders, pooling, `build_nn_value_network()` | PyTorch only |
| `nn_state_encoder.py` | Converts `MDPState` → `{station_block, vehicle_block, global_context}` | `MDPState`, `settings` |
| `train_nn_rollout.py` | Training loop: episodes, replay buffer, TD loss, optimizer, checkpointing | All above + simulator + `candidate_generator` |
| `NNRolloutPolicy.py` | Rollout inference: clones simulator, runs scenarios, calls NN for terminal value | `nn_model`, `nn_state_encoder`, simulator |
| `NNGreedyPolicy.py` | Lightweight greedy policy: scores candidates with NN directly, no cloning | `nn_model`, `nn_state_encoder` |
| `evaluate_nn_rollout.py` | Loads `.pt` checkpoints, runs `test_policies` vs. baselines | `NNRolloutPolicy`, `NNGreedyPolicy`, `LinearVFAPolicy` |

### 2.3 Class Hierarchy

```
Policy (base)
├── NNRolloutPolicy          — production rollout policy (inference)
│   └── uses NNGreedyPolicy  — as base policy for cloned vehicles
├── NNGreedyPolicy           — cheap greedy scoring, no simulator cloning
└── NNEpisodeTrainingPolicy  — training wrapper (warmup + learning routing)
    ├── GreedyPolicy         — warmup phase (days 1–2)
    └── NNLearningPolicy     — learning phase (days 3–14)
        └── ReplayBuffer     — stores TD transitions
```

---

## 3. Data Flow: From Simulator State to NN Prediction

### 3.1 The Encoding Pipeline

Every time the NN needs to evaluate a state, `encode_state(mdp_state)` is called:

```
sim.State  (live simulator)
    │
    └─► extract_mdp_state()          [mdp_formulation.py]
            │  Snapshot: stations, vehicles, time, depot, travel_times
            ▼
        MDPState
            │
            ├─► encode_station_block()   →  [N × 8]  float32 tensor
            ├─► encode_vehicle_block()   →  [M × 6]  float32 tensor
            └─► encode_global_context()  →  [8]      float32 tensor
                    │
                    ▼
            {"station_block", "vehicle_block", "global_context"}
                    │
                    ▼
            NNValueNetwork.forward(...)
                    │
                    ▼
              scalar V(S^x) ∈ ℝ
```

### 3.2 Station Feature Vector — [N × 8]

Each of the N stations is encoded into an 8-element row. All values are normalized by station capacity so they live in [0, 1] regardless of station size:

| Index | Feature | Formula | Meaning |
|-------|---------|---------|---------|
| 0 | `functional_ratio` | `functional / cap` | Fraction of capacity with rentable bikes |
| 1 | `onsite_ratio` | `onsite / cap` | Bikes repairable at station |
| 2 | `depot_ratio` | `depot / cap` | Bikes requiring depot removal |
| 3 | `eta_from_vehicle` | `min(1.0, travel_time / 60)` | Normalized travel time from current vehicle position to this station (dense for ALL stations) |
| 4 | `target_ratio` | `target / cap` | Time-of-day optimal fill level |
| 5 | `deficit_ratio` | `(target - functional) / cap` | Signed imbalance: >0=starving, <0=congested |
| 6 | `departure_rate` | `min(2.0, expected_departures / cap)` | Expected outflow as fraction of capacity per hour |
| 7 | `arrival_rate` | `min(2.0, expected_arrivals / cap)` | Expected inflow as fraction of capacity per hour |

**Intentional omissions:**
- `empty_dock_ratio`: linearly dependent on the other three inventory ratios — always derivable, wastes a dimension.
- `time_sin/cos` in station block: identical for all stations at a given epoch → zero per-station information. Time is captured in the global context instead.

### 3.3 Vehicle Feature Vector — [M × 6]

| Index | Feature | Meaning |
|-------|---------|---------|
| 0 | `functional_cargo_ratio` | Rentable bikes on vehicle / capacity |
| 1 | `depot_cargo_ratio` | Depot-bound bikes on vehicle / capacity |
| 2 | `dest_functional_ratio` | Destination station's current fill ratio |
| 3 | `dest_target_ratio` | Destination station's target fill ratio |
| 4 | `dest_deficit_ratio` | Destination station's signed imbalance |
| 5 | `eta_normalized` | `min(1.0, time_until_arrival / 60)` — normalized ETA |

### 3.4 Global Context Vector — [8]

| Index | Feature | Meaning |
|-------|---------|---------|
| 0 | `time_sin` | `sin(2π · hour / 24)` — cyclic daily time |
| 1 | `time_cos` | `cos(2π · hour / 24)` — cyclic daily time |
| 2 | `starved_ratio` | Fraction of stations with 0 functional bikes |
| 3 | `low_ratio` | Fraction of stations with <10% functional |
| 4 | `broken_ratio` | Total broken bikes / total capacity |
| 5 | `depot_queue_ratio` | Repaired bikes / (repaired + in-repair) at depot |
| 6 | `shift_remaining` | `(shift_end - time_of_day) / shift_length` ∈ [0, 1] |
| 7 | `mean_load_ratio` | Mean functional cargo ratio across all vehicles |

**Why cyclic time encoding?** A raw `hour/24` has a discontinuity at midnight: 23:59 → 0. The (sin, cos) pair maps time onto the unit circle, giving a smooth periodic signal that respects daily demand patterns.

---

## 4. Neural Network Architecture

### 4.1 Design Rationale: Deep Sets

The bike-sharing state has a natural **set structure**: N stations and M vehicles are unordered collections. A flat MLP would treat position-in-vector as meaningful (station 3 always maps to input slots 24–31), preventing generalization if station ordering changes and failing to share knowledge across stations.

**Deep Sets (Zaheer et al., 2017)** solves this by:
1. Applying a **shared MLP (encoder)** independently to each element.
2. **Pooling** across elements with a permutation-invariant aggregation (mean/max/attention).
3. Combining pooled summaries to produce the final scalar.

```
V(S^x) = ρ( pool_n φ(s_n),  pool_m ψ(v_m),  g )
```

where φ = StationEncoder, ψ = VehicleEncoder, ρ = ValueMLP, g = global context.

### 4.2 Sub-Networks

#### StationEncoder (φ)
- **Input**: `[N × 8]`
- **Architecture**: Linear(8→64) → ReLU → Linear(64→64) → ReLU
- **Output**: `[N × 64]`
- Same weights applied to every station row (permutation-invariant).

#### VehicleEncoder (ψ)
- **Input**: `[M × 6]`
- **Architecture**: Linear(6→16) → ReLU → Linear(16→16) → ReLU
- **Output**: `[M × 16]`

#### AttentionPool
- **Input**: `[N × D]`
- **Architecture**: single linear score `w · x` (no bias), then softmax, then weighted sum.
- **Output**: `[D]`
- Learns to focus on critical stations (e.g. starving) rather than weighting all equally.

#### CrossAttentionPool (station pooling conditioned on vehicle state)
- **Query**: vehicle summary vector `[2 × vehicle_embed = 32]` projected to station dimension via `nn.Linear(32, 64)`.
- **Keys/Values**: station embeddings `[N × 64]`.
- **Scores**: dot product `(station_emb × query).sum(dim=1)` → softmax → weighted sum.
- **Output**: `[64]`
- Allows the station summary to dynamically reflect what the vehicle currently needs (e.g., if vehicle is empty, focus on stations with surplus bikes).

#### ValueMLP (ρ)
- **Input**: `[combined_dim = 2×64 + 2×16 + 8 = 168]`  
  *(2× because each branch uses attention+max dual pooling)*
- **Architecture**: Linear(168→128) → ReLU → Linear(128→64) → ReLU → Linear(64→1)
- **Output**: scalar `[1]`, unconstrained real value.

### 4.3 Forward Pass (active implementation)

The current `forward()` method processes branches in a specific order:

```python
# Step 1: Vehicle branch first
vehicle_embeddings = VehicleEncoder(vehicle_block)         # [M, 16]
vehicle_summary    = cat([AttentionPool(vh_emb),           # [16]
                          vh_emb.max(dim=0).values])        # [16]  → [32]

# Step 2: Station branch conditioned on vehicle summary
station_embeddings = StationEncoder(station_block)         # [N, 64]
station_summary    = cat([CrossAttentionPool(st_emb,        # [64]
                               vehicle_summary),
                          st_emb.max(dim=0).values])        # [64]  → [128]

# Step 3: Combine and score
combined = cat([station_summary, vehicle_summary, global])  # [168]
value    = ValueMLP(combined)                               # [1]
```

**Why vehicle first?** The CrossAttentionPool uses the vehicle summary as a query to weight stations. Vehicle must be encoded before stations so its summary can guide the station pooling.

### 4.4 Batched Forward Pass

`forward_batch()` handles `[B, N, D]` tensors directly via PyTorch's natural broadcasting over leading batch dimensions. Used by `_compute_td_loss()` for efficient gradient computation — replaces a Python loop of B separate `forward()` calls with one kernel call (~30–50x speedup on MPS/GPU).

### 4.5 Parameter Count

With default dimensions (station_embed=64, vehicle_embed=16, value_hidden=128):
- StationEncoder: 8×64 + 64 + 64×64 + 64 = 4,672 params
- VehicleEncoder: 6×16 + 16 + 16×16 + 16 = 368 params
- AttentionPool (vehicle): 16 params
- CrossAttentionPool (station): 32×64 = 2,048 params
- ValueMLP: 168×128 + 128 + 128×64 + 64 + 64×1 + 1 = 29,953 params
- **Total ≈ ~37,000 parameters** (exact count printed at training start)

---

## 5. The NN's Role in the Rollout

### 5.1 How NNRolloutPolicy Uses the NN

`NNRolloutPolicy.get_best_action()` implements a **two-stage evaluation**:

**Stage 1 — Cheap pre-score (no simulator cloning):**
```python
for (mdp_action, sim_action) in all_candidates:
    post_state = PostDecisionState.apply(mdp_state, mdp_action)
    enc        = encode_state(post_state)
    v          = nn_model(enc["station_block"], ...)   # one forward pass
    pre_scores.append((v, sim_action))
pre_scores.sort(reverse=True)
top_candidates = pre_scores[:N_ROLLOUT_CANDIDATES]   # default: 5
```

**Stage 2 — Full Monte Carlo rollout (top 5 only):**
```python
for action in top_candidates:
    for scenario in range(num_scenarios):   # default: 8 (eval: 3)
        clone_sim = self._clone_simulator()
        assign NNGreedyPolicy to cloned vehicles
        apply candidate action to clone
        fast-forward clone for lookahead_minutes (default: 60)
        accumulated_reward = Σ γ^(t/60) * step_reward(t)
        terminal_value = _estimate_terminal_value(clone_state, vehicle_id)
        Q(action, scenario) = accumulated_reward + γ^(H/60) * terminal_value
    mean_Q[action] = mean over scenarios
return argmax mean_Q
```

### 5.2 Terminal Value Estimation — The NN's Exact Call Site

```python
def _estimate_terminal_value(self, clone_sim_state, vehicle_id):
    # 1. Snapshot terminal simulator state as MDPState
    terminal_mdp_state = extract_mdp_state(
        sim_state=clone_sim_state,
        active_vehicle_id=vehicle_id,
        config=self._mdp_config,
        depot_id=self.depot_id,
        shift_end_time=_shift_end,
    )
    # 2. Encode into tensors
    encoded = encode_state(terminal_mdp_state)
    # 3. NN forward pass (no gradient)
    with torch.no_grad():
        value_tensor = self.nn_model(
            encoded["station_block"].to(device),
            encoded["vehicle_block"].to(device),
            encoded["global_context"].to(device),
        )
    return value_tensor.item()   # Python float
```

This replaces the linear VFA call `θᵀ φ(S^x_terminal)` in `HybridRolloutPolicy` with `NNValueNetwork(encode_state(S^x_terminal))`. The Q-value formula:

```
Q(a, ω) = Σ_t γ^(t/60) · r_t  +  γ^(H/60) · V_NN(S^x_terminal)
```

### 5.3 NNGreedyPolicy as Base Policy During Rollout

During fast-forward, cloned vehicles need a policy to make routing decisions. `NNGreedyPolicy` is assigned as their policy. It:
1. Extracts `MDPState` from the clone.
2. Generates candidates via `generate_candidates()`.
3. For each candidate, computes `PostDecisionState.apply()` + `encode_state()` + one NN forward pass.
4. Returns the `sim.Action` with the highest V value.

This prevents recursive rollout (NNRolloutPolicy calling itself inside a clone) while still using the NN's learned value judgments to guide the cloned vehicle.

### 5.4 How NN Output Influences Decisions

The NN produces a **scalar estimate of future penalty** (negative values, since rewards are penalties). The rollout aggregates this with accumulated reward from the simulation window to produce a Q-value per action per scenario. The action with the highest mean Q-value across scenarios is selected.

The NN does **not** directly output actions — it only outputs a scalar value used in the objective function of the outer optimization (argmax over candidates).

---

## 6. Training Pipeline

### 6.1 Episode Structure

Each training episode spans **14 simulation days** with seed `seed_offset + episode_index`:

```
Days 1–2   WARMUP    GreedyPolicy drives system
                     No TD updates
                     Purpose: build realistic non-trivial initial state
                     without biasing NN weights on pristine early data

Days 3–14  LEARNING  NNLearningPolicy
                     Boltzmann action selection (temperature τ, annealing)
                     Records (S^x_prev, r_k, S^x_chosen) transitions
                     n-step return accumulation (N_STEP_RETURN=3)
```

After each episode: batch gradient updates from replay buffer.

### 6.2 Per-Decision Flow (NNLearningPolicy)

```
get_best_action(state, vehicle):
  1. r_k = RewardCalculator.compute_step_reward(state.metrics)
         [Δ starvations + 0.7·Δ congestions since last call]

  2. mdp_state = extract_mdp_state(sim_state, vehicle)

  3. pairs = generate_candidates(state, vehicle, return_pairs=True)
         [enumerate routing × maintenance candidate actions]

  4. For each valid (mdp_action, sim_action):
         post_state = PostDecisionState.apply(mdp_state, mdp_action)
         encoded    = encode_state(post_state)
         V(post)    = online_model(encoded)
         [invalid PostDecisionState: skip candidate entirely]

  5. idx = Boltzmann_select(V_values, tau)
         [τ→0: greedy; τ→∞: uniform random]

  6. If _prev_post_encoded is not None:
         append (S^x_{k-1}, r_k) to n-step pending window
         If len(window) >= N_STEP_RETURN:
             G_n = Σ γ^i · r_i   (for i=0..N_STEP_RETURN-1)
             replay_buffer.push(S^x_0, G_n, S^x_chosen, n=3, done=False)
             pop oldest from window

  7. _prev_post_encoded = post_encodings[idx]
  8. return pairs[idx][1]   (sim.Action)
```

At episode end, `flush_terminal_transition()` drains remaining n-step window with `done=True` (no bootstrap from terminal state).

### 6.3 Replay Buffer

```
Class: ReplayBuffer (deque, max_size=50,000)

Each entry: (encoded_cur, reward, encoded_next, n_steps, done)
  - encoded_*: dict with station_block, vehicle_block, global_context tensors
  - reward:    accumulated n-step return G_n (not single-step)
  - n_steps:   used to compute correct γ^n exponent
  - done:      True for terminal transitions (no bootstrap)

Key design choices:
  - POST-DECISION states stored: removes demand randomness from NN input
  - Random sampling: breaks temporal correlation in gradient estimates
  - Shared across ALL episodes: early exploratory transitions remain available
  - Min buffer size (256): gradient updates only start once buffer has this many entries
```

### 6.4 Gradient Update Loop (per episode)

```python
n_updates = min(max(1, len(buffer) // batch_size) * 4, 400)
for step in range(n_updates):
    batch = replay_buffer.sample(batch_size)           # 128 random transitions
    loss, debug = _compute_td_loss(online, target, batch, gamma, normalizer)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 0.5)  # gradient clipping
    optimizer.step()
    # Polyak target update every gradient step:
    θ_target ← (1 - polyak) · θ_target + polyak · θ_online
```

### 6.5 Target Network

Two identical `NNValueNetwork` instances are maintained:
- **Online model** (`online_model`): updated every gradient step via backprop.
- **Target model** (`target_model`): `requires_grad=False`; updated via Polyak soft-copy or periodic hard-copy.

**Why a target network?** In TD learning, both the prediction V(S^x_cur) and the target `r + γ·V(S^x_next)` depend on the same network. Without a frozen target, the target shifts with every gradient step, creating a moving target problem (the "deadly triad" in neural TD). The target network lags behind the online model, providing stable regression targets.

**Polyak (soft) update:**
```
θ_target ← (1 - τ) · θ_target + τ · θ_online      (τ = 0.0005 per gradient step)
```
After 400 gradient steps per episode, the target is ~18% online weights per episode (soft tracking, not hard copy). This was found to be more stable than the original hard copy every N episodes.

---

## 7. Loss Function, Optimizer, and Regularization

### 7.1 TD Loss: n-step Huber

The loss for a mini-batch of B transitions:

```
TD target:   y_i = G_n^i + (1 - done_i) · γ^{n_i} · V_target(S^x_{next,i})

where G_n^i = Σ_{j=0}^{n-1} γ^j · r_{i+j}    (n-step accumulated return)

Loss = (1/B) · Σ_i Huber_δ(V_online(S^x_{cur,i}) - y_i)
```

**Huber loss** (δ=1.0):
```
L(δ) = 0.5 · δ²                  if |δ| ≤ 1
       1.0 · (|δ| - 0.5)          if |δ| > 1
```
This is less sensitive to large TD errors (outliers from rare catastrophic starvation events) than MSE, but still provides a quadratic gradient near 0 for precision.

### 7.2 Reward Normalization

Raw rewards range from −76.7 to 0 (mean ≈ −2.7). Without normalization, a single catastrophic step creates a gradient ≈5900× larger than a typical step.

**Fixed-range normalization:**
```python
def normalize(reward, n_steps, gamma):
    discount_sum = Σ_{i=0}^{n-1} γ^i
    return reward / (76.7 * discount_sum)
```
This keeps TD targets in [−1, 0] regardless of n. For n=3: divides by 76.7 × (1 + 0.99 + 0.98) ≈ 227.8.

`RewardNormalizer` maintains EMA statistics (α=0.005, half-life ≈ 139 samples) for monitoring, but the actual normalization uses the fixed-range formula (not EMA-based z-scoring).

### 7.3 Optimizer: Adam

```python
optimizer = optim.Adam(online_model.parameters(), lr=lr_start, weight_decay=1e-4)
```

**Adam** was chosen over SGD because:
- Adaptive per-parameter learning rates handle the non-uniform gradient magnitudes typical of RL/TD learning.
- Momentum terms help escape local flat regions in the loss landscape.
- Standard choice for neural TD / deep RL.

**Weight decay (L2 regularization, λ=1e-4):**  
Added to prevent value estimates from drifting to large positive or negative constants unconstrained by the training signal. Prior runs without weight decay showed near-zero value spread (all outputs collapsing toward a constant), which caused Boltzmann selection to degenerate to random.

### 7.4 Learning Rate Schedule

```
current_lr = lr_start + (lr_end - lr_start) * (episode / (total_episodes - 1))
lr_start = 5e-4   (episode 0)
lr_end   = 5e-5   (episode 799)
```
Linear decay. High early LR for fast initial learning; low late LR for fine-grained convergence.

### 7.5 Gradient Clipping

```python
nn.utils.clip_grad_norm_(online_model.parameters(), max_norm=0.5)
```

Clips the global gradient norm to 0.5. Prevents rare large TD spikes (from outlier transitions with very negative rewards) from destabilizing weights. Standard in neural TD/DQN.

---

## 8. Hyperparameters: Full Reference

### 8.1 Training Hyperparameters

| Parameter | Current Value | Effect |
|-----------|--------------|--------|
| `NUM_EPISODES` | 800 | Total training episodes |
| `EPISODE_DAYS` | 14 | Simulation days per episode |
| `WARMUP_DAYS` | 2 | Days using GreedyPolicy (no learning) |
| `LR_START` | 5e-4 | Adam initial learning rate |
| `LR_END` | 5e-5 | Adam final learning rate (linear decay) |
| `GAMMA` | 0.99 | TD discount factor |
| `TARGET_UPDATE_FREQ` | 10 | Fallback hard-copy frequency (only used if POLYAK=0) |
| `POLYAK` | 0.0005 | Soft target update rate per gradient step |
| `N_STEP_RETURN` | 3 | n-step TD return length |
| `BUFFER_SIZE` | 50,000 | Max replay buffer capacity |
| `BATCH_SIZE` | 128 | Mini-batch size per gradient update |
| `MIN_BUFFER_SIZE` | 256 | Minimum transitions before learning starts |
| `GRAD_CLIP_NORM` | 0.5 | Max gradient L2 norm |
| `TAU_START` | 0.05 | Boltzmann initial temperature |
| `TAU_END` | 0.001 | Boltzmann final temperature |
| `EVAL_GREEDY_EVERY` | 10 | Run greedy eval episode every N episodes |
| `DEBUG_EVERY` | 5 | Print full diagnostic summary every N episodes |

### 8.2 Architecture Hyperparameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `STATION_EMBED_DIM` | 64 | StationEncoder output dimension |
| `VEHICLE_EMBED_DIM` | 16 | VehicleEncoder output dimension |
| `VALUE_HIDDEN_DIM` | 128 | ValueMLP hidden layer width |
| `STATION_FEATURE_DIM` | 8 | Features per station |
| `VEHICLE_FEATURE_DIM` | 6 | Features per vehicle |
| `GLOBAL_FEATURE_DIM` | 8 | Global context features |

### 8.3 Rollout Hyperparameters (NNRolloutPolicy)

| Parameter | Value | Effect |
|-----------|-------|--------|
| `lookahead_minutes` | 60.0 | Simulation horizon H per rollout scenario |
| `num_scenarios` | 8 (train) / 3 (eval) | Monte Carlo samples per candidate action |
| `N_ROLLOUT_CANDIDATES` | 5 | Top-k pre-scored candidates sent to full rollout |
| `gamma` | 0.99 | Discount factor for rollout accumulation |

### 8.4 Hyperparameter Sensitivity Guide

| Hyperparameter | Too Low | Too High |
|----------------|---------|----------|
| `LR_START` | Slow convergence, may not escape initialization | Unstable loss, value explosion |
| `POLYAK` | Target lags too far behind (effectively frozen) | Target moves too fast, defeats its purpose |
| `TAU_START` | Near-greedy early; misses exploration | Uniform random early; wastes first episodes |
| `N_STEP_RETURN` | High variance targets (n=1 = noisy) | Inflated bootstrapped targets if not scaled correctly |
| `BUFFER_SIZE` | Old transitions squeezed out; less diversity | Memory pressure; diminishing returns |
| `GRAD_CLIP_NORM` | Weights update too slowly | Outlier TD spikes corrupt weights |
| `weight_decay` | Value drift / constant output collapse | Over-regularization; NN cannot fit complex patterns |
| `lookahead_minutes` | Short horizon; NN dominates too early | Very slow rollout (each scenario takes proportionally longer) |

---

## 9. Experiments and Tuning History

### 9.1 Pre-April-19 Failures

#### Problem 1: Mean Pooling Only
- **Symptom**: `mean_value_spread` near 0 → Boltzmann → uniform random → no learning signal.
- **Root cause**: Mean pooling of station embeddings diluted starved stations among the N-station average. A single starved station becomes invisible.
- **Fix**: Dual pooling (attention + max). `AttentionPool` learns to upweight critical stations. Max pooling captures the worst station unconditionally.

#### Problem 2: Sparse Travel Time Encoding
- **Symptom**: NN had no spatial context → could not learn distance-aware routing.
- **Root cause**: Station feature [3] (`eta_from_vehicle`) was 0 for every station except the destination (an old feature encoding the destination ID as a sparse indicator).
- **Fix**: `encode_station_block()` now computes `min(1.0, travel_time_to_station / 60)` for **all** stations, using `mdp_state.travel_times` (a dict populated by `extract_mdp_state()` once per decision epoch).

#### Problem 3: Replay Buffer Too Small (10k)
- **Symptom**: Gradient estimates noisy; early episode behavior dominated later training.
- **Fix**: Buffer increased to 50,000. More diverse replay → more stable gradient signal.

#### Problem 4: No Regularization
- **Symptom**: Value estimates drifted unconstrained. Near-zero spreads.
- **Fix**: `weight_decay=1e-4` in Adam. Soft L2 penalty keeps weights near zero unless the training signal forces them to grow.

#### Problem 5: `generate_candidates()` Missing `return_pairs`
- **Symptom**: Crash at training start.
- **Fix**: `candidate_generator.py` was updated to support `return_pairs=True`, returning `List[(MdpAction, sim.Action)]`.

### 9.2 April 19–20: Attention Pooling + Feature Engineering

These fixes were applied together in one batch. The first run with all fixes reached `greedy_sl ≈ 0.947` (previously ~0.94).

**Key change: Cross-attention pooling (conditioned station summarization)**  
After attention pooling improved things, a further architectural change was made: instead of the station pooling being unconditional (just learning which stations matter globally), the station summary is now conditioned on the vehicle state via `CrossAttentionPool`. The vehicle's current load and destination are used as a query to score which stations are most relevant given what the vehicle can actually do. This is more expressive: an empty vehicle looking for pickups attends to different stations than a full vehicle looking to drop off.

### 9.3 Tau Experiments

| TAU_START | Observed Behavior | Result |
|-----------|------------------|--------|
| 0.5 | Wide Boltzmann exploration early; slow convergence | Suboptimal SL; too much randomness persisted |
| 0.7 | Very wide exploration; model sometimes improved later but inconsistently | Best greedy SL = 0.948 (marginal) |
| 0.05 | Near-greedy from start; fast alignment to NN's own estimates | Current setting; better greedy SL in early episodes |

**Interpretation**: With the current architecture and feature set, early exploration (high tau) does not help because the NN is not yet able to provide a meaningful signal — purely random Boltzmann selection does not generate better training data than near-greedy. Starting near-greedy (tau=0.05) lets the NN align to its own self-consistent value estimates faster.

### 9.4 Weight Decay (Poly) Experiments

Tested on `tau=0.5, lr=0.001`:

| poly | greedy_sl plateau | Observation |
|------|-------------------|-------------|
| 0.0001 | ~0.94 | Insufficient regularization; value drift |
| 0.0005 | ~0.944–0.948 | Best observed; current default |
| 0.001 | ~0.944 | Slight over-regularization; NN output too constrained |

### 9.5 N-Step Return

**N=1** (standard TD(0)): High variance TD targets from single noisy step rewards.  
**N=3**: Reduces variance by accumulating 3 steps. However, requires dividing the normalization constant by the n-step discount sum (Σ γ^i for i=0..n-1), otherwise targets are inflated relative to n=1. After fixing the normalization, N=3 improved stability.

**Previous N=3 failure**: An earlier attempt with N=3 hurt `greedy_sl` (0.947→0.940) because the normalization still divided by the 1-step max penalty (76.7) instead of 76.7 × discount_sum. This inflated TD targets, causing instability. After fixing normalization, N=3 was beneficial.

### 9.6 Polyak vs. Hard Copy

Originally: hard copy every 10 episodes. Problem: target was 100% stale from random initialization for the first 10 episodes, then snapped to current weights, causing TD targets to jump discontinuously.

**Polyak update per gradient step (τ=0.0005)**: After 400 gradient steps, the target has absorbed ≈18% of the online model's weights per episode. This provides a smooth lag rather than periodic jumps. Empirically more stable.

---

## 10. Diagnostics, Metrics, and Training Curve Interpretation

### 10.1 CSV Log Fields

Each episode appends one row to `training_log_seed{N}_*.csv`:

| Field | Healthy Range | Red Flag |
|-------|--------------|----------|
| `mean_loss` | Decreasing first 50 eps, then stable | Sustained increase = LR/value issue |
| `service_level` | Trending up; ≥ 0.94 | Flat or declining after ep 100 |
| `greedy_sl` | Should exceed training SL; target ≥0.96 | Flat or below 0.94 after ep 200 |
| `mean_value_spread` | 0.05–0.5 | <0.001 = constant output; >5 = instability |
| `mean_reward` | Negative, trending toward 0 | Exactly 0 every step = reward missing |
| `pct_zero_reward` | 10–30% expected | >80% = reward too sparse |
| `fallback_rate` | 0.0 | >5% = bad PostDecisionState inputs |
| `pct_idx0` | High early (exploration), low late | >70% at episode 500 = NN collapsed |
| `mean_chosen_idx` | Trending from ~3–4 (exploration) to ~1–2 (exploitation) | Always 0 = NN picks first candidate always |
| `reward_norm_std` | ~4–8 (tracks reward scale) | <0.1 = normalization squashing everything |

### 10.2 How to Diagnose Specific Problems

**NN not discriminating candidates (mean_value_spread ≈ 0):**
- All station/vehicle embeddings being pooled to the same vector → all candidates look identical.
- Check: weight_decay too high? Feature normalization creating degenerate inputs?
- Fix options: reduce weight_decay, check encode_state output for NaN/constant values.

**NN collapsed to "always pick nearest" (pct_idx0 ≈ 100%):**
- NN learned that travel time ≈ 0 is the dominant feature, ignoring inventory states.
- `eta_from_vehicle` feature [3] dominates because it directly correlates with action cost.
- This is a known partial collapse risk; cross-attention on deficit features helps.

**Greedy eval SL much higher than training SL:**
- Boltzmann exploration during training is masking the NN's true quality.
- Good sign: means the NN learned useful values but is being forced to take random actions.
- Fix: reduce tau or move to smaller tau_start.

**Loss decreasing but greedy_sl flat:**
- NN is overfitting to replay buffer distribution without generalizing to greedy evaluation.
- Or: the value function estimates are improving but the rollout policy isn't using them effectively.
- Check: num_scenarios, lookahead_minutes settings for evaluation.

### 10.3 Current Training Results (tau=0.05 run, April 21)

Early episodes show:
- `greedy_sl` at ep 10: 0.9066 — still below prior VFA baseline (~0.947).
- `mean_value_spread`: ~0.002–0.004 (low but non-zero; model is discriminating slightly).
- `fallback_rate`: 0.0 (clean PostDecisionState computation).
- `pct_zero_reward`: ~18–21% (healthy; reward signal present).
- `pct_idx0`: 1–3% (low! model is NOT collapsed to always-nearest — good sign).
- `mean_chosen_idx`: ~24 (high! exploration is very broad early, expected with uniform Boltzmann).

The high `mean_chosen_idx` (~24) is suspicious — with near-greedy tau=0.05, we would expect the model to mostly pick the highest-valued action (index 0 after sorting). A mean chosen index of 24 suggests either: (a) candidates are not sorted before selection, or (b) the NN's values are nearly uniform so Boltzmann is still effectively random. This warrants investigation.

---

## 11. Known Issues and Potential Improvements

### 11.1 Undocumented Assumptions

1. **`travel_times` in MDPState are from the active vehicle to every station.** This is valid for single-vehicle instances. For multi-vehicle cases, each vehicle gets its own `travel_times` dict, but the encoder currently encodes the active vehicle's travel times in the station block — other vehicles see a zero `eta_from_vehicle` to all stations in their slot. This may be incorrect for multi-vehicle coordination.

2. **PostDecisionState changes travel_times for moved vehicles.** The current implementation passes `travel_times` through PostDecisionState unchanged (static distances). This is approximately correct for short horizons but misses that after an action, the vehicle is at a different location. The terminal travel times are recalculated from scratch in `extract_mdp_state()` at the terminal state, so this only affects the post-decision encoding used in the replay buffer — not the rollout terminal value.

3. **Reward normalization uses a fixed max penalty of 76.7.** This constant was measured from the specific instance `TD_W34_old` with the current reward config (starvation cost + 0.7 × congestion cost). If the instance or reward config changes, this constant needs to be updated.

4. **`NNGreedyPolicy` skips invalid PostDecisionState candidates.** If all candidates are invalid, `get_best_action` returns `None`. The simulator must handle `None` gracefully (fall back to DoNothing). This is not explicitly documented.

5. **The `forward()` (single-state) and `forward_batch()` paths are separate code.** A bug in one would not be caught by tests using the other. Currently there is no test verifying that `forward(x)` == `forward_batch(x.unsqueeze(0)).squeeze()`.

### 11.2 Architectural Questions

- **Cross-attention on stations conditioned on vehicle**: Currently uses vehicle embedding before station encoding, making the computation order: vehicles → cross-attend stations. The vehicle embedding uses `AttentionPool + max`, then this summary is used as a query for `CrossAttentionPool` over stations. The dimensional coupling (`vehicle_summary_dim = 2 * vehicle_embed_dim = 32` → projected to `station_embed_dim = 64`) is handled by `nn.Linear(32, 64)` in `CrossAttentionPool`. This is the active architecture but the commented-out original (unconditional `AttentionPool` for stations) is preserved in the code and can be reverted.

- **Old `forward_batch` is commented out.** The batched variant of the original (unconditional attention) pooling is in a `'''` block. The new batched cross-attention `forward_batch` is active. Both compute the same conceptual quantity but with different pooling mechanisms. Keeping both code paths as comments adds cognitive overhead.

### 11.3 Potential Improvements

1. **Validate that `forward(x) ≈ forward_batch(x.unsqueeze(0))`** — add a unit test.
2. **Investigate high `mean_chosen_idx` early in training** — either add sorting diagnostics or verify candidate indexing logic in `NNLearningPolicy.get_best_action()`.
3. **Multi-vehicle encoding**: Currently, the station block encodes travel times from only the active vehicle. For multi-vehicle instances, a vehicle-specific station block per vehicle would give each vehicle a correct spatial context.
4. **Reward normalization constant**: Make `76.7` configurable or compute it dynamically from `RewardCalculator.max_step_penalty()`.
5. **Remove commented-out dead code** in `NNGreedyPolicy.py`, `nn_model.py` (old `forward/forward_batch` blocks), and `evaluate_nn_rollout.py` (old version in triple-quote block) to reduce file length and confusion.
6. **Greedy eval SL vs. rollout eval SL gap**: The training loop saves the best model by `greedy_sl` (no rollout). Consider saving a separate checkpoint based on rollout evaluation SL from `evaluate_nn_rollout.py`, as greedy SL and rollout SL can diverge.
7. **No validation set / held-out episodes**: The model is evaluated on the same instance (`TD_W34_old`) with seeds used sequentially from `seed_offset`. There is no structural separation between training and evaluation seeds, meaning greedy SL during training is a noisy proxy (different random demand realizations) rather than a clean generalization test.
