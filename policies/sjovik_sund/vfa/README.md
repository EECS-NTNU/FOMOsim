# Value Function Approximation (VFA) for DSJBRMP

## Overview

This module implements an **Approximate Dynamic Programming (ADP)** solver with **Value Function Approximation (VFA)** for the **Dynamic Stochastic Joint Bike Rebalancing and Maintenance Problem (DSJBRMP)**.

## Problem Formulation

### MDP Structure

#### State Space $S_k$
Decision epochs occur when a vehicle arrives at a station at time $t_k$:

- **Station Inventories**: $f_k^n = (f_k^{n,\text{func}}, f_k^{n,\text{onsite}}, f_k^{n,\text{depot}})$
  - `functional`: Rentable bikes
  - `onsite`: Bikes needing on-site repair
  - `depot`: Bikes requiring depot removal

- **Vehicle Status**: $V_k^v = (n_k^v, \alpha_k^v, f_k^{v,\text{func}}, f_k^{v,\text{depot}})$
  - Current/destination station
  - Arrival time
  - Functional and damaged cargo

#### Action Space $x \in \mathcal{X}_{S_k}$
$x = (\iota^x, m_{\text{rep}}^x, m_{\text{rem}}^x, \rho^x)$

- $\iota^x$: **Rebalancing** (+ delivery, - pickup)
- $m_{\text{rep}}^x$: **On-site repairs**
- $m_{\text{rem}}^x$: **Depot removals**
- $\rho^x$: **Next station** (routing)

#### Post-Decision State $S_k^x$
Deterministic state immediately after action, before stochastic events.

#### Reward Function
Minimize:
- Failed rentals (station has 0 functional bikes)
- Failed returns (station at capacity)

### Bellman Equation

$$V(S_k) = \min_{x \in \mathcal{X}_{S_k}} \big[ C(S_k^x) + \bar{V}(S_k^x) \big]$$

where:
- $C(S_k^x)$: Immediate cost
- $\bar{V}(S_k^x) \approx \theta^T \phi(S_k^x)$: Approximate value function

## Implementation

### 1. State Representation (`vfa_state.py`)

Unchanged — provides helper dataclasses used internally by `LinearVFAPolicy`.
`LinearVFAPolicy` works directly with the live `sim.State` object that the
simulator passes to `get_best_action()`; no manual extraction is required.

**Key Classes** (for reference / rollout integration):
- `StationInventory`: $(f^{\text{func}}, f^{\text{onsite}}, f^{\text{depot}}, \text{capacity})$
- `VehicleStatus`: Location, cargo, capacity  
- `MDPState`: Complete network state snapshot
- `PostDecisionState`: Deterministic state after an action

### 2. Feature Vector `φ(S^x)` — five network-level scalars

`LinearVFAPolicy.extract_features()` builds the feature vector from the
**post-decision state** using fully vectorised numpy operations (no Python
for-loops over stations after the initial inventory read).

| # | Name | Formula |
|---|------|---------|
| φ₁ | Rebalancing imbalance | $\sum_i \lvert I_i^{\text{func}} - \hat{I}_i^{\text{func}} \rvert$ |
| φ₂ | Trailer cannibalization | $q_v^{\text{depot}} / K$ |
| φ₃ | Global onsite backlog | $\sum_i I_i^{\text{onsite}}$ |
| φ₄ | Demand-weighted depot backlog | $\sum_i I_i^{\text{depot}} \times \lambda_i$ |
| φ₅ | Depot pull | $\phi_2 \times \text{dist}(v, \text{depot})$ |

where $\hat{I}_i^{\text{func}}$ is the time-indexed target state and
$\lambda_i$ is the time-averaged arrival rate at station $i$.

```python
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy

policy = LinearVFAPolicy()
# phi is a numpy array of shape (5,)
phi = policy.extract_features(state, vehicle, delta_func=2)
print(f'V(S^x) = {policy.value(phi):.4f}')
```

### 3. `LinearVFAPolicy` (`LinearVFAPolicy.py`)

Single class that combines feature extraction, VFA scoring, Boltzmann
selection, and TD(0) updates.  Integrates directly with the simulator via
the `Policy` base-class interface.

```python
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy

# Exploitation (frozen θ after training)
policy = LinearVFAPolicy.load('models/my_vfa.pkl')  # learning_mode=False

# Training (Boltzmann + TD updates)
policy = LinearVFAPolicy(
    n_features    = 5,
    alpha         = 0.01,
    gamma         = 0.99,
    tau           = 5.0,    # Boltzmann temperature
    learning_mode = True,
    seed          = 42,
)
```

**TD(0) update rule** applied at every vehicle-decision during the learning phase:

$$\theta \leftarrow \theta + \alpha \bigl( r + \gamma V(S^x_{\text{next}}) - V(S^x_{\text{cur}}) \bigr) \phi(S^x_{\text{cur}})$$

**Action generation** uses action-space splitting:
1. **Micro-step** (inventory): push current station toward its target state (greedy, fixed)
2. **Macro-step** (routing): evaluate the `N_CANDIDATES = 8` nearest next stations with VFA

### 4. `EpisodeTrainingPolicy` (`LinearVFAPolicy.py`)

Episodic wrapper created fresh each episode; routes decisions to the correct
phase while sharing a single persistent `LinearVFAPolicy` (θ persists).

```python
from policies.sjovik_sund.vfa.LinearVFAPolicy import EpisodeTrainingPolicy
from policies.greedy_policy import GreedyPolicy
from helpers import timeInMinutes

WARMUP_END = timeInMinutes(hours=7) + 4 * 24 * 60   # 07:00 + 4 days

episode_policy = EpisodeTrainingPolicy(
    vfa_policy      = vfa,           # shared LinearVFAPolicy
    greedy_policy   = GreedyPolicy(),
    warmup_end_time = WARMUP_END,
)
# state.time < WARMUP_END  → GreedyPolicy  (no TD updates)
# state.time >= WARMUP_END → LinearVFAPolicy (Boltzmann + TD)
```

## Usage Examples

### Full Offline Training (recommended)

```bash
# Default: 200 episodes × 14 days, saves to models/vfa_trained_<ts>.pkl
python policies/sjovik_sund/vfa/train_vfa.py

# Custom options
python policies/sjovik_sund/vfa/train_vfa.py \
    --episodes 200 \
    --save models/my_vfa.pkl \
    --seed 0 \
    --instance TD_W34_37
```

### Training Programmatically

```python
from policies.sjovik_sund.vfa.train_vfa import train
from pathlib import Path

vfa = train(
    num_episodes  = 200,
    save_path     = Path('models/my_vfa.pkl'),
    seed_offset   = 0,
    instance_name = 'TD_W34_old',
)
print(f'Final θ: {vfa.theta}')
```

### Using a Frozen Pre-trained Model

```python
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
from policies.sjovik_sund.run_simulation import run_simulation, SimulationConfig
from pathlib import Path

# load() always sets learning_mode=False
vfa = LinearVFAPolicy.load(Path('models/my_vfa.pkl'))

simulator = run_simulation(
    seed=100,
    policy=vfa,
    duration=24*7,
    num_vehicles=1,
    instance_name='TD_W34_old',
    config=SimulationConfig(),
)
```

### Evaluating on Multiple Test Seeds

```python
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
from policies.sjovik_sund.run_simulation import test_seeds
from pathlib import Path

vfa = LinearVFAPolicy.load(Path('models/my_vfa.pkl'))  # frozen

test_seeds(
    list_of_seeds = list(range(100, 110)),
    policy        = vfa,
    filename      = 'vfa_test_results.csv',
    duration      = 24 * 7,
    use_multiprocessing = True,
)
```

## Hyperparameter Tuning

All hyperparameters are constructor arguments of `LinearVFAPolicy` and
constants at the top of `train_vfa.py`:

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `alpha` | `LinearVFAPolicy` | `0.01` | TD step size |
| `gamma` | `LinearVFAPolicy` | `0.99` | Discount factor |
| `n_features` | `LinearVFAPolicy` | `5` | φ vector length |
| `N_CANDIDATES` | `LinearVFAPolicy` (class attr) | `8` | Routing candidates per decision |
| `TAU_START` | `train_vfa.py` | `5.0` | Initial Boltzmann temperature |
| `TAU_END` | `train_vfa.py` | `0.1` | Final Boltzmann temperature |
| `NUM_EPISODES` | `train_vfa.py` | `200` | Training episodes |
| `EPISODE_DAYS` | `train_vfa.py` | `14` | Days per episode |
| `WARMUP_DAYS` | `train_vfa.py` | `4` | Greedy warm-up days (no TD) |

### Recommended Tuning Process

1. **Start with default τ schedule** (5.0 → 0.1 over 200 episodes) — well-calibrated for `TD_W34_old`.
2. **Plot the learning curve** (`_learning_curve.npy`) to diagnose:
   - Flat curve → increase `TAU_START` or `alpha`.
   - Oscillating curve → decrease `alpha`.
   - θ norm explodes → decrease `alpha`.
3. **Adjust `WARMUP_DAYS`** if the system needs more (complex degradation) or less warm-up time.
4. **Reduce `N_CANDIDATES`** if training is too slow (fewer routing options to evaluate per decision).

## Architecture Diagram

```
 OFFLINE TRAINING  (train_vfa.py)
 ─────────────────────────────────────────────────────────────────
 for episode in range(200):
   ┌──── Days 1-4: GreedyPolicy warm-up ─── no θ update ────────┐
   │  τ decays exponentially   5.0 → 0.1 over 200 episodes      │
   ├──── Days 5-14: LinearVFAPolicy ────── TD(0) updates ────────┤
   │                                                              │
   │  Each vehicle decision:                                      │
   │  1. Extract inventories from sim.State (one loop/station)   │
   │  2. Build φ(S^x) with numpy ops  → shape (5,)              │
   │  3. Boltzmann-select next station via  P∝exp(−V/τ)         │
   │  4. TD update:  θ += α(r + γV_next − V_cur) φ_cur          │
   └──────────────────────────────────────────────────────────────┘
   θ persists; only per-episode tracking state is reset

 Save frozen model  →  models/vfa_trained_<ts>.pkl
 ─────────────────────────────────────────────────────────────────

 ONLINE DEPLOYMENT  (future Rollout Algorithm)
 ─────────────────────────────────────────────────────────────────
 For each vehicle decision:
   ┌── Generate candidate actions (action-space splitting) ──────┐
   │   Micro: greedy inventory push toward target state          │
   │   Macro: N_CANDIDATES nearest next stations                 │
   ├── For each candidate action a: ────────────────────────────┤
   │   Rollout H decisions with default policy                   │
   │   Q(s,a) ≈ Σ γ^h c_h  +  γ^H V̄(S_H)   ← frozen VFA      │
   └── Select a* = argmin Q(s, a) ─────────────────────────────┘
```

## Key Design Decisions

### 1. **POMDP Simplification**
- **Simulator**: Tracks exact Weibull component ages (odometers)
- **VFA Agent**: Observes only aggregate counts $(f^{\text{func}}, f^{\text{onsite}}, f^{\text{depot}})$
- **Rationale**: VFA learns memoryless (exponential) approximation of failure dynamics through experience

### 2. **Post-Decision State VFA**
- Approximate $\bar{V}(S^x)$ instead of $V(S)$
- **Advantage**: Decouple action evaluation from stochastic transitions
- **Update**: Use next pre-decision state for bootstrapping

### 3. **Separable Value Functions**
- Network value = sum of station values
- **Advantage**: Reduces feature dimensionality from exponential to linear in number of stations
- **Trade-off**: Ignores cross-station correlations (acceptable approximation)

### 4. **Action Space Splitting**
- **Micro-optimization**: Sample local actions $(ι, m_{\text{rep}}, m_{\text{rem}})$
- **Macro-optimization**: Evaluate routing using VFA
- **Advantage**: Avoid enumerating exponential joint action space

## Monitoring Learning Progress

### During Training

`train_vfa.py` prints one line per episode:

```
Episode  42/200 | τ = 1.843 | ‖θ‖ = 3.2041 | θ̄  = -0.1203 | SL = 0.8712 | t = 324s
```

Checkpoints are saved every 50 episodes to `models/vfa_checkpoint_ep<N>.pkl`.

### Post-Training Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
from pathlib import Path

# Inspect learned weights
vfa = LinearVFAPolicy.load(Path('models/my_vfa.pkl'))
feature_names = ['φ₁ rebalancing', 'φ₂ trailer', 'φ₃ onsite', 'φ₄ depot-demand', 'φ₅ depot-pull']
for name, w in zip(feature_names, vfa.theta):
    print(f'  {name:25s} {w:+.6f}')

# Plot service-level learning curve
sl = np.load('models/my_vfa_learning_curve.npy')
plt.plot(sl, alpha=0.4, label='per-episode SL')
plt.plot(np.convolve(sl, np.ones(10)/10, mode='valid'), label='10-ep MA')
plt.xlabel('Episode')
plt.ylabel('Service level')
plt.title('VFA Training Convergence')
plt.legend()
plt.tight_layout()
plt.savefig('learning_curve.png')
```

## Extending the System

### Adding or Changing Features

Subclass `LinearVFAPolicy` and override `extract_features()`.
Remember to also update `n_features`:

```python
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
import numpy as np

class ExtendedVFA(LinearVFAPolicy):
    """Adds φ₆: global functional fill level."""

    def __init__(self, **kwargs):
        kwargs.setdefault('n_features', 6)
        super().__init__(**kwargs)

    def extract_features(self, state, vehicle, delta_func=0, delta_depot_cargo=0):
        phi5 = super().extract_features(state, vehicle, delta_func, delta_depot_cargo)
        func, _, _ = self._extract_inventories(state)
        return np.append(phi5, float(np.sum(func)))
```

### Changing the Reward Signal

Override `_get_reward()` in `LinearVFAPolicy`.
The default penalises starvations and long-congestions:

```python
def _get_reward(self, state) -> float:
    cur_s = state.metrics.get_aggregate_value('starvation') or 0
    cur_c = state.metrics.get_aggregate_value('long_congestion') or 0
    reward = -(1.0 * (cur_s - self._prev_starvations) +
               0.5 * (cur_c - self._prev_congestions))
    self._prev_starvations = cur_s
    self._prev_congestions = cur_c
    return reward
```

## Performance Tips

1. **`N_CANDIDATES`**: Reduce from 8 to 4-5 to halve decision time with minimal quality loss.
2. **Warm Start**: `LinearVFAPolicy.load()` then set `learning_mode=True` and `tau` low for fine-tuning.
3. **Parallel evaluation**: `test_seeds(..., use_multiprocessing=True)` is safe because the frozen policy does not mutate state.
4. **Longer episodes**: Increasing `EPISODE_DAYS` from 14 to 28 captures stronger day-of-week patterns at the cost of slower training.

## References

- Powell, W. B. (2011). *Approximate Dynamic Programming*. Wiley.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.

## Authors

DSJBRMP Research Team  
Implementation Date: March 2026

## License

[Your License Here]
