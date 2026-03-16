"""
Quick Start Guide – VFA for DSJBRMP

New Architecture (March 2026)
──────────────────────────────
• mdp_formulation.py – canonical MDP dataclasses + extraction helpers
• mdp_config.py      – orthogonal scenario configuration switches
• vfa_features.py    – canonical feature registry and feature computation
• LinearVFAPolicy    – MDP-backed feature preparation + TD(0) + Boltzmann selection
• EpisodeTrainingPolicy – episodic warm-up/learning phase router
• train_vfa.py       – offline episodic training loop (200 episodes × 14 days)

After offline training the frozen θ vector is used as a tail-value estimator
inside a separate Rollout Algorithm (not implemented here).
"""

# =============================================================================
# EXAMPLE 1: Full Offline Training (recommended entry point)
# =============================================================================
# Run the training loop from the terminal:
#   python policies/sjovik_sund/vfa/train_vfa.py
#   python policies/sjovik_sund/vfa/train_vfa.py --episodes 200 --save models/my_vfa.pkl
#   python policies/sjovik_sund/vfa/train_vfa.py --episodes 50 --seed 100 --instance TD_W34_37
#
# This runs 200 episodes of 14 days each:
#   Days 1-4  : GreedyPolicy warm-up  (no TD updates)
#   Days 5-14 : VFA + Boltzmann exploration + TD(0) updates
#   τ decays exponentially from 5.0 → 0.1 over the 200 episodes
#
# Outputs:
#   models/vfa_trained_<timestamp>.pkl     – frozen θ vector
#   models/vfa_trained_<timestamp>_learning_curve.npy
#   models/vfa_checkpoint_ep0050.pkl  (every 50 episodes)


# =============================================================================
# EXAMPLE 2: Train Programmatically (single call)
# =============================================================================

from policies.sjovik_sund.vfa.train_vfa import train
from pathlib import Path

# Run the full 200-episode training loop and get back the frozen policy
vfa = train(
    num_episodes  = 200,
    save_path     = Path('models/my_vfa.pkl'),
    seed_offset   = 0,
    instance_name = 'TD_W34_old',
)

# vfa.theta is now frozen; use it as a tail-value estimator
print(f'θ = {vfa.theta}')


# =============================================================================
# EXAMPLE 3: Use a Pre-trained Frozen Model
# =============================================================================

from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
from policies.sjovik_sund.run_simulation import run_simulation, SimulationConfig
from pathlib import Path

# Load frozen model (learning_mode=False is the default when using .load())
vfa = LinearVFAPolicy.load(Path('models/my_vfa.pkl'))
print(f'Loaded θ: {vfa.theta}')

# Run simulation in pure exploitation mode (no exploration, no TD updates)
config = SimulationConfig()
simulator = run_simulation(
    seed=100,
    policy=vfa,
    duration=24*7,          # 1 week test run
    num_vehicles=1,
    instance_name='TD_W34_old',
    config=config,
)


# =============================================================================
# EXAMPLE 4: Custom Hyperparameters
# =============================================================================

from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy

# All hyperparameters are constructor arguments – easy to change
from policies.sjovik_sund.mdp.mdp_config import MDPConfig

policy = LinearVFAPolicy(
    n_features    = 5,      # φ vector length (change if you add/remove features)
    alpha         = 0.005,  # smaller α → slower but more stable learning
    gamma         = 0.99,   # discount factor
    tau           = 5.0,    # initial Boltzmann temperature (overridden by training loop)
    learning_mode = True,
    config        = MDPConfig.full_maintenance(),
    seed          = 42,
)

# Run a single episode manually
from policies.sjovik_sund.run_simulation import run_simulation, SimulationConfig
from helpers import timeInMinutes

WARMUP_END = timeInMinutes(hours=7) + 4 * 24 * 60  # 4-day warm-up

from policies.sjovik_sund.vfa.LinearVFAPolicy import EpisodeTrainingPolicy
from policies.greedy_policy import GreedyPolicy

episode_policy = EpisodeTrainingPolicy(
    vfa_policy      = policy,
    greedy_policy   = GreedyPolicy(),
    warmup_end_time = WARMUP_END,
)

simulator = run_simulation(
    seed          = 42,
    policy        = episode_policy,
    duration      = 24 * 14,   # 14-day episode
    num_vehicles  = 1,
    instance_name = 'TD_W34_old',
    config        = SimulationConfig(),
)
print(f'θ after episode: {policy.theta}')


# =============================================================================
# EXAMPLE 5: Plot the Learning Curve After Training
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# train_vfa.py saves the per-episode service level array alongside the model
sl = np.load('models/my_vfa_learning_curve.npy')

plt.figure(figsize=(10, 4))
plt.plot(sl, linewidth=1, label='Service level')
plt.plot(np.convolve(sl, np.ones(10)/10, mode='valid'), linewidth=2, label='10-ep MA')
plt.xlabel('Episode')
plt.ylabel('Service level  (1 − starvations / trips)')
plt.title('VFA Training Convergence')
plt.legend()
plt.tight_layout()
plt.savefig('learning_curve.png')
print(f'Best SL = {sl.max():.4f}  (episode {sl.argmax() + 1})')


# =============================================================================
# EXAMPLE 6: Inspect / Change the Canonical Feature Set
# =============================================================================
# The canonical feature registry lives in vfa_features.py.
# Permanent feature changes should be made there, then the VFA should be retrained.

from policies.sjovik_sund.vfa.vfa_features import FEATURE_NAMES, N_FEATURES, as_dict

print('Feature names:', FEATURE_NAMES)
print('Number of features:', N_FEATURES)

# Example: inspect a computed feature vector from the policy
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy

policy = LinearVFAPolicy()
phi = policy.extract_features(state, vehicle, delta_func=0, delta_depot_cargo=0)
print(as_dict(phi))

# If a temporary experiment still overrides extract_features(), note that
# inventory extraction now uses the canonical MDP snapshot and the helper
# signature is:
#
#   func, onsite, depot = self._extract_inventories(state, vehicle)


# =============================================================================
# EXAMPLE 7: Warm-Start / Fine-Tune on a New Instance
# =============================================================================

from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
from policies.sjovik_sund.vfa.train_vfa import train
from pathlib import Path

# Load a previously trained θ and continue training on a different instance.
# Because .load() sets learning_mode=False by default, flip it back.
vfa = LinearVFAPolicy.load(Path('models/my_vfa.pkl'))
vfa.learning_mode = True
vfa.tau = 1.0       # start with low temperature (already partially trained)
vfa.alpha = 0.005   # smaller step size for fine-tuning

# Fine-tune for 50 episodes on a different city
train(
    num_episodes  = 50,
    save_path     = Path('models/my_vfa_finetuned.pkl'),
    seed_offset   = 200,
    instance_name = 'OS_W31',
)


# =============================================================================
# EXAMPLE 8: Evaluate Frozen Model Across Multiple Test Seeds
# =============================================================================

from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
from policies.sjovik_sund.run_simulation import test_seeds
from pathlib import Path

# Load the frozen, trained policy
vfa = LinearVFAPolicy.load(Path('models/my_vfa.pkl'))  # learning_mode=False

# test_seeds uses multiprocessing internally – safe because θ is not modified
test_seeds(
    list_of_seeds = [100, 101, 102, 103, 104],
    policy        = vfa,
    filename      = 'vfa_test_results.csv',
    num_vehicles  = 1,
    duration      = 24 * 7,            # 1-week test horizon
    use_multiprocessing = True,
)


# =============================================================================
# TROUBLESHOOTING
# =============================================================================

"""
Issue: θ norm grows without bound during training
Solution: Reduce alpha (e.g. 0.001) or shorten the episode duration.

Issue: Boltzmann always picks the same action (no exploration)
Solution: TAU_START is too small.  Increase it in train_vfa.py (default 5.0).

Issue: Service level doesn't improve over episodes
Solution: Check that WARMUP_DAYS < EPISODE_DAYS.  Try more episodes or a
          higher learning rate.  Inspect the learning curve .npy file.

Issue: Policy is too greedy even during warm-up
Solution: Check that warmup_end_time is passed correctly to
          EpisodeTrainingPolicy.  The simulator clock starts at
          timeInMinutes(hours=START_HOUR), not at 0.

Issue: _lazy_init() called on every episode after domain change
Solution: This is expected when switching instances.  Set
          policy._initialized = False explicitly before the first episode
          on the new instance.

Issue: run_simulation complains about policy.weights
Solution: LinearVFAPolicy keeps self.weights = list(self.theta)
          in sync automatically after every td_update().
"""


# =============================================================================
# EXPECTED TRAINING TIMELINE  (TD_W34_old, 1 vehicle)
# =============================================================================

"""
Episodes 1–50   (τ: 5.0 → 1.2)  – heavy exploration;
                                   θ moves a lot, SL may be noisy.
Episodes 51–150 (τ: 1.2 → 0.3)  – exploitation increasing;
                                   SL trend should become visible.
Episodes 151–200 (τ: 0.3 → 0.1) – near-greedy; θ should stabilise.

Full 200-episode run:  ~2–4 hours on a laptop (CPU only).
Checkpoints every 50 episodes let you resume or evaluate earlier.

Baseline comparison (GreedyPolicy, same seeds):
  Run train_vfa.py with --episodes 1 to get a 1-episode greedy reference.
"""
