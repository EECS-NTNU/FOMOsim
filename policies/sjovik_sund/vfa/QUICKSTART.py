"""
Quick Start Guide - VFA for DSJBRMP

This guide shows the minimal steps to get started with the VFA system.
"""

# =============================================================================
# EXAMPLE 1: Train VFA Agent on Single Seed
# =============================================================================

from policies.sjovik_sund.vfa import VFAPolicy
from policies.sjovik_sund.run_simulation import run_simulation, SimulationConfig

# Create VFA policy with learning enabled
policy = VFAPolicy(
    learning_mode=True,        # Enable learning
    maintenance_enabled=True,  # Enable maintenance actions
    seed=42
)

# Run simulation
config = SimulationConfig()
simulator = run_simulation(
    seed=42,
    policy=policy,
    duration=24*5,            # 5 days
    num_vehicles=1,
    instance_name="TD_W34_old",
    config=config
)

# Save trained model
policy.save_agent('models/my_vfa_model.pkl')

# Print statistics
stats = policy.vfa_agent.get_statistics_summary()
print(f"Training completed: {stats['iteration']} iterations")
print(f"Final cost: {stats['mean_cost']:.2f}")


# =============================================================================
# EXAMPLE 2: Use Pre-trained Model
# =============================================================================

from policies.sjovik_sund.vfa import VFAPolicy

# Load trained model
policy = VFAPolicy(learning_mode=False)  # No learning
policy.load_agent('models/my_vfa_model.pkl')

# Run simulation in test mode
simulator = run_simulation(
    seed=100,
    policy=policy,
    duration=24*7,  # 1 week
    num_vehicles=2
)


# =============================================================================
# EXAMPLE 3: Custom Learning Parameters
# =============================================================================

from policies.sjovik_sund.vfa import (
    VFAPolicy, VFAAgent, LearningParameters,
    VFAFeatures, DemandForecaster
)

# Configure custom learning parameters
learning_params = LearningParameters(
    initial_learning_rate=0.02,      # Higher learning rate
    initial_epsilon=0.5,             # More exploration
    epsilon_decay=0.999,             # Slower decay
    failed_rental_cost=15.0,         # Higher penalty for stockouts
    discount_factor=0.98             # More weight on future
)

# Create components
forecaster = DemandForecaster()
features = VFAFeatures(forecaster)
agent = VFAAgent(features, learning_params, seed=42)

# Create policy
policy = VFAPolicy(vfa_agent=agent, learning_mode=True)

# Train
simulator = run_simulation(
    seed=42,
    policy=policy,
    duration=24*10  # Longer training
)


# =============================================================================
# EXAMPLE 4: Multi-Seed Training with CLI
# =============================================================================

# From terminal:
# python policies/sjovik_sund/vfa/train_vfa.py --mode train --seeds 42 43 44 45 46 --duration 120

# This will:
# - Train on seeds 42-46
# - Each seed runs 120 hours (5 days)
# - Save checkpoints after each seed
# - Save final trained model


# =============================================================================
# EXAMPLE 5: Test Trained Model on New Seeds
# =============================================================================

# From terminal:
# python policies/sjovik_sund/vfa/train_vfa.py --mode test \
#   --model models/vfa_trained_20260311_143022.pkl \
#   --seeds 100 101 102 103 104 \
#   --duration 168

# This will:
# - Load trained model
# - Test on seeds 100-104
# - Each seed runs 168 hours (1 week)
# - Write results to CSV


# =============================================================================
# EXAMPLE 6: Analyze Learned Feature Weights
# =============================================================================

# From terminal:
# python policies/sjovik_sund/vfa/train_vfa.py --mode analyze \
#   --model models/vfa_trained_20260311_143022.pkl

# This will print feature weights and their interpretation


# =============================================================================
# EXAMPLE 7: Monitor Learning Progress Programmatically
# =============================================================================

from policies.sjovik_sund.vfa import VFAPolicy
import matplotlib.pyplot as plt

# Create and train policy
policy = VFAPolicy(learning_mode=True, seed=42)

# ... run simulation ...

# Extract learning curves
td_errors = policy.vfa_agent.stats['td_errors']
costs = policy.vfa_agent.stats['costs']
theta_norms = policy.vfa_agent.stats['theta_norms']

# Plot
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

axes[0].plot(td_errors)
axes[0].set_ylabel('TD Error')
axes[0].set_title('Learning Convergence')

axes[1].plot(costs)
axes[1].set_ylabel('Immediate Cost')

axes[2].plot(theta_norms)
axes[2].set_ylabel('||θ||')
axes[2].set_xlabel('Iteration')

plt.tight_layout()
plt.savefig('learning_curves.png')


# =============================================================================
# EXAMPLE 8: Custom Demand Forecaster
# =============================================================================

from policies.sjovik_sund.vfa import DemandForecaster, DemandForecast

class MyForecaster(DemandForecaster):
    """Custom forecaster using external model."""
    
    def forecast(self, station_id, time, horizon=1.0):
        # Get day/hour
        day_of_week = int(time // (24 * 60)) % 7
        hour_of_day = int((time // 60) % 24)
        
        # Custom logic
        if hour_of_day >= 7 and hour_of_day <= 9:
            # Morning rush
            rental_rate = 3.0 * horizon
            return_rate = 0.5 * horizon
        elif hour_of_day >= 17 and hour_of_day <= 19:
            # Evening rush
            rental_rate = 0.5 * horizon
            return_rate = 3.0 * horizon
        else:
            rental_rate = 1.0 * horizon
            return_rate = 1.0 * horizon
        
        return DemandForecast(
            station_id=station_id,
            time=time,
            expected_rentals=rental_rate,
            expected_returns=return_rate,
            rental_variance=rental_rate,
            return_variance=return_rate
        )

# Use custom forecaster
from policies.sjovik_sund.vfa import VFAFeatures, VFAAgent, LearningParameters

forecaster = MyForecaster()
features = VFAFeatures(forecaster)
agent = VFAAgent(features, LearningParameters(), seed=42)
policy = VFAPolicy(vfa_agent=agent, learning_mode=True)


# =============================================================================
# EXAMPLE 9: Warm Start from Pre-trained Model
# =============================================================================

from policies.sjovik_sund.vfa import VFAPolicy

# Load pre-trained model
policy = VFAPolicy(learning_mode=True)  # Still learning
policy.load_agent('models/vfa_base.pkl')

# Continue training on new instance
simulator = run_simulation(
    seed=42,
    policy=policy,
    instance_name="OS_W31",  # Different instance
    duration=24*5
)

# Save updated model
policy.save_agent('models/vfa_base_finetuned.pkl')


# =============================================================================
# EXAMPLE 10: Integration with Existing Simulation Pipeline
# =============================================================================

from policies.sjovik_sund.run_simulation import test_seeds
from policies.sjovik_sund.vfa import VFAPolicy

# Create VFA policy
vfa_policy = VFAPolicy(learning_mode=True, seed=42)

# Test on multiple seeds (uses existing infrastructure)
test_seeds(
    list_of_seeds=[42, 43, 44],
    policy=vfa_policy,
    filename='vfa_results.csv',
    num_vehicles=1,
    duration=24*5,
    use_multiprocessing=False  # Keep False for learning
)

# Model is trained across all seeds
vfa_policy.save_agent('models/vfa_multiseed.pkl')


# =============================================================================
# TROUBLESHOOTING
# =============================================================================

"""
Issue: TD error not decreasing
Solution: Reduce learning rate or increase regularization

Issue: Agent not exploring enough
Solution: Increase initial_epsilon or slow epsilon_decay

Issue: Value estimates exploding
Solution: Increase l2_regularization or add feature normalization

Issue: Actions always the same
Solution: Check epsilon value (if 0, no exploration)

Issue: Slow training
Solution: Reduce number of candidate actions in action space splitting
"""


# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================

"""
Expected performance on TD_W34_old (1 vehicle, 5 days):

Untrained agent (ε=0.3):
- Failed rentals: ~50-100
- Failed returns: ~20-50
- Training time: ~5-10 min per seed

After 5 seeds (ε→0.05):
- Failed rentals: ~20-40
- Failed returns: ~10-20
- Training time: Same

After 10+ seeds:
- Failed rentals: ~10-25
- Failed returns: ~5-15
- Converged policy
"""
