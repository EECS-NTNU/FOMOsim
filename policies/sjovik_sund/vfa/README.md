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

```python
from policies.sjovik_sund.vfa import MDPState, StationInventory, VehicleStatus

# Extract state from simulator
state = StateObservationWrapper.extract_mdp_state(simul, vehicle_id)

# Access station inventory
station = state.get_station("S01")
print(f"Functional: {station.functional}, Onsite: {station.onsite}, Depot: {station.depot}")
```

**Key Classes**:
- `StationInventory`: $(f^{\text{func}}, f^{\text{onsite}}, f^{\text{depot}}, \text{capacity})$
- `VehicleStatus`: Location, cargo, capacity
- `MDPState`: Complete network state
- `Action`: $(ι, m_{\text{rep}}, m_{\text{rem}}, ρ)$
- `PostDecisionState`: Apply actions deterministically

### 2. Basis Functions (`vfa_features.py`)

Implements **separable value function**:
$$\bar{V}(S^x) \approx \sum_{n \in \mathcal{N}} \bar{v}_n(f_n^x) = \sum_n \sum_f \theta_f \phi_f(f_n^x)$$

**Features**:
1. **Functional Shortage**: $(E[\text{rentals}] - f^{\text{func}})^2$
2. **Return Rejection**: $(f^{\text{total}} + E[\text{returns}] - c)^2$
3. **Unattended Depot**: $f^{\text{depot}}$ (bikes needing removal)
4. **Unattended Onsite**: $f^{\text{onsite}}$ (bikes needing repair)
5. **Spatial Synergy**: Discounted functional cargo of inbound vehicles
6. **Utilization Deviation**: $(f^{\text{total}}/c - 0.5)^2$

```python
from policies.sjovik_sund.vfa import VFAFeatures, DemandForecaster

forecaster = DemandForecaster()
features = VFAFeatures(forecaster)

# Compute features for a state
feature_vector = features.feature_vector(post_decision_state)
```

### 3. VFA Agent (`vfa_agent.py`)

Implements **Temporal Difference (TD) Learning**:

$$\theta_{k+1} = \theta_k - \alpha_k \nabla_\theta [\theta^T \phi(S_k^x) - \hat{v}_k]$$

where:
$$\hat{v}_k = C_k + \gamma \bar{V}(S_{k+1})$$

```python
from policies.sjovik_sund.vfa import VFAAgent, LearningParameters

# Configure learning
params = LearningParameters(
    initial_learning_rate=0.01,
    initial_epsilon=0.3,  # Exploration rate
    discount_factor=0.95
)

# Create agent
agent = VFAAgent(features, params, seed=42)

# Evaluate state value
value = agent.evaluate_value(post_state)

# Update from transition
agent.update_theta(post_state, observed_cost, next_state)
```

**Learning Features**:
- **ε-greedy exploration**: Balance exploration vs. exploitation
- **Learning rate decay**: Converge to optimal policy
- **Feature normalization**: Improve numerical stability
- **L2 regularization**: Prevent overfitting
- **Experience replay** (optional): Batch updates

### 4. VFA Policy (`vfa_policy.py`)

Integrates VFA agent with the simulator's policy interface.

```python
from policies.sjovik_sund.vfa import VFAPolicy

# Training mode (learning enabled)
policy = VFAPolicy(learning_mode=True, maintenance_enabled=True, seed=42)

# Or load pre-trained agent
policy = VFAPolicy(learning_mode=False)
policy.load_agent('models/vfa_trained.pkl')
```

**Action Space Splitting** (for computational efficiency):
1. **Micro-step**: Optimize $(ι, m_{\text{rep}}, m_{\text{rem}})$ at current station
2. **Macro-step**: Evaluate routing $\rho$ using VFA

## Usage Examples

### Training a New Agent

```python
from policies.sjovik_sund.run_simulation import run_simulation, SimulationConfig
from policies.sjovik_sund.vfa import VFAPolicy

# Create VFA policy with learning enabled
policy = VFAPolicy(learning_mode=True, maintenance_enabled=True)

# Run simulation (agent learns during execution)
config = SimulationConfig()
simulator = run_simulation(
    seed=42,
    policy=policy,
    duration=24*5,  # 5 days
    num_vehicles=1,
    instance_name="TD_W34_old",
    config=config
)

# Save learned model
policy.save_agent('models/vfa_seed42.pkl')

# Print learning statistics
policy.vfa_agent.print_learning_status()
```

### Using a Pre-trained Agent

```python
# Load trained agent
policy = VFAPolicy(learning_mode=False)
policy.load_agent('models/vfa_trained.pkl')

# Run simulation in exploitation mode
simulator = run_simulation(
    seed=100,
    policy=policy,
    duration=24*7,  # 1 week
    num_vehicles=2
)
```

### Multi-Seed Training

```python
from policies.sjovik_sund.run_simulation import test_seeds

seeds = range(42, 52)  # 10 different seeds
policy = VFAPolicy(learning_mode=True)

test_seeds(
    list_of_seeds=seeds,
    policy=policy,
    filename='vfa_results.csv',
    duration=24*5
)
```

## Hyperparameter Tuning

Key parameters in `LearningParameters`:

```python
params = LearningParameters(
    # Learning rate schedule
    initial_learning_rate=0.01,      # α₀
    learning_rate_decay=0.9999,      # Multiplicative decay
    min_learning_rate=0.001,         # Floor
    
    # Exploration
    initial_epsilon=0.3,             # ε₀ (30% random actions)
    epsilon_decay=0.9995,            # Decay toward exploitation
    min_epsilon=0.05,                # Minimum exploration
    
    # Regularization
    l2_regularization=0.001,         # λ for ||θ||²
    
    # Costs
    failed_rental_cost=10.0,         # Penalty for stockout
    failed_return_cost=5.0,          # Penalty for full station
    
    # Discount
    discount_factor=0.95             # γ (future value weight)
)
```

### Recommended Tuning Process

1. **Start with high exploration** (`initial_epsilon=0.5`) for diverse experiences
2. **Monitor TD error** convergence:
   ```python
   agent.vfa_agent.print_learning_status()  # Every 50 decisions
   ```
3. **Adjust learning rate** if TD error oscillates
4. **Decay epsilon** faster if agent converges slowly
5. **Increase regularization** if θ norm grows unbounded

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       SIMULATION                            │
│  (High-fidelity: Weibull failures, exact bike tracking)    │
└──────────────────┬──────────────────────────────────────────┘
                   │ Event: Vehicle arrives
                   ▼
┌─────────────────────────────────────────────────────────────┐
│            StateObservationWrapper                          │
│  Aggregate: Bike objects → (f^func, f^onsite, f^depot)     │
└──────────────────┬──────────────────────────────────────────┘
                   │ MDPState
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    VFAPolicy                                │
│  1. Generate feasible actions (micro + macro splitting)    │
│  2. Select action (ε-greedy or greedy)                     │
│  3. Convert to simulator Action                            │
└──────────────────┬──────────────────────────────────────────┘
                   │ Action
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    VFAAgent                                 │
│  • Evaluate: V̄(S^x) = θᵀφ(S^x)                            │
│  • Select: arg min_x [C(S^x) + V̄(S^x)]                    │
│  • Learn: θ ← θ - α∇[V̄(S^x) - v̂]                         │
└──────────────────┬──────────────────────────────────────────┘
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
┌──────────────────┐  ┌──────────────────┐
│  VFAFeatures     │  │ DemandForecaster │
│  • Shortage      │  │ • Historical     │
│  • Rejection     │  │ • Time patterns  │
│  • Damage        │  │ • Online update  │
│  • Synergy       │  └──────────────────┘
└──────────────────┘
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

```python
# Automatic status every 50 decisions
VFA Learning Status (Iteration 1000)
================================================================================
  Learning rate: 0.005123
  Epsilon: 0.182341
  θ norm: 2.3451
  Mean TD error (last 100): 0.0234
  Mean cost (last 100): 5.23
  Mean value estimate: -12.45
================================================================================
```

### Post-Training Analysis

```python
# Extract statistics
stats = policy.vfa_agent.get_statistics_summary()
print(f"Total iterations: {stats['iteration']}")
print(f"Final θ: {policy.vfa_agent.theta}")

# Plot learning curves
import matplotlib.pyplot as plt

td_errors = policy.vfa_agent.stats['td_errors']
plt.plot(td_errors)
plt.xlabel('Iteration')
plt.ylabel('TD Error')
plt.title('Learning Convergence')
plt.show()
```

## Extending the System

### Custom Features

```python
class CustomFeatures(VFAFeatures):
    def compute_station_features(self, station, state, forecast_horizon=2.0):
        features = super().compute_station_features(station, state, forecast_horizon)
        
        # Add custom feature
        features['custom_metric'] = my_custom_function(station, state)
        
        return features
```

### Custom Demand Forecasts

```python
class MLDemandForecaster(DemandForecaster):
    def __init__(self, model_path):
        super().__init__()
        self.ml_model = load_model(model_path)
    
    def forecast(self, station_id, time, horizon=1.0):
        # Use ML model for forecasting
        prediction = self.ml_model.predict(station_id, time, horizon)
        return DemandForecast(...)
```

## Performance Tips

1. **Feature Scaling**: Features are automatically normalized (Welford's algorithm)
2. **Parallel Training**: Run multiple seeds in parallel, then ensemble
3. **Warm Start**: Load pre-trained θ for new scenarios
4. **Curriculum Learning**: Start with short horizons, gradually increase
5. **Batch Updates**: Enable experience replay for smoother learning

## References

- Powell, W. B. (2011). *Approximate Dynamic Programming*. Wiley.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.

## Authors

DSJBRMP Research Team  
Implementation Date: March 2026

## License

[Your License Here]
