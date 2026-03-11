"""
VFA Agent - Core Learning Algorithm for DSJBRMP

Implements Approximate Dynamic Programming with Value Function Approximation:
    V(S_k) = min_{x ∈ X(S_k)} [C(S_k^x) + V̄(S_k^x)]

where:
    - C(S_k^x) is the immediate cost (failed rentals/returns)
    - V̄(S_k^x) ≈ θᵀφ(S_k^x) is the approximate value function
    - θ are learned parameters
    - φ(S_k^x) are basis functions (features)

Update rule (Temporal Difference):
    θ_{k+1} = θ_k - α_k ∇_θ [θᵀφ(S_k^x) - v̂_k]
where v̂_k is the observed sample value from the next state.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import pickle
from pathlib import Path

from .vfa_state import MDPState, Action, PostDecisionState, StochasticTransition
from .vfa_features import VFAFeatures, DemandForecaster, FeatureNormalizer


@dataclass
class LearningParameters:
    """Hyperparameters for VFA learning."""
    
    # Learning rate schedule
    initial_learning_rate: float = 0.01
    learning_rate_decay: float = 0.9999
    min_learning_rate: float = 0.001
    
    # Exploration vs. exploitation
    initial_epsilon: float = 0.3  # Epsilon-greedy exploration
    epsilon_decay: float = 0.9995
    min_epsilon: float = 0.05
    
    # Regularization
    l2_regularization: float = 0.001
    
    # Experience replay (optional)
    use_experience_replay: bool = False
    replay_buffer_size: int = 10000
    replay_batch_size: int = 32
    
    # Value function initialization
    initial_theta: Optional[np.ndarray] = None
    
    # Cost function weights
    failed_rental_cost: float = 10.0
    failed_return_cost: float = 5.0
    congestion_cost: float = 3.0
    
    # Discount factor
    discount_factor: float = 0.95


@dataclass
class Experience:
    """Single transition experience for replay buffer."""
    pre_state: MDPState
    action: Action
    post_state: MDPState
    immediate_cost: float
    next_state: Optional[MDPState]  # None if terminal
    next_cost: float


class ExperienceReplayBuffer:
    """Buffer for storing and sampling past experiences."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer: List[Experience] = []
        self.position = 0
    
    def add(self, experience: Experience):
        """Add experience to buffer (circular)."""
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.max_size
    
    def sample(self, batch_size: int, rng: np.random.Generator) -> List[Experience]:
        """Sample random batch of experiences."""
        indices = rng.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]
    
    def size(self) -> int:
        """Current buffer size."""
        return len(self.buffer)


class VFAAgent:
    """
    Value Function Approximation Agent for DSJBRMP.
    
    Uses linear VFA with temporal difference learning.
    """
    
    def __init__(self, 
                 features: VFAFeatures,
                 learning_params: Optional[LearningParameters] = None,
                 seed: int = 42):
        """
        Initialize VFA agent.
        
        Args:
            features: Feature computation module
            learning_params: Learning hyperparameters
            seed: Random seed
        """
        self.features = features
        self.params = learning_params or LearningParameters()
        self.rng = np.random.default_rng(seed)
        
        # Initialize value function parameters θ
        num_features = self.features.get_num_features()
        if self.params.initial_theta is not None:
            self.theta = self.params.initial_theta.copy()
        else:
            # Initialize with small random values
            self.theta = self.rng.normal(0, 0.01, num_features)
        
        # Feature normalization
        self.normalizer = FeatureNormalizer(num_features)
        
        # Learning state
        self.iteration = 0
        self.current_learning_rate = self.params.initial_learning_rate
        self.current_epsilon = self.params.initial_epsilon
        
        # Experience replay
        if self.params.use_experience_replay:
            self.replay_buffer = ExperienceReplayBuffer(self.params.replay_buffer_size)
        
        # Statistics tracking
        self.stats = {
            'td_errors': [],
            'theta_norms': [],
            'value_estimates': [],
            'learning_rates': [],
            'costs': []
        }
    
    def evaluate_value(self, state: MDPState, use_raw_features: bool = False) -> float:
        """
        Evaluate V̄(S) = θᵀφ(S) for a given state.
        
        Args:
            state: State to evaluate (typically post-decision state)
            use_raw_features: If True, skip normalization
        
        Returns:
            Estimated value
        """
        features = self.features.feature_vector(state)
        
        if not use_raw_features:
            features = self.normalizer.normalize(features)
        
        value = np.dot(self.theta, features)
        return value
    
    def compute_immediate_cost(self, post_state: MDPState, 
                               transition: Optional[StochasticTransition] = None) -> float:
        """
        Compute immediate cost C(S^x) for post-decision state.
        
        Cost includes:
        - Failed rentals (demand when functional bikes = 0)
        - Failed returns (returns when station is full)
        - Congestion penalties
        
        Args:
            post_state: Post-decision state
            transition: Stochastic transition information (if available)
        
        Returns:
            Immediate cost
        """
        cost = 0.0
        
        if transition is not None:
            # Use actual observed transitions
            cost += (self.params.failed_rental_cost * 
                    transition.get_total_failed_rentals())
            cost += (self.params.failed_return_cost * 
                    transition.get_total_failed_returns())
        else:
            # Estimate based on current state (for lookahead)
            for station in post_state.stations.values():
                # Penalize low functional inventory
                if station.functional == 0:
                    cost += self.params.failed_rental_cost
                
                # Penalize near-capacity
                if station.total_bikes() >= station.capacity - 1:
                    cost += self.params.failed_return_cost
        
        return cost
    
    def update_theta(self, post_state: MDPState, observed_cost: float, 
                    next_state: Optional[MDPState] = None):
        """
        Update θ using temporal difference learning.
        
        TD Update:
            θ ← θ - α ∇_θ [V̄(S_k^x) - v̂_k]
        where:
            v̂_k = observed_cost (if terminal)
                 = observed_cost + V̄(S_{k+1}^{x'}) (if non-terminal)
        
        Args:
            post_state: Post-decision state S_k^x
            observed_cost: Observed immediate cost
            next_state: Next state S_{k+1} (None if terminal)
        """
        # Get features and current value estimate
        features = self.features.feature_vector(post_state)
        self.normalizer.update(features)
        features_norm = self.normalizer.normalize(features)
        
        current_value = np.dot(self.theta, features_norm)
        
        # Compute TD target
        if next_state is None:
            # Terminal state
            td_target = observed_cost
        else:
            # Non-terminal: bootstrap from next state value
            next_value = self.evaluate_value(next_state)
            td_target = observed_cost + self.params.discount_factor * next_value
        
        # TD error
        td_error = current_value - td_target
        
        # Gradient: ∇_θ V̄(S) = φ(S)
        gradient = features_norm
        
        # L2 regularization gradient: λθ
        reg_gradient = self.params.l2_regularization * self.theta
        
        # Update rule: θ ← θ - α[∇V̄ · td_error + λθ]
        self.theta -= self.current_learning_rate * (gradient * td_error + reg_gradient)
        
        # Update learning rate schedule
        self.current_learning_rate = max(
            self.params.min_learning_rate,
            self.current_learning_rate * self.params.learning_rate_decay
        )
        
        # Track statistics
        self.stats['td_errors'].append(td_error)
        self.stats['theta_norms'].append(np.linalg.norm(self.theta))
        self.stats['value_estimates'].append(current_value)
        self.stats['learning_rates'].append(self.current_learning_rate)
        self.stats['costs'].append(observed_cost)
        
        self.iteration += 1
    
    def experience_replay_update(self):
        """
        Perform batch update using experience replay.
        
        Sample a batch from replay buffer and update θ.
        """
        if not self.params.use_experience_replay:
            return
        
        if self.replay_buffer.size() < self.params.replay_batch_size:
            return
        
        # Sample batch
        batch = self.replay_buffer.sample(self.params.replay_batch_size, self.rng)
        
        # Compute batch gradient
        batch_gradient = np.zeros_like(self.theta)
        
        for exp in batch:
            # Get features
            features = self.features.feature_vector(exp.post_state)
            features_norm = self.normalizer.normalize(features)
            
            # Current value
            current_value = np.dot(self.theta, features_norm)
            
            # TD target
            if exp.next_state is None:
                td_target = exp.immediate_cost
            else:
                next_value = self.evaluate_value(exp.next_state)
                td_target = exp.immediate_cost + self.params.discount_factor * next_value
            
            # TD error
            td_error = current_value - td_target
            
            # Accumulate gradient
            batch_gradient += features_norm * td_error
        
        # Average gradient over batch
        batch_gradient /= len(batch)
        
        # Add regularization
        batch_gradient += self.params.l2_regularization * self.theta
        
        # Update
        self.theta -= self.current_learning_rate * batch_gradient
    
    def select_action_epsilon_greedy(self, state: MDPState, 
                                     feasible_actions: List[Action]) -> Action:
        """
        Select action using ε-greedy policy.
        
        Args:
            state: Current pre-decision state
            feasible_actions: List of feasible actions
        
        Returns:
            Selected action
        """
        if self.rng.random() < self.current_epsilon:
            # Explore: random action
            return self.rng.choice(feasible_actions)
        else:
            # Exploit: greedy action
            return self.select_greedy_action(state, feasible_actions)
    
    def select_greedy_action(self, state: MDPState, 
                            feasible_actions: List[Action]) -> Action:
        """
        Select action greedily: arg min_x [C(S^x) + V̄(S^x)].
        
        Args:
            state: Current pre-decision state
            feasible_actions: List of feasible actions
        
        Returns:
            Best action according to current value function
        """
        best_action = None
        best_value = float('inf')
        
        for action in feasible_actions:
            try:
                # Apply action to get post-decision state
                post_state = PostDecisionState.apply_action(state, action)
                
                # Evaluate: C(S^x) + V̄(S^x)
                immediate_cost = self.compute_immediate_cost(post_state)
                future_value = self.evaluate_value(post_state)
                total_value = immediate_cost + future_value
                
                if total_value < best_value:
                    best_value = total_value
                    best_action = action
            
            except ValueError:
                # Action not feasible (constraint violation)
                continue
        
        if best_action is None:
            # Fallback: random action
            best_action = self.rng.choice(feasible_actions)
        
        return best_action
    
    def decay_epsilon(self):
        """Decay exploration rate ε."""
        self.current_epsilon = max(
            self.params.min_epsilon,
            self.current_epsilon * self.params.epsilon_decay
        )
    
    def save_model(self, filepath: Path):
        """
        Save learned parameters to disk.
        
        Args:
            filepath: Path to save file
        """
        model_data = {
            'theta': self.theta,
            'iteration': self.iteration,
            'normalizer_stats': self.normalizer.get_statistics(),
            'learning_params': self.params,
            'stats': self.stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"VFA model saved to {filepath}")
    
    def load_model(self, filepath: Path):
        """
        Load learned parameters from disk.
        
        Args:
            filepath: Path to saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.theta = model_data['theta']
        self.iteration = model_data['iteration']
        
        # Restore normalizer statistics
        norm_stats = model_data['normalizer_stats']
        self.normalizer.mean = norm_stats['mean']
        self.normalizer.count = norm_stats['count']
        # Restore m2 from std
        self.normalizer.m2 = (norm_stats['std'] ** 2) * (self.normalizer.count - 1)
        self.normalizer.initialized = True
        
        print(f"VFA model loaded from {filepath}")
        print(f"  Iteration: {self.iteration}")
        print(f"  θ shape: {self.theta.shape}")
        print(f"  θ norm: {np.linalg.norm(self.theta):.4f}")
    
    def get_statistics_summary(self) -> Dict:
        """Get summary statistics for monitoring learning."""
        if len(self.stats['td_errors']) == 0:
            return {}
        
        recent_window = 100
        
        return {
            'iteration': self.iteration,
            'theta_norm': np.linalg.norm(self.theta),
            'learning_rate': self.current_learning_rate,
            'epsilon': self.current_epsilon,
            'mean_td_error': np.mean(self.stats['td_errors'][-recent_window:]),
            'mean_cost': np.mean(self.stats['costs'][-recent_window:]),
            'mean_value': np.mean(self.stats['value_estimates'][-recent_window:]),
        }
    
    def print_learning_status(self):
        """Print current learning status."""
        summary = self.get_statistics_summary()
        if not summary:
            return
        
        print(f"\n{'='*80}")
        print(f"VFA Learning Status (Iteration {summary['iteration']})")
        print(f"{'='*80}")
        print(f"  Learning rate: {summary['learning_rate']:.6f}")
        print(f"  Epsilon: {summary['epsilon']:.4f}")
        print(f"  θ norm: {summary['theta_norm']:.4f}")
        print(f"  Mean TD error (last 100): {summary['mean_td_error']:.4f}")
        print(f"  Mean cost (last 100): {summary['mean_cost']:.4f}")
        print(f"  Mean value estimate: {summary['mean_value']:.4f}")
        print(f"{'='*80}\n")
