"""
VFA Package for Dynamic Stochastic Joint Bike Rebalancing and Maintenance Problem

This package implements Approximate Dynamic Programming (ADP) with Value Function 
Approximation (VFA) for solving the DSJBRMP.

Main Components:
- vfa_state: State space representation (stations, vehicles, actions)
- vfa_features: Basis functions for value function approximation
- vfa_agent: Core learning algorithm (temporal difference learning)
- vfa_policy: Policy interface for integration with simulator

Usage:
    from policies.sjovik_sund.vfa import VFAPolicy, VFAAgent, LearningParameters
    
    # Create policy with learning enabled
    policy = VFAPolicy(learning_mode=True, maintenance_enabled=True)
    
    # Or create with pre-trained agent
    agent = VFAAgent.load_model('path/to/model.pkl')
    policy = VFAPolicy(vfa_agent=agent, learning_mode=False)
"""

from .vfa_state import (
    MDPState,
    StationInventory,
    VehicleStatus,
    Action,
    PostDecisionState,
    StateObservationWrapper,
    StochasticTransition
)

from .vfa_features import (
    VFAFeatures,
    DemandForecaster,
    DemandForecast,
    FeatureNormalizer
)

from .vfa_agent import (
    VFAAgent,
    LearningParameters,
    Experience,
    ExperienceReplayBuffer
)

from .vfa_policy import VFAPolicy


__all__ = [
    # State representation
    'MDPState',
    'StationInventory',
    'VehicleStatus',
    'Action',
    'PostDecisionState',
    'StateObservationWrapper',
    'StochasticTransition',
    
    # Features
    'VFAFeatures',
    'DemandForecaster',
    'DemandForecast',
    'FeatureNormalizer',
    
    # Agent
    'VFAAgent',
    'LearningParameters',
    'Experience',
    'ExperienceReplayBuffer',
    
    # Policy
    'VFAPolicy',
]


__version__ = '1.0.0'
__author__ = 'DSJBRMP Research Team'
