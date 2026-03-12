"""
VFA Package for DSJBRMP  –  Time-Indexed Linear VFA + Offline Episodic Training

Main Components:
    LinearVFAPolicy       – feature extraction, VFA scoring, Boltzmann selection,
                            TD(0) updates, save/load
    EpisodeTrainingPolicy – episodic warm-up / learning phase router
    train_vfa.py          – offline training loop (run directly or call train())

MDP formulation (state, actions, post-decision state) lives in:
    policies/sjovik_sund/mdp/mdp_formulation.py

Quickstart:
    # Offline training (recommended)
    from policies.sjovik_sund.vfa.train_vfa import train
    vfa = train(num_episodes=200, save_path=Path('models/my_vfa.pkl'))

    # Load frozen model for deployment / rollout
    from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy
    vfa = LinearVFAPolicy.load(Path('models/my_vfa.pkl'))
"""

from .LinearVFAPolicy import LinearVFAPolicy, EpisodeTrainingPolicy
from .vfa_features import FEATURE_NAMES, N_FEATURES, extract as extract_features, as_dict as features_as_dict

__all__ = [
    'LinearVFAPolicy',
    'EpisodeTrainingPolicy',
    'FEATURE_NAMES',
    'N_FEATURES',
    'extract_features',
    'features_as_dict',
]

__version__ = '2.0.0'
__author__ = 'DSJBRMP Research Team'
