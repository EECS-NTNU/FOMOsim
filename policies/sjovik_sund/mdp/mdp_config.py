"""
mdp_config.py – MDP scenario configurations for benchmarking.

Defines orthogonal switches for damage tracking, maintenance control,
and action availability across the state/action/transition layers.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MDPConfig:
    """
    Scenario configuration for MDP variant.

    Attributes
    ──────────
    track_damage           : whether station/vehicle state includes onsite/depot bikes
    allow_onsite_repairs   : whether actions can include onsite_repairs
    allow_depot_removals   : whether actions can include depot_removals
    """
    track_damage: bool
    allow_onsite_repairs: bool
    allow_depot_removals: bool

    @staticmethod
    def full_maintenance() -> "MDPConfig":
        """Full maintenance formulation (current production)."""
        return MDPConfig(
            track_damage=True,
            allow_onsite_repairs=True,
            allow_depot_removals=True,
        )

    #NB! If used, maintenance in simulation should be disabled to avoid mismatch between MDP state and sim state.
    @staticmethod
    def no_maintenance() -> "MDPConfig":
        """Functional-only MDP (base-case benchmarking vs XPilot)."""
        return MDPConfig(
            track_damage=False,
            allow_onsite_repairs=False,
            allow_depot_removals=False,
        )

