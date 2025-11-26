"""
Phase 1: Basecase Weight Configurations

This module defines all weight combinations to test in Phase 1.
Goal: Find optimal service weights (w_S, w_C, w_D) WITHOUT maintenance consideration.

Maintenance reward r_M = 0 for all configurations in this phase.
"""

# Define weight configurations to test
# Format: [w_S, w_C, w_D, r_M]
# Note: r_M = 0 for all Phase 1 experiments

WEIGHT_CONFIGURATIONS = {
    # ========================================================================
    # BASELINE: Balanced approach
    # ========================================================================
    'baseline_balanced': [0.45, 0.45, 0.10, 0.0],
    
    # ========================================================================
    # STARVATION-FOCUSED: Prioritize preventing failed trips
    # ========================================================================
    'starvation_high': [0.70, 0.20, 0.10, 0.0],
    'starvation_medium': [0.60, 0.30, 0.10, 0.0],
    'starvation_extreme': [0.80, 0.15, 0.05, 0.0],
    
    # ========================================================================
    # CONGESTION-FOCUSED: Prioritize preventing full stations
    # ========================================================================
    'congestion_high': [0.20, 0.70, 0.10, 0.0],
    'congestion_medium': [0.30, 0.60, 0.10, 0.0],
    'congestion_extreme': [0.15, 0.80, 0.05, 0.0],
    
    # ========================================================================
    # DEVIATION-FOCUSED: Prioritize staying near target states
    # ========================================================================
    'deviation_high': [0.30, 0.30, 0.40, 0.0],
    'deviation_medium': [0.35, 0.35, 0.30, 0.0],
    'deviation_extreme': [0.25, 0.25, 0.50, 0.0],
    
    # ========================================================================
    # EQUAL WEIGHTS: All metrics weighted equally
    # ========================================================================
    'equal_all': [0.33, 0.33, 0.34, 0.0],
    
    # ========================================================================
    # STARVATION + CONGESTION: Balance these two, less on deviation
    # ========================================================================
    'starv_cong_balanced': [0.475, 0.475, 0.05, 0.0],
    'starv_cong_60_30': [0.60, 0.35, 0.05, 0.0],
    'starv_cong_30_60': [0.35, 0.60, 0.05, 0.0],
    
    # ========================================================================
    # FINE-TUNED: Small variations around baseline
    # ========================================================================
    'baseline_plus_starv': [0.50, 0.40, 0.10, 0.0],
    'baseline_plus_cong': [0.40, 0.50, 0.10, 0.0],
    'baseline_plus_dev': [0.40, 0.40, 0.20, 0.0],
}


def validate_weights(weights, tolerance=0.01):
    """
    Validate that weights are properly configured.
    
    Args:
        weights: List [w_S, w_C, w_D, r_M]
        tolerance: Acceptable deviation from sum=1.0 for first three weights
        
    Returns:
        bool: True if valid, raises ValueError otherwise
    """
    w_S, w_C, w_D, r_M = weights
    
    # Check all weights are non-negative
    if any(w < 0 for w in weights):
        raise ValueError(f"All weights must be non-negative: {weights}")
    
    # Check service weights sum to approximately 1
    service_sum = w_S + w_C + w_D
    if abs(service_sum - 1.0) > tolerance:
        raise ValueError(f"Service weights (w_S + w_C + w_D) should sum to 1.0, got {service_sum}: {weights}")
    
    # Check r_M is 0 for Phase 1
    if r_M != 0.0:
        raise ValueError(f"Phase 1 requires r_M = 0, got {r_M}: {weights}")
    
    return True


def get_all_configurations():
    """
    Get all weight configurations with validation.
    
    Returns:
        dict: Dictionary of configuration_name -> weights
    """
    validated_configs = {}
    
    for name, weights in WEIGHT_CONFIGURATIONS.items():
        try:
            validate_weights(weights)
            validated_configs[name] = weights
        except ValueError as e:
            print(f"WARNING: Configuration '{name}' is invalid: {e}")
    
    return validated_configs


def print_configurations():
    """
    Pretty print all configurations for review.
    """
    print("="*80)
    print("PHASE 1: BASECASE WEIGHT CONFIGURATIONS")
    print("="*80)
    print(f"\nTotal configurations: {len(WEIGHT_CONFIGURATIONS)}\n")
    
    configs = get_all_configurations()
    
    for name, weights in configs.items():
        w_S, w_C, w_D, r_M = weights
        print(f"{name:25s}: w_S={w_S:.2f}, w_C={w_C:.2f}, w_D={w_D:.2f}, r_M={r_M:.2f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Print all configurations when run directly
    print_configurations()
    
    # Validate all
    configs = get_all_configurations()
    print(f"\nAll {len(configs)} configurations validated successfully!")

