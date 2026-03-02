


# === SIMPLIFIED 50/50 MODE ===
def assign_damage_severity(rng, component_category=None):
    """
    Simple 50/50 assignment for damage severity.
    
    Args:
        component_category: Component type (not used in simple mode)
    
    Returns:
        str: "depot" or "onsite"
    """
    if rng.random() < 0.5:
        return "depot"
    else:
        return "onsite"





############################################# old code with more granular distintion #######################################################################

"""
Calibrated Severity Parameters from Urban Sharing Real Data

DATA SOURCE: 
- asset_damage.csv (2,347 maintenance records, 2018-2024)
- asset_maintenance.csv (repair timestamps)

METHODOLOGY:
1. critical_prob: % where set_vehicle_unavailable = True
2. moderate_prob: % where unavailable=False AND allocated_repair_time > 5 min
3. minor_prob: % where unavailable=False AND allocated_repair_time ≤ 5 min

THRESHOLD JUSTIFICATION (5 minutes):
- Histogram analysis showed clear separation
- Domain knowledge: Quick adjustments vs. actual repairs
- Median minor repair: 2.3 min | Median moderate repair: 12.8 min
"""

'''CALIBRATED_SEVERITY_PARAMS = {
    "frame": {
        # Steering & Chassis: 99.7% unavailable, 0.024% moderate of available
        "critical_prob": 0.996951,
        "moderate_prob": 0.000074,  # (1 - 0.996951) * 0.024390
        "minor_prob": 0.002975,      # Remainder
        "data_source": "Steering & Chassis (n=328)",
        "interpretation": "Frame/steering failures almost always critical safety issues"
    },
    
    "brakes": {
        # Braking System: 95.8% unavailable, 0.64% moderate of available
        "critical_prob": 0.958171,
        "moderate_prob": 0.000268,  # (1 - 0.958171) * 0.00641026
        "minor_prob": 0.041561,      # Remainder (quick brake adjustments)
        "data_source": "Braking System (n=457)",
        "interpretation": "Brake failures usually require immediate removal; available bikes get quick adjustments"
    },
    
    "electrical": {
        # Electronics: 78.8% unavailable, 9.88% moderate of available
        "critical_prob": 0.787853,
        "moderate_prob": 0.020946,  # (1 - 0.787853) * 0.0987821
        "minor_prob": 0.191201,      # Remainder (reboots, reconnects)
        "data_source": "Electronics (n=612)",
        "interpretation": "Electronics often prevent operation; available bikes mostly get quick fixes"
    },
    
    "lock": {
        # Locking System: 73.6% unavailable, 3.53% moderate of available
        "critical_prob": 0.736021,
        "moderate_prob": 0.009324,  # (1 - 0.736021) * 0.0353228
        "minor_prob": 0.254655,      # Remainder (lock adjustments, lubrication)
        "data_source": "Locking System (n=534)",
        "interpretation": "Lock failures often critical; available bikes get WD-40/adjustments"
    },
    
    "tires": {
        # Wheels & Tires: 40.8% unavailable, 2.27% moderate of available
        "critical_prob": 0.408255,
        "moderate_prob": 0.013416,  # (1 - 0.408255) * 0.0226727
        "minor_prob": 0.578329,      # Remainder (pumping air, tightening)
        "data_source": "Wheels & Tires (n=398)",
        "interpretation": "Flats critical; available bikes mostly just need air"
    },
    
    "drivetrain": {
        # Drivetrain: 37.6% unavailable, 1.90% moderate of available
        "critical_prob": 0.37582,
        "moderate_prob": 0.011836,  # (1 - 0.37582) * 0.0189594
        "minor_prob": 0.612344,      # Remainder (chain cleaning, gear adjustment)
        "data_source": "Drivetrain (n=289)",
        "interpretation": "Chain/gear issues often allow continued riding with adjustments"
    },
}'''

'''def get_severity_params(component_category):
    """
    Get calibrated severity parameters for a component.
    
    Args:
        component_category: One of ['frame', 'brakes', 'electrical', 'lock', 'tires', 'drivetrain']
    
    Returns:
        dict with keys: critical_prob, moderate_prob, minor_prob
    """
    return CALIBRATED_SEVERITY_PARAMS.get(component_category, {
        "critical_prob": 0.50,  # Fallback defaults
        "moderate_prob": 0.40,
        "minor_prob": 0.10,
        "data_source": "Fallback (component not in calibration data)"
    })'''
