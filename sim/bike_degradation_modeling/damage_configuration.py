"""
Weibull distribution parameters for component failures
Derived from maintenance data analysis (2023-2024)
"""

DAMAGE_CATEGORIES = {
    'Body & Accessories': {
        'shape': 1.46,
        'scale': 802.52,
        'median_km': 624.34,
        'mttf_km': 727.00
    },
    'Drivetrain': {
        'shape': 1.25,
        'scale': 899.84,
        'median_km': 671.42,
        'mttf_km': 837.85
    },
    'Steering & Chassis': {
        'shape': 1.35,
        'scale': 890.44,
        'median_km': 678.64,
        'mttf_km': 816.60
    },
    'Wheels & Tires': {
        'shape': 1.43,
        'scale': 904.85,
        'median_km': 700.79,
        'mttf_km': 821.72
    },
    'Braking System': {
        'shape': 1.56,
        'scale': 1160.34,
        'median_km': 916.99,
        'mttf_km': 1043.10
    },
    'Electronics': {
        'shape': 1.26,
        'scale': 1283.28,
        'median_km': 958.64,
        'mttf_km': 1193.79
    },
    'Locking System': {
        'shape': 1.13,
        'scale': 1763.17,
        'median_km': 1273.36,
        'mttf_km': 1688.59
    }
}