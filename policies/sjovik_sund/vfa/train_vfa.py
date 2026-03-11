#!/usr/bin/env python3
"""
Example Training Script for VFA Agent

This script demonstrates how to:
1. Train a VFA agent from scratch
2. Monitor learning progress
3. Save the trained model
4. Test the learned policy

Usage:
    python train_vfa.py --mode train --seeds 42 43 44
    python train_vfa.py --mode test --model models/vfa_trained.pkl
"""

import sys
from pathlib import Path
import argparse
import time
from datetime import datetime

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(WORKSPACE_ROOT))

from policies.sjovik_sund.run_simulation import run_simulation, test_seeds, SimulationConfig
from policies.sjovik_sund.vfa import (
    VFAPolicy, VFAAgent, LearningParameters,
    VFAFeatures, DemandForecaster
)
from settings import ENABLE_COMPONENT_FAILURES


def train_vfa_agent(seeds, duration_hours=24*5, num_vehicles=1, 
                    instance_name="TD_W34_old", save_dir="models"):
    """
    Train VFA agent over multiple simulation runs.
    
    Args:
        seeds: List of random seeds for training
        duration_hours: Simulation duration per seed
        num_vehicles: Number of vehicles
        instance_name: Instance to train on
        save_dir: Directory to save models
    """
    print(f"\n{'='*80}")
    print(f"TRAINING VFA AGENT")
    print(f"{'='*80}")
    print(f"  Seeds: {seeds}")
    print(f"  Duration: {duration_hours} hours")
    print(f"  Vehicles: {num_vehicles}")
    print(f"  Instance: {instance_name}")
    print(f"  Component failures: {ENABLE_COMPONENT_FAILURES}")
    print(f"{'='*80}\n")
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Configure learning parameters
    learning_params = LearningParameters(
        initial_learning_rate=0.01,
        learning_rate_decay=0.9999,
        min_learning_rate=0.001,
        initial_epsilon=0.3,  # 30% exploration initially
        epsilon_decay=0.9995,
        min_epsilon=0.05,
        l2_regularization=0.001,
        failed_rental_cost=10.0,
        failed_return_cost=5.0,
        discount_factor=0.95
    )
    
    # Create VFA components
    forecaster = DemandForecaster()
    features = VFAFeatures(forecaster)
    agent = VFAAgent(features, learning_params, seed=seeds[0])
    
    # Create policy with learning enabled
    policy = VFAPolicy(
        vfa_agent=agent,
        learning_mode=True,
        maintenance_enabled=True,
        seed=seeds[0]
    )
    
    # Create simulation config
    config = SimulationConfig()
    
    # Train over multiple seeds
    start_time = time.time()
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*80}")
        print(f"Training Episode {i+1}/{len(seeds)} (Seed: {seed})")
        print(f"{'='*80}\n")
        
        # Run simulation (agent learns during execution)
        simulator = run_simulation(
            seed=seed,
            policy=policy,
            duration=duration_hours,
            num_vehicles=num_vehicles,
            instance_name=instance_name,
            config=config
        )
        
        # Print learning statistics
        policy.vfa_agent.print_learning_status()
        
        # Save checkpoint after each episode
        checkpoint_path = save_path / f"vfa_checkpoint_seed{seed}.pkl"
        policy.save_agent(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = save_path / f"vfa_trained_{timestamp}.pkl"
    policy.save_agent(final_model_path)
    
    training_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"  Total training time: {training_time:.2f}s ({training_time/60:.2f} min)")
    print(f"  Total iterations: {agent.iteration}")
    print(f"  Final learning rate: {agent.current_learning_rate:.6f}")
    print(f"  Final epsilon: {agent.current_epsilon:.4f}")
    print(f"  Final θ norm: {agent.get_statistics_summary()['theta_norm']:.4f}")
    print(f"  Model saved to: {final_model_path}")
    print(f"{'='*80}\n")
    
    return policy, final_model_path


def test_vfa_agent(model_path, seeds, duration_hours=24*7, num_vehicles=1,
                   instance_name="TD_W34_old", results_file="vfa_test_results.csv"):
    """
    Test a trained VFA agent (no learning).
    
    Args:
        model_path: Path to trained model
        seeds: List of random seeds for testing
        duration_hours: Simulation duration per seed
        num_vehicles: Number of vehicles
        instance_name: Instance to test on
        results_file: CSV file to write results
    """
    print(f"\n{'='*80}")
    print(f"TESTING VFA AGENT")
    print(f"{'='*80}")
    print(f"  Model: {model_path}")
    print(f"  Seeds: {seeds}")
    print(f"  Duration: {duration_hours} hours")
    print(f"  Vehicles: {num_vehicles}")
    print(f"  Instance: {instance_name}")
    print(f"{'='*80}\n")
    
    # Load trained policy
    policy = VFAPolicy(learning_mode=False, maintenance_enabled=True)
    policy.load_agent(Path(model_path))
    
    # Run test simulations
    config = SimulationConfig()
    
    test_seeds(
        list_of_seeds=seeds,
        policy=policy,
        filename=results_file,
        num_vehicles=num_vehicles,
        duration=duration_hours,
        use_multiprocessing=False,
        instance_name=instance_name,
        config=config
    )
    
    print(f"\n{'='*80}")
    print(f"TESTING COMPLETE")
    print(f"{'='*80}")
    print(f"  Results saved to: {results_file}")
    print(f"{'='*80}\n")


def compare_with_baseline(vfa_seeds, baseline_seeds, duration_hours=24*5,
                         num_vehicles=1, instance_name="TD_W34_old"):
    """
    Compare VFA policy with a baseline policy.
    
    Args:
        vfa_seeds: Seeds for VFA policy
        baseline_seeds: Seeds for baseline policy
        duration_hours: Simulation duration
        num_vehicles: Number of vehicles
        instance_name: Instance name
    """
    print(f"\n{'='*80}")
    print(f"COMPARING VFA WITH BASELINE")
    print(f"{'='*80}\n")
    
    # Train VFA policy
    vfa_policy, model_path = train_vfa_agent(
        seeds=vfa_seeds[:2],  # Use first 2 seeds for training
        duration_hours=duration_hours,
        num_vehicles=num_vehicles,
        instance_name=instance_name
    )
    
    # Test VFA policy
    test_vfa_agent(
        model_path=model_path,
        seeds=vfa_seeds[2:],  # Use remaining seeds for testing
        duration_hours=duration_hours,
        num_vehicles=num_vehicles,
        instance_name=instance_name,
        results_file="vfa_policy_results.csv"
    )
    
    # Test baseline (using greedy policy from existing codebase)
    # You would import your baseline policy here
    # For example:
    # from policies.greedy_policy import GreedyPolicy
    # baseline_policy = GreedyPolicy()
    # test_seeds(baseline_seeds, baseline_policy, "baseline_results.csv", ...)
    
    print(f"\n{'='*80}")
    print(f"COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"  VFA results: vfa_policy_results.csv")
    print(f"  Baseline results: baseline_results.csv")
    print(f"  Compare metrics to evaluate improvement")
    print(f"{'='*80}\n")


def analyze_learned_features(model_path):
    """
    Analyze the learned feature weights.
    
    Args:
        model_path: Path to trained model
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING LEARNED FEATURES")
    print(f"{'='*80}\n")
    
    # Load model
    policy = VFAPolicy(learning_mode=False)
    policy.load_agent(Path(model_path))
    
    # Get feature names and weights
    feature_names = policy.vfa_agent.features.get_feature_names()
    theta = policy.vfa_agent.theta
    
    print(f"{'Feature':<30} {'Weight':<15} {'|Weight|':<15}")
    print(f"{'-'*60}")
    
    for name, weight in zip(feature_names, theta):
        print(f"{name:<30} {weight:>14.6f} {abs(weight):>14.6f}")
    
    print(f"\n{'='*80}")
    print(f"Feature Interpretation:")
    print(f"{'='*80}")
    print(f"  Positive weights → penalize feature (increase cost)")
    print(f"  Negative weights → reward feature (decrease cost)")
    print(f"  Large |weight| → strong influence on decisions")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train and test VFA agent for DSJBRMP"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test', 'compare', 'analyze'],
        required=True,
        help='Operation mode'
    )
    
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[42, 43, 44, 45, 46],
        help='Random seeds for training/testing'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Path to trained model (for test/analyze mode)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=24*5,
        help='Simulation duration in hours (default: 5 days)'
    )
    
    parser.add_argument(
        '--vehicles',
        type=int,
        default=1,
        help='Number of vehicles (default: 1)'
    )
    
    parser.add_argument(
        '--instance',
        type=str,
        default='TD_W34_old',
        help='Instance name (default: TD_W34_old)'
    )
    
    parser.add_argument(
        '--save_dir',
        type=str,
        default='models',
        help='Directory to save models (default: models)'
    )
    
    args = parser.parse_args()
    
    # Execute based on mode
    if args.mode == 'train':
        train_vfa_agent(
            seeds=args.seeds,
            duration_hours=args.duration,
            num_vehicles=args.vehicles,
            instance_name=args.instance,
            save_dir=args.save_dir
        )
    
    elif args.mode == 'test':
        if not args.model:
            print("Error: --model required for test mode")
            return
        
        test_vfa_agent(
            model_path=args.model,
            seeds=args.seeds,
            duration_hours=args.duration,
            num_vehicles=args.vehicles,
            instance_name=args.instance
        )
    
    elif args.mode == 'compare':
        compare_with_baseline(
            vfa_seeds=args.seeds,
            baseline_seeds=args.seeds,
            duration_hours=args.duration,
            num_vehicles=args.vehicles,
            instance_name=args.instance
        )
    
    elif args.mode == 'analyze':
        if not args.model:
            print("Error: --model required for analyze mode")
            return
        
        analyze_learned_features(args.model)


if __name__ == "__main__":
    main()
