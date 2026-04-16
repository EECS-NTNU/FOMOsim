import subprocess
import sys
import os

def main():
    # Define the parameters for your run
    seeds = ["1000"]
    episodes = "200"
    
    # 1. Run the Training (Ablation Study)
    print("=" * 80)
    print("STEP 1: RUNNING TRAINING (ABLATION STUDY)")
    print("=" * 80)
    
    train_command = [
        sys.executable,  # Uses the current active python environment
        "policies/sjovik_sund/ablation_study/run_ablation_study.py",
        "--seeds", *seeds,
        "--episodes", episodes
    ]
    
    try:
        # run() waits for the process to finish
        subprocess.run(train_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Training failed with exit code {e.returncode}. Stopping pipeline.")
        sys.exit(1)


    # 2. Run the Evaluation
    print("\n" + "=" * 80)
    print("STEP 2: RUNNING EVALUATION (HYBRID ROLLOUT)")
    print("=" * 80)
    
    # Because evaluate_hybrid_rollout has been updated to point to SGDMINIBATCH_2,
    # running it without arguments will automatically batch evaluate everything found there.
    eval_command = [
        sys.executable,
        "policies/sjovik_sund/vfa/evaluate_hybrid_rollout.py",
    ]
    
    try:
        subprocess.run(eval_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Evaluation failed with exit code {e.returncode}.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE! Check simulation_results/csv/ for outputs.")
    print("=" * 80)

if __name__ == "__main__":
    main()
