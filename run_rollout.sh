#!/bin/bash
mkdir -p output
echo "[$(date)] Starting hybrid rollout evaluation..."

python policies/sjovik_sund/vfa/evaluate_hybrid_rollout.py \
  --train_seeds 5000 6000 \
  --train_alphas 0.1 0.05 \
  --last_n 10 \
  --base_dir models/final_ablation_300ep \
  --run_only Short_term_only Exponential_Temporal Squared_Temporal Combined_Temporal Starvation_focused \
  --episodes 5 \
  --seed 42 \
  --duration 336 \
  --scenarios 5 \
  --lookahead 60 \
  --instance TD_W34_old \
  --vehicles 1

echo "[$(date)] Evaluation complete!"
