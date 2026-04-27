#!/bin/bash
BASE_ARGS="--episodes 350 --seed 1000 --alphas 0.1 --output_dir final_ablation_350ep"

echo "[$(date)] Starting Full Ablation Study..."

experiments=(
    "Squared_Temporal"
    "Starvation_focused"
    "Short_term_only"
    "Exponential_Temporal"
    "Count_Temporal"
    "Net_Demand_Temporal"
    "Severity_Temporal"
    "Combined_Temporal"
)

for exp in "${experiments[@]}"; do
    echo "Launching $exp..."
    python policies/sjovik_sund/ablation_study/run_ablation_study.py $BASE_ARGS --experiments "$exp" > "output/${exp}.log" 2>&1
done

echo "[$(date)] All ablation experiments are complete!"
