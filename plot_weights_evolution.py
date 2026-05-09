import pandas as pd
import matplotlib.pyplot as plt
import os

files = [
    'models/results/Maintenance_full_test_alpha_0.1_20260430_163816/vfa_Maintenance_full_test_seed1000_weights_evolution.csv',
    'models/results/Maintenance_full_test_alpha_0.1_20260430_165038/vfa_Maintenance_full_test_seed1000_weights_evolution.csv'
]

features = ['trailer_cannibalization', 'global_onsite_backlog', 'demand_weighted_depot_backlog',
               'demand_weighted_onsite_backlog', 'fleet_broken_fraction', 'depot_in_repair_fraction',
               'depot_idle_fraction', 'recoverable_starvation', 'maintenance_urgency', 'rush_hour_onsite_backlog']

for file in files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        plt.figure(figsize=(12, 8))
        for feat in features:
            if feat in df.columns:
                plt.plot(df['episode'], df[feat], label=feat)
        plt.title(f"Weight Evolution: {file.split('/')[-2]}")
        plt.xlabel("Episode")
        plt.ylabel("Weight Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        out_name = file.replace('.csv', '.png')
        plt.savefig(out_name)
        print(f"Saved plot to {out_name}")
    else:
        print(f"File not found: {file}")
