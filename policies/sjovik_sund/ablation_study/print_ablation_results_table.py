import os
import csv
from collections import defaultdict

RUN_DIR = os.path.join(
    os.path.dirname(__file__), "../../../run_logs/run_20260423_233511"
)
SEEDS = {42, 43, 44, 45}

# exp_alpha -> list of (service_level, starvations, congestions)
data = defaultdict(list)

for entry in os.scandir(RUN_DIR):
    if not entry.is_dir():
        continue
    results_path = os.path.join(entry.path, "results.csv")
    if not os.path.exists(results_path):
        continue
    with open(results_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["seed"]) not in SEEDS:
                continue
            key = (row["exp_name"], float(row["alpha"]))
            data[key].append(
                (float(row["service_level"]), int(row["starvations"]), int(row["congestions"]))
            )

# Sort: by exp_name then alpha
sorted_keys = sorted(data.keys(), key=lambda k: (k[0], k[1]))

col_w = [30, 6, 14, 14, 14]
header = (
    f"{'Experiment':<{col_w[0]}} {'Alpha':>{col_w[1]}} "
    f"{'Service Level':>{col_w[2]}} {'Starvations':>{col_w[3]}} {'Congestions':>{col_w[4]}}"
)
sep = "-" * len(header)

print(sep)
print(header)
print(sep)

for exp_name, alpha in sorted_keys:
    rows = data[(exp_name, alpha)]
    n = len(rows)
    avg_sl = sum(r[0] for r in rows) / n
    avg_st = sum(r[1] for r in rows) / n
    avg_co = sum(r[2] for r in rows) / n
    label = f"{exp_name}-alpha_{alpha}"
    print(
        f"{label:<{col_w[0]}} {alpha:>{col_w[1]}.2f} "
        f"{avg_sl:>{col_w[2]}.4f} {avg_st:>{col_w[3]}.1f} {avg_co:>{col_w[4]}.1f}"
    )

print(sep)
print(f"(averages over seeds {sorted(SEEDS)}, n={len(rows)} per row)")
