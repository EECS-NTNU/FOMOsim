#!/usr/bin/env python3
"""
diagnose_linear_vfa.py — Linear VFA diagnostic script.

Runs a short instrumented simulation and writes:
  diagnostics/feature_stats.csv
  diagnostics/theta_report.csv
  diagnostics/feature_target_correlations.csv
  diagnostics/feature_feature_correlations.csv
  diagnostics/td_update_samples.csv
  diagnostics/diagnostic_summary.txt

Usage:
  python diagnose_linear_vfa.py
  python diagnose_linear_vfa.py --model path/to/trained.pkl
  python diagnose_linear_vfa.py --episodes 5
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from io import StringIO

WORKSPACE_ROOT = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from settings import ENABLE_COMPONENT_FAILURES
from policies.sjovik_sund.vfa.vfa_features import get_feature_names, extract as _extract_phi
from policies.sjovik_sund.vfa.LinearVFAPolicy import LinearVFAPolicy, EpisodeTrainingPolicy
from policies.sjovik_sund.mdp.reward import RewardConfig, RewardCalculator
from policies.sjovik_sund.run_simulation_ingvild import run_simulation, SimulationConfig
from policies.greedy_policy import GreedyPolicy
from helpers import timeInMinutes

OUT = Path("diagnostics")
OUT.mkdir(exist_ok=True)

SUMMARY = StringIO()


def _log(msg: str) -> None:
    print(msg)
    SUMMARY.write(msg + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Feature-to-weight mapping
# ─────────────────────────────────────────────────────────────────────────────

def check_feature_mapping(policy: LinearVFAPolicy) -> None:
    _log("\n" + "=" * 70)
    _log("SECTION 1: FEATURE-TO-WEIGHT MAPPING")
    _log("=" * 70)

    all_names = policy.ALL_FEATURE_NAMES
    active_names = policy.FEATURE_NAMES
    mask = policy._feature_mask

    _log(f"  ALL features (canonical): {len(all_names)}")
    _log(f"  Active features:          {len(active_names)}")
    _log(f"  Theta length:             {len(policy.theta)}")
    _log(f"  Mask True count:          {int(np.sum(mask))}")

    # Assert invariants
    ok = True
    if len(policy.theta) != len(active_names):
        _log(f"  [BUG] len(theta)={len(policy.theta)} != len(FEATURE_NAMES)={len(active_names)}")
        ok = False
    if int(np.sum(mask)) != len(active_names):
        _log(f"  [BUG] mask True count != len(FEATURE_NAMES)")
        ok = False

    # Check canonical order preserved
    derived_order = [n for n in all_names if n in active_names]
    if derived_order != active_names:
        _log("\n  [BUG] FEATURE ORDER MISMATCH:")
        _log(f"    Expected canonical order: {derived_order}")
        _log(f"    Actual FEATURE_NAMES:     {active_names}")
        ok = False

    if ok:
        _log("  [OK] Feature-weight mapping is consistent.")

    # Print index table
    rows = []
    for i, (name, w) in enumerate(zip(active_names, policy.theta)):
        sign_ok = "?" if w == 0.0 else ("NEG" if w < 0 else "POS")
        rows.append({"idx": i, "feature": name, "theta": round(w, 6), "sign": sign_ok})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "theta_report.csv", index=False)
    _log(f"\n  Theta report saved -> {OUT}/theta_report.csv")
    _log("\n  idx  feature                              theta     sign")
    _log("  " + "-" * 56)
    for r in rows:
        _log(f"  {r['idx']:<4} {r['feature']:<38} {r['theta']:>+9.4f}  {r['sign']}")

    # Flag semantically wrong-sign thetas
    expected_negative = [
        "rebalancing_imbalance", "squared_starvation_penalty",
        "exponential_starvation_penalty", "starvation_severity_max",
        "starvation_count", "fleet_broken_fraction", "depot_in_repair_fraction",
        "trailer_cannibalization", "global_onsite_backlog",
        "demand_weighted_depot_backlog", "demand_weighted_onsite_backlog",
        "recoverable_starvation", "maintenance_urgency", "rush_hour_onsite_backlog",
        "gross_starvation_risk", "net_starvation_shortfall",
    ]
    # depot_idle_fraction (MP7) = fixed_queue / fleet → bikes READY to redeploy.
    # Positive theta may actually be rational (more fixed bikes = better capacity).
    # Flag as "AMBIGUOUS" rather than wrong.
    ambiguous = {"depot_idle_fraction", "squared_congestion_penalty",
                 "exponential_congestion_penalty", "congestion_severity_max",
                 "congestion_count", "gross_congestion_risk", "net_congestion_shortfall"}

    _log("\n  ── SIGN ANALYSIS ──────────────────────────────────────────────")
    wrong = []
    for r in rows:
        n = r["feature"]
        w = r["theta"]
        if n in expected_negative and w > 0.01:
            _log(f"  [WRONG SIGN] {n}: theta={w:+.4f}  (expected negative)")
            wrong.append(n)
        elif n in ambiguous:
            _log(f"  [AMBIGUOUS]  {n}: theta={w:+.4f}  (semantics depend on context)")

    if not wrong:
        _log("  No obviously wrong-sign thetas detected.")
    else:
        _log(f"\n  {len(wrong)} wrong-sign feature(s) detected.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Reward sign and scale
# ─────────────────────────────────────────────────────────────────────────────

def check_reward_convention(policy: LinearVFAPolicy) -> None:
    _log("\n" + "=" * 70)
    _log("SECTION 2: REWARD SIGN & SCALE CONVENTION")
    _log("=" * 70)

    rc = policy.reward_calc
    cfg = rc.config
    sf = rc._scale_factor

    _log(f"  weight_starvation:       {cfg.weight_starvation}")
    _log(f"  weight_congestion:       {cfg.weight_congestion}")
    _log(f"  weight_fleet_degradation:{cfg.weight_fleet_degradation}")
    _log(f"  _scale_factor:           {sf}  (= 1 - gamma = {1 - rc.gamma:.4f})")

    if abs(sf - 1.0) > 1e-6 and abs(sf) < 0.1:
        _log(f"\n  [WARNING] scale_factor={sf:.4f} crushes reward signal to {sf*100:.1f}%.")
        _log(f"  Effective starvation weight per event: {cfg.weight_starvation * sf:.4f}")
        _log(f"  Effective congestion weight per event: {cfg.weight_congestion * sf:.4f}")
        _log(f"  Effective fleet penalty per ratio unit: {cfg.weight_fleet_degradation * sf:.4f}")
        _log(f"\n  DIAGNOSIS: With scale_factor≈0.01, rewards are near-zero. TD errors are")
        _log(f"  dominated by bootstrapped V estimates, not actual reward signal. This")
        _log(f"  causes theta to drift based on value bootstrap noise, not true penalty.")
        _log(f"  FIX: Set _scale_factor = 1.0 in RewardCalculator.__init__.")

    if cfg.weight_starvation > 0:
        _log(f"\n  [BUG] weight_starvation is POSITIVE ({cfg.weight_starvation}). Should be negative.")
    if cfg.weight_congestion > 0:
        _log(f"\n  [BUG] weight_congestion is POSITIVE ({cfg.weight_congestion}). Should be negative.")
    if cfg.weight_fleet_degradation > 0:
        _log(f"\n  [BUG] weight_fleet_degradation is POSITIVE. Should be negative.")

    _log(f"\n  Both compute_step_reward AND compute_fleet_penalty multiply by")
    _log(f"  _scale_factor. Fleet penalty fires every decision step (not delta-based).")
    _log(f"  Over 14 learning days with ~700 decisions, total fleet_penalty contribution")
    _log(f"  ≈ {cfg.weight_fleet_degradation * sf * 700 * 0.1:.2f} (assuming 10% broken fraction).")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Feature semantics audit
# ─────────────────────────────────────────────────────────────────────────────

def check_feature_semantics() -> None:
    _log("\n" + "=" * 70)
    _log("SECTION 3: FEATURE SEMANTICS AUDIT (human-readable)")
    _log("=" * 70)

    issues = [
        ("depot_idle_fraction (MP7)",
         "Counts bikes in depot FIXED QUEUE — bikes that are REPAIRED and waiting pickup.",
         "Comment says '0=good, positive=bad' but POSITIVE means more bikes ready to redeploy.",
         "HIGH depot_idle → agent picks up fixed bikes → future starvations reduced → POSITIVE theta is RATIONAL.",
         "ACTION: Rename to 'depot_fixed_queue_fraction' and document as opportunity signal."),

        ("depot_in_repair_fraction (MP6)",
         "Counts bikes CURRENTLY IN REPAIR at depot (active repair pipeline).",
         "High value means the repair system is ACTIVE — bikes being fixed RIGHT NOW.",
         "HIGH in_repair → more bikes return to service soon → POSITIVE theta may be rational.",
         "ACTION: Distinguish from 'bikes awaiting repair' vs 'bikes being fixed'. Separate features."),

        ("recoverable_starvation (MP8)",
         "Onsite broken bikes at stations that are ALSO starving.",
         "This represents REACHABLE REPAIR OPPORTUNITY: go there, fix bikes, relieve starvation.",
         "HIGH recoverable_starvation → vehicle can take high-value repair action → POSITIVE theta rational.",
         "ACTION: Accept positive theta OR separate into 'starvation_at_repair_stations' (bad) + "
         "'repair_capacity_at_starving_stations' (opportunity)."),

        ("maintenance_urgency (MP9)",
         "= global_onsite_backlog × starvation_count (product of two bad signals).",
         "Amplifies both signals, but also encodes 'there are repair opportunities at starving stations'.",
         "Theta sign depends on whether VFA sees this as 'bad double signal' or 'action opportunity'.",
         "ACTION: Consider removing this interaction term; it's redundant with MP2 + CIM8."),
    ]

    for name, what, problem, diagnosis, action in issues:
        _log(f"\n  {name}")
        _log(f"    WHAT: {what}")
        _log(f"    PROBLEM: {problem}")
        _log(f"    DIAGNOSIS: {diagnosis}")
        _log(f"    {action}")


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented policy for data collection
# ─────────────────────────────────────────────────────────────────────────────

class _InstrumentedPolicy(LinearVFAPolicy):
    """Wraps LinearVFAPolicy to record TD transition data."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transitions: list = []  # (phi_cur, reward, phi_next, td_error, discount)

    def td_update(self, reward: float, phi_next: np.ndarray, elapsed_minutes: float) -> float:
        if self._prev_phi is None:
            return 0.0

        v_cur = self.value(self._prev_phi)
        v_next = self.value(phi_next)
        discount = self.gamma ** (elapsed_minutes / 60.0)
        td_error = reward + discount * v_next - v_cur

        self._transitions.append({
            "phi_cur": self._prev_phi.copy(),
            "reward": reward,
            "v_cur": v_cur,
            "v_next": v_next,
            "target": reward + discount * v_next,
            "td_error": td_error,
            "discount": discount,
            "phi_next": phi_next.copy(),
        })

        # Still do the actual update
        return super().td_update(reward, phi_next, elapsed_minutes)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4–7: Run simulation and collect data
# ─────────────────────────────────────────────────────────────────────────────

def run_instrumented(num_episodes: int, instance: str, model_path: Path | None) -> None:
    _log("\n" + "=" * 70)
    _log("SECTION 4–7: INSTRUMENTED SIMULATION")
    _log("=" * 70)

    EPISODE_DAYS = 14
    WARMUP_DAYS  = 4
    START_HOUR   = 0

    reward_config = RewardConfig()
    _log(f"\n  Running {num_episodes} episode(s), instance={instance}")
    _log(f"  scale_factor={reward_config.weight_starvation * (1 - 0.99):.4f} (effective per starvation)")

    policy = _InstrumentedPolicy(
        alpha=0.05,
        gamma=0.99,
        learning_mode=True,
        seed=42,
        maintenance_enabled=ENABLE_COMPONENT_FAILURES,
        logistics_enabled=False,
        reward_calculator=RewardCalculator(config=reward_config, gamma=0.99),
    )

    if model_path is not None and model_path.exists():
        saved = LinearVFAPolicy.load(model_path)
        policy.theta = saved.theta.copy()
        _log(f"  Loaded theta from {model_path}")

    greedy_policy = GreedyPolicy()
    sim_start_min   = timeInMinutes(hours=START_HOUR)
    warmup_end_time = sim_start_min + WARMUP_DAYS * 24 * 60

    all_transitions: list = []

    for ep in range(num_episodes):
        policy._transitions.clear()
        episode_policy = EpisodeTrainingPolicy(
            vfa_policy      = policy,
            warmup_policy   = greedy_policy,
            warmup_end_time = warmup_end_time,
        )
        simulator = run_simulation(
            seed          = 1000 + ep,
            policy        = episode_policy,
            duration      = 24 * EPISODE_DAYS,
            num_vehicles  = 1,
            instance_name = instance,
            config        = SimulationConfig(),
        )
        all_transitions.extend(policy._transitions)
        _log(f"  Ep {ep+1}: {len(policy._transitions)} TD transitions collected")

    if not all_transitions:
        _log("\n  [ERROR] No transitions collected — learning phase may not have been reached.")
        return

    feature_names = policy.FEATURE_NAMES
    n_feat = len(feature_names)

    # Build DataFrames
    phi_cur_mat  = np.array([t["phi_cur"]  for t in all_transitions])  # (T, F)
    phi_next_mat = np.array([t["phi_next"] for t in all_transitions])
    rewards      = np.array([t["reward"]   for t in all_transitions])
    targets      = np.array([t["target"]   for t in all_transitions])
    td_errors    = np.array([t["td_error"] for t in all_transitions])
    v_curs       = np.array([t["v_cur"]    for t in all_transitions])
    v_nexts      = np.array([t["v_next"]   for t in all_transitions])

    # ── Section 4: Feature scale report ──────────────────────────────────────
    _log("\n" + "─" * 70)
    _log("SECTION 4: FEATURE SCALE REPORT")
    _log("─" * 70)

    stats_rows = []
    for i, name in enumerate(feature_names):
        col = phi_cur_mat[:, i]
        stats_rows.append({
            "feature":      name,
            "mean":         round(float(np.mean(col)), 5),
            "std":          round(float(np.std(col)),  5),
            "min":          round(float(np.min(col)),  5),
            "p5":           round(float(np.percentile(col, 5)),  5),
            "p50":          round(float(np.median(col)), 5),
            "p95":          round(float(np.percentile(col, 95)), 5),
            "max":          round(float(np.max(col)),  5),
            "frac_zero":    round(float(np.mean(col == 0.0)), 4),
            "frac_gt1":     round(float(np.mean(col > 1.0)),  4),
            "theta":        round(float(policy.theta[i]), 6) if i < len(policy.theta) else None,
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(OUT / "feature_stats.csv", index=False)
    _log(f"  Saved -> {OUT}/feature_stats.csv")

    _log(f"\n  {'Feature':<38} {'mean':>7} {'std':>7} {'max':>7} {'frac>1':>7}  ALERT")
    for r in stats_rows:
        alerts = []
        if r["std"] < 1e-4:
            alerts.append("NEAR-CONSTANT")
        if r["max"] > 5.0:
            alerts.append("LARGE-SCALE")
        if r["frac_zero"] > 0.95:
            alerts.append("MOSTLY-ZERO")
        _log(f"  {r['feature']:<38} {r['mean']:>7.3f} {r['std']:>7.3f} {r['max']:>7.3f} {r['frac_gt1']:>7.3f}  {', '.join(alerts)}")

    # ── Section 5: Feature-target correlation ─────────────────────────────────
    _log("\n" + "─" * 70)
    _log("SECTION 5: FEATURE-TARGET CORRELATIONS")
    _log("─" * 70)
    _log("  (Positive corr with target = feature appears in high-value transitions)")
    _log("  (For 'bad' features: positive corr with target → opportunity-state effect)")

    corr_rows = []
    for i, name in enumerate(feature_names):
        col = phi_cur_mat[:, i]
        c_target = float(np.corrcoef(col, targets)[0, 1]) if np.std(col) > 1e-8 and np.std(targets) > 1e-8 else 0.0
        c_reward = float(np.corrcoef(col, rewards)[0, 1]) if np.std(col) > 1e-8 and np.std(rewards) > 1e-8 else 0.0
        c_td     = float(np.corrcoef(col, td_errors)[0, 1]) if np.std(col) > 1e-8 and np.std(td_errors) > 1e-8 else 0.0
        corr_rows.append({
            "feature": name, "theta": round(policy.theta[i], 5),
            "corr_with_target": round(c_target, 4),
            "corr_with_reward": round(c_reward, 4),
            "corr_with_td_error": round(c_td, 4),
        })

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT / "feature_target_correlations.csv", index=False)
    _log(f"  Saved -> {OUT}/feature_target_correlations.csv")

    _log(f"\n  {'Feature':<38} {'theta':>8}  {'r_target':>9}  {'r_reward':>9}  {'r_td':>9}  DIAGNOSIS")
    for r in corr_rows:
        diagnosis = []
        if r["corr_with_reward"] > 0.1 and r["theta"] > 0:
            diagnosis.append("SIGN_OK: bad_feature+high_reward?")
        if r["corr_with_target"] > 0.15 and r["theta"] < 0:
            diagnosis.append("THETA_CONFLICTS_CORR")
        if r["corr_with_target"] > 0.15 and r["theta"] > 0:
            diagnosis.append("OPPORTUNITY_STATE_EFFECT")
        _log(f"  {r['feature']:<38} {r['theta']:>+8.4f}  {r['corr_with_target']:>+9.4f}  "
             f"{r['corr_with_reward']:>+9.4f}  {r['corr_with_td_error']:>+9.4f}  {'; '.join(diagnosis)}")

    # ── Section 6: Feature-feature correlation ─────────────────────────────────
    _log("\n" + "─" * 70)
    _log("SECTION 6: FEATURE-FEATURE MULTICOLLINEARITY")
    _log("─" * 70)

    phi_df = pd.DataFrame(phi_cur_mat, columns=feature_names)
    feat_corr = phi_df.corr(method="pearson")
    feat_corr.to_csv(OUT / "feature_feature_correlations.csv")
    _log(f"  Saved -> {OUT}/feature_feature_correlations.csv")

    _log("\n  Highly correlated feature pairs (|r| > 0.80):")
    found_any = False
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            r_val = feat_corr.iloc[i, j]
            if abs(r_val) > 0.80:
                found_any = True
                _log(f"    {feature_names[i]:<38} ↔  {feature_names[j]:<38}  r={r_val:+.3f}")
    if not found_any:
        _log("    None found above 0.80 threshold.")

    # ── Section 7: TD update direction ─────────────────────────────────────────
    _log("\n" + "─" * 70)
    _log("SECTION 7: TD UPDATE DIRECTION ANALYSIS")
    _log("─" * 70)

    _log(f"\n  Total transitions: {len(all_transitions)}")
    _log(f"  reward:    mean={np.mean(rewards):.5f}  std={np.std(rewards):.5f}  min={np.min(rewards):.5f}")
    _log(f"  target:    mean={np.mean(targets):.5f}  std={np.std(targets):.5f}")
    _log(f"  td_error:  mean={np.mean(td_errors):.5f}  std={np.std(td_errors):.5f}")
    _log(f"  v_cur:     mean={np.mean(v_curs):.5f}  std={np.std(v_curs):.5f}")
    _log(f"  v_next:    mean={np.mean(v_nexts):.5f}  std={np.std(v_nexts):.5f}")
    _log(f"\n  Fraction positive TD errors: {np.mean(td_errors > 0):.3f}")
    _log(f"  Fraction negative TD errors: {np.mean(td_errors < 0):.3f}")

    if np.mean(td_errors > 0) > 0.6:
        _log("\n  [WARNING] TD errors mostly POSITIVE. Theta will be pushed POSITIVE.")
        _log("  This can happen when bootstrap target dominates tiny reward signal.")

    # Per-feature update contribution
    _log("\n  Per-feature mean update contribution (alpha * mean_td_error * mean_phi):")
    _log(f"  {'Feature':<38} {'mean_phi':>10}  {'mean_update*':>12}  {'actual_theta':>12}")
    _log("  * = alpha * mean_td_error * mean_phi  (approximate direction of update)")
    alpha = 0.05
    mean_td = float(np.mean(td_errors))
    for i, name in enumerate(feature_names):
        mean_phi = float(np.mean(phi_cur_mat[:, i]))
        est_update = alpha * mean_td * mean_phi
        _log(f"  {name:<38} {mean_phi:>10.4f}  {est_update:>+12.6f}  {policy.theta[i]:>+12.6f}")

    # Save TD samples
    td_df = pd.DataFrame([
        {**{f"phi_{n}": t["phi_cur"][i] for i, n in enumerate(feature_names)},
         "reward": t["reward"], "v_cur": t["v_cur"], "v_next": t["v_next"],
         "target": t["target"], "td_error": t["td_error"]}
        for t in all_transitions[:2000]  # cap at 2k rows
    ])
    td_df.to_csv(OUT / "td_update_samples.csv", index=False)
    _log(f"\n  TD samples (first 2000) saved -> {OUT}/td_update_samples.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Section 8: Monotonicity sanity test
# ─────────────────────────────────────────────────────────────────────────────

def monotonicity_test(policy: LinearVFAPolicy) -> None:
    _log("\n" + "=" * 70)
    _log("SECTION 8: MONOTONICITY SANITY TEST")
    _log("=" * 70)
    _log("  For each 'bad' feature, does increasing it DECREASE value?")
    _log("  (If theta is wrong-signed, increasing a bad feature INCREASES value.)")

    if not policy._initialized:
        _log("  [SKIP] Policy not initialized (no simulation run). Pass a loaded model.")
        return

    fn = policy.FEATURE_NAMES
    theta = policy.theta

    base_phi = np.zeros(len(fn), dtype=np.float32)
    bad_features = [
        "squared_starvation_penalty", "exponential_starvation_penalty",
        "starvation_severity_max", "starvation_count",
        "fleet_broken_fraction", "depot_in_repair_fraction",
        "global_onsite_backlog", "recoverable_starvation",
        "maintenance_urgency",
    ]

    _log(f"\n  {'Feature':<40} {'V_base':>10}  {'V_high':>10}  {'ΔV':>10}  MONOTONE?")
    _log("  " + "-" * 75)
    for fname in bad_features:
        if fname not in fn:
            continue
        idx = fn.index(fname)
        test_phi = base_phi.copy()
        test_phi[idx] = 0.5  # set feature to a "bad" level

        v_base = float(np.dot(theta, base_phi))
        v_high = float(np.dot(theta, test_phi))
        delta  = v_high - v_base
        mono   = "OK (V decreased)" if delta < 0 else "[WRONG] V increased — bad feature looks good!"
        _log(f"  {fname:<40} {v_base:>+10.4f}  {v_high:>+10.4f}  {delta:>+10.4f}  {mono}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 9: Top-5 fixes
# ─────────────────────────────────────────────────────────────────────────────

def print_top_fixes() -> None:
    _log("\n" + "=" * 70)
    _log("SECTION 9: TOP FIXES TO TRY (PRIORITY ORDER)")
    _log("=" * 70)

    fixes = [
        ("FIX 1 [RESOLVED] — Reward scaling is no longer tied to gamma",
         "reward.py now uses a fixed `self._scale_factor = 0.1`.",
         "Do not describe the reward scale as `(1 - gamma)`.",
         "Gamma still controls TD discounting; the reward scale only sets numerical magnitude.",
         "If rewards look too small or large, tune this fixed scale directly.",
         "This item is retained only as a historical diagnostic note."),

        ("FIX 2 [CRITICAL] — Clarify depot_idle_fraction semantics",
         "MP7 `depot_idle_fraction` = depot.fixed_queue / fleet.",
         "Fixed queue = bikes REPAIRED and ready for pickup. This is GOOD, not bad.",
         "Action: rename to `depot_ready_fraction` and accept POSITIVE theta as correct.",
         "OR: replace with `depot_waiting_for_repair_fraction` (bikes not yet fixed)."),

        ("FIX 3 [IMPORTANT] — Disambiguate opportunity vs. penalty in MP6/MP8",
         "depot_in_repair_fraction and recoverable_starvation both encode 'recovery opportunity'.",
         "Positive theta for these may be RATIONAL (agent values reachable repairs).",
         "Action: test by removing them and checking if remaining thetas stabilize."),

        ("FIX 4 [IMPORTANT] — Verify TD(λ) vs batch is not conflicting",
         "use_td_lambda=True → theta updated ONLINE per step, batch_buffer NEVER filled.",
         "apply_batch_update() called at episode end → reads empty buffer → no-op.",
         "This is harmless but wasteful. Pick one: set use_td_lambda=False for batch mode,",
         "OR remove the apply_batch_update() call at episode end when td_lambda=True."),

        ("FIX 5 [DIAGNOSTIC] — Add reward/target logging to training loop",
         "Print mean(reward), mean(target), mean(td_error) per episode.",
         "If mean(td_error) > 0 throughout training → theta diverges positive.",
         "If mean(reward) ≈ 0 → scale_factor is crushing the signal (see Fix 1)."),
    ]

    for title, *details in fixes:
        _log(f"\n  {title}")
        for d in details:
            _log(f"    {d}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Linear VFA diagnostics")
    parser.add_argument("--model",    type=str, default=None, help="Path to trained .pkl model")
    parser.add_argument("--episodes", type=int, default=2,    help="Diagnostic episodes to run (default 2)")
    parser.add_argument("--instance", type=str, default="TD_W34_old")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else None

    # Load policy for static checks (no simulation needed)
    reward_config = RewardConfig()
    dummy_policy  = LinearVFAPolicy(
        alpha=0.05, gamma=0.99, learning_mode=False,
        maintenance_enabled=ENABLE_COMPONENT_FAILURES,
        reward_calculator=RewardCalculator(config=reward_config, gamma=0.99),
    )
    if model_path is not None and model_path.exists():
        loaded = LinearVFAPolicy.load(model_path)
        dummy_policy.theta        = loaded.theta
        dummy_policy.FEATURE_NAMES = loaded.FEATURE_NAMES
        dummy_policy._feature_mask = loaded._feature_mask
        dummy_policy.ALL_FEATURE_NAMES = loaded.ALL_FEATURE_NAMES

    check_feature_mapping(dummy_policy)
    check_reward_convention(dummy_policy)
    check_feature_semantics()
    monotonicity_test(dummy_policy)
    print_top_fixes()

    # Run simulation for data-driven diagnostics
    _log("\n" + "=" * 70)
    _log("RUNNING INSTRUMENTED SIMULATION...")
    _log("=" * 70)
    run_instrumented(args.episodes, args.instance, model_path)

    # Save summary
    summary_path = OUT / "diagnostic_summary.txt"
    summary_path.write_text(SUMMARY.getvalue())
    print(f"\n  Full diagnostic summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
