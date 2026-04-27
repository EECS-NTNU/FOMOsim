"""
rollout_debug_logger.py — Pruning loss detection for NNRolloutPolicy.

Central question answered per step:
  "Did the NN tail-estimator prune a candidate that would have been the
   rollout winner — i.e. did we lose the best action at the pruning stage?"

We can't run rollout on pruned candidates, but we CAN measure risk:
  - Pruning margin: score gap between last-kept and first-pruned (small = risky)
  - Rollout winner margin: how far above the cutoff was the chosen action's NN score
  - NN vs rollout rank agreement (τ) within the kept set — low τ = NN is unreliable
  - Operational profile: which action types survive pruning systematically

Per-step block layout:
  [A] PRUNING RISK        ← the headline: could we have lost the best action?
  [B] SCORE LADDER        ← full ranked list showing the cutoff line
  [C] ROLLOUT RESULT      ← what rollout actually found
  [D] OPERATIONAL PROFILE ← type breakdown: initial / kept / pruned

Cumulative summary appended after every step for full-run bias detection.
"""

from pathlib import Path
from collections import defaultdict

_ALL_TYPES   = ["deliver", "pickup", "onsite_repair", "depot_removal", "depot_pickup", "idle"]
_MAINT_TYPES = {"onsite_repair", "depot_removal", "depot_pickup"}


def _action_type(mdp_action) -> str:
    if mdp_action.onsite_repairs  > 0: return "onsite_repair"
    if mdp_action.depot_removals  > 0: return "depot_removal"
    if mdp_action.load_from_queue > 0: return "depot_pickup"
    if mdp_action.rebalancing     > 0: return "deliver"
    if mdp_action.rebalancing     < 0: return "pickup"
    return "idle"


def _action_label(mdp_action) -> str:
    parts = []
    if mdp_action.rebalancing > 0:
        parts.append(f"deliver+{mdp_action.rebalancing}")
    elif mdp_action.rebalancing < 0:
        parts.append(f"pickup{mdp_action.rebalancing}")
    if mdp_action.onsite_repairs  > 0: parts.append(f"onsite×{mdp_action.onsite_repairs}")
    if mdp_action.depot_removals  > 0: parts.append(f"depot_rem×{mdp_action.depot_removals}")
    if mdp_action.load_from_queue > 0: parts.append(f"depot_pick×{mdp_action.load_from_queue}")
    if not parts:
        parts.append("idle")
    return f"{'|'.join(parts)}  {mdp_action.current_station}→{mdp_action.next_station}"


def _risk_level(pruning_margin: float, tau: float | None) -> str:
    """
    Heuristic risk that the best action was pruned.
    Small margin + low tau = both conditions for a dangerous pruning cut.
    """
    if pruning_margin is None:
        return "N/A  (no pruned candidates)"
    tight = pruning_margin < 1.0        # score units same as reward scale
    unreliable = tau is not None and tau < 0.3
    if tight and unreliable:
        return "HIGH   *** tight margin + NN is unreliable ranker ***"
    if tight:
        return "MEDIUM (tight margin — borderline candidates may be mislabelled)"
    if unreliable:
        return "MEDIUM (NN rank order disagrees with rollout within kept set)"
    return "LOW"


def _pct(num, denom) -> str:
    return f"{100*num/denom:5.1f}%" if denom else "   n/a"


def _kendall_tau(pre_scores_kept, rollout_sorted) -> float | None:
    rollout_rank = {id(sim_a): i for i, (sim_a, _) in enumerate(rollout_sorted)}
    pre_rank_map = {id(sim_a): r for r, (_, _, sim_a) in enumerate(pre_scores_kept)}
    concordant = discordant = 0
    ids = list(pre_rank_map.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if a in rollout_rank and b in rollout_rank:
                if (pre_rank_map[a] < pre_rank_map[b]) == (rollout_rank[a] < rollout_rank[b]):
                    concordant += 1
                else:
                    discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else None


class PruningDebugLogger:
    """
    Args:
        log_path   : file to write to (created fresh each run).
        log_every  : write a full per-step block every Nth decision.
                     Cumulative summary is always updated regardless.
    """

    def __init__(self, log_path: Path, log_every: int = 1):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._f         = open(log_path, "w", buffering=1)
        self._log_every = log_every
        self._step      = 0

        # Cumulative counters
        self._cum_initial : dict[str, int] = defaultdict(int)
        self._cum_kept    : dict[str, int] = defaultdict(int)
        self._cum_pruned  : dict[str, int] = defaultdict(int)
        self._cum_chosen  : dict[str, int] = defaultdict(int)
        self._high_risk_steps = 0
        self._med_risk_steps  = 0

    # ─────────────────────────────────────────────────────────────────────────

    def log_decision(
        self,
        sim_time:             float,
        vehicle_id,
        pre_scores:           list,   # [(raw_v, mdp_action, sim_action), ...]  best→worst
        n_rollout_candidates: int,
        rollout_scores:       list,   # [(sim_action, mean_q), ...]
        chosen_sim_action,
    ) -> None:
        self._step += 1
        f       = self._f
        n_total = len(pre_scores)
        n_kept  = min(n_rollout_candidates, n_total)
        kept    = pre_scores[:n_kept]
        pruned  = pre_scores[n_kept:]

        initial_types = [_action_type(mdp_a) for _, mdp_a, _ in pre_scores]
        kept_types    = [_action_type(mdp_a) for _, mdp_a, _ in kept]
        pruned_types  = [_action_type(mdp_a) for _, mdp_a, _ in pruned]

        chosen_type = next(
            (_action_type(mdp_a) for _, mdp_a, sim_a in kept if sim_a is chosen_sim_action),
            None,
        )

        # Update cumulative counts
        for t in initial_types: self._cum_initial[t] += 1
        for t in kept_types:    self._cum_kept[t]    += 1
        for t in pruned_types:  self._cum_pruned[t]  += 1
        if chosen_type:         self._cum_chosen[chosen_type] += 1

        # ── Key metrics ───────────────────────────────────────────────────────
        last_kept_score  = kept[-1][0]   if kept   else None
        first_prune_score= pruned[0][0]  if pruned else None
        pruning_margin   = (last_kept_score - first_prune_score) if (last_kept_score is not None and first_prune_score is not None) else None

        rollout_sorted   = sorted(rollout_scores, key=lambda x: x[1], reverse=True)
        tau              = _kendall_tau(kept, rollout_sorted) if len(kept) > 1 else None

        best_rollout_sim, best_rollout_q = rollout_sorted[0] if rollout_sorted else (None, None)
        winner_nn_rank   = next((r+1 for r, (_, _, sim_a) in enumerate(kept) if sim_a is best_rollout_sim), None)
        winner_nn_score  = next((s for s, _, sim_a in kept if sim_a is best_rollout_sim), None)
        winner_margin_above_cutoff = (winner_nn_score - first_prune_score) if (winner_nn_score and first_prune_score is not None) else None

        risk = _risk_level(pruning_margin, tau)
        if "HIGH"   in risk: self._high_risk_steps += 1
        if "MEDIUM" in risk: self._med_risk_steps  += 1

        write_block = (self._step % self._log_every == 0)

        SEP = "=" * 72

        if write_block:
            f.write(f"\n{SEP}\n")
            f.write(f"=== Step {self._step:4d}   t={sim_time:7.1f} min   vehicle={vehicle_id} ===\n")
            f.write(f"{SEP}\n")

            # ── [A] PRUNING RISK ──────────────────────────────────────────────
            f.write(f"\n[A] PRUNING RISK ASSESSMENT\n")
            f.write(f"    Risk level        : {risk}\n")
            if pruning_margin is not None:
                f.write(f"    Pruning margin    : {pruning_margin:+.4f}  "
                        f"(last-kept score {last_kept_score:+.4f}  —  first-pruned score {first_prune_score:+.4f})\n")
            else:
                f.write(f"    Pruning margin    : n/a (nothing pruned)\n")
            if tau is not None:
                f.write(f"    NN rank reliability (τ): {tau:+.3f}  "
                        f"({'NN order ≈ rollout order' if tau > 0.5 else 'NN order DIVERGES from rollout'})\n")
            if winner_margin_above_cutoff is not None:
                f.write(f"    Rollout winner margin above cutoff: {winner_margin_above_cutoff:+.4f}  "
                        f"(NN rank {winner_nn_rank}/{n_kept})\n")
                if winner_margin_above_cutoff < 1.0:
                    f.write(f"    *** Winner barely survived pruning — high chance a pruned candidate "
                            f"would have won if rollout had evaluated it ***\n")

            # ── [B] SCORE LADDER ──────────────────────────────────────────────
            f.write(f"\n[B] SCORE LADDER  ({n_total} candidates, cutoff after rank {n_kept})\n")
            kept_lookup = {id(sim_a): (score, mdp_a) for score, mdp_a, sim_a in kept}
            for rank, (score, mdp_a, sim_a) in enumerate(pre_scores):
                is_cutoff = (rank == n_kept)
                is_winner = (sim_a is best_rollout_sim)
                is_chosen = (sim_a is chosen_sim_action)
                tag  = "KEEP " if rank < n_kept else "PRUNE"
                note = ""
                if is_chosen and is_winner: note = "  ← CHOSEN + ROLLOUT WINNER"
                elif is_chosen:             note = "  ← CHOSEN"
                elif is_winner:             note = "  ← ROLLOUT WINNER"
                if is_cutoff:
                    f.write(f"    {'- '*34} cutoff\n")
                f.write(f"    [{tag}] {rank+1:3d}  {score:+9.4f}  {_action_label(mdp_a)}{note}\n")

            # ── [C] ROLLOUT RESULT ────────────────────────────────────────────
            f.write(f"\n[C] ROLLOUT RESULT  (only kept candidates evaluated)\n")
            for sim_a, mean_q in rollout_sorted:
                pre_score, mdp_a = kept_lookup.get(id(sim_a), (None, None))
                label   = _action_label(mdp_a) if mdp_a else "?"
                pre_str = f"NN={pre_score:+.4f}" if pre_score is not None else "NN=?      "
                is_ch   = (sim_a is chosen_sim_action)
                arrow   = ">>>" if is_ch else "   "
                marker  = "  ← CHOSEN" if is_ch else ""
                f.write(f"    {arrow}  Q={mean_q:+9.4f}  {pre_str}  {label}{marker}\n")

            # ── [D] OPERATIONAL PROFILE ───────────────────────────────────────
            f.write(f"\n[D] OPERATIONAL PROFILE\n")
            all_seen = sorted(set(initial_types),
                              key=lambda t: _ALL_TYPES.index(t) if t in _ALL_TYPES else 99)
            f.write(f"    {'TYPE':<16}  {'INIT':>5}  {'KEPT':>5}  {'PRUNED':>6}  {'KEEP%':>6}\n")
            f.write(f"    {'-'*16}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*6}\n")
            for t in all_seen:
                ni = initial_types.count(t)
                nk = kept_types.count(t)
                np = pruned_types.count(t)
                f.write(f"    {t:<16}  {ni:>5}  {nk:>5}  {np:>6}  {_pct(nk,ni):>6}\n")

        # ── Cumulative summary (always written) ───────────────────────────────
        f.write(f"\n{'~'*72}\n")
        f.write(f"CUMULATIVE SUMMARY  after {self._step} decisions  "
                f"[HIGH risk: {self._high_risk_steps}  MEDIUM: {self._med_risk_steps}]\n")
        all_seen = sorted(set(self._cum_initial),
                          key=lambda t: _ALL_TYPES.index(t) if t in _ALL_TYPES else 99)
        f.write(f"    {'TYPE':<16}  {'INIT':>6}  {'KEPT':>6}  {'PRUNED':>7}  {'KEEP%':>6}  {'CHOSEN':>7}\n")
        f.write(f"    {'-'*16}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*7}\n")
        for t in all_seen:
            ni = self._cum_initial[t]
            nk = self._cum_kept[t]
            np = self._cum_pruned[t]
            nc = self._cum_chosen[t]
            f.write(f"    {t:<16}  {ni:>6}  {nk:>6}  {np:>7}  {_pct(nk,ni):>6}  {nc:>7}\n")
        f.write(f"{'~'*72}\n")

    def close(self) -> None:
        self._f.close()
