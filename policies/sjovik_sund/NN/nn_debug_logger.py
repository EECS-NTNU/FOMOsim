"""
nn_debug_logger.py — Structured maintenance-aware debug logger

Writes a human-readable .log file alongside the training CSV.
Primary goal: diagnose whether the NN is learning to value maintenance
actions (ONSITE_REP / DEPOT_REM / DEPOT_PICK) relative to rebalancing.

Key signals tracked:
  Action distribution     — how often each action type is chosen
  Q-value by type         — is maintenance consistently undervalued?
  Value gap               — chosen_V - best_maint_V when rebal wins
  Maintenance exploration — is maintenance being tried at all?
  Fleet health            — broken / in_repair / fixed_queue over time

Usage (training):
    logger = MaintenanceDebugLogger(log_path)
    logger.log_episode_start(ep, total, tau, lr)
    # inside NNLearningPolicy.get_best_action():
    logger.log_training_decision(sim_time, chosen_mdp_action, all_candidates, reward, mdp_state)
    # end of episode:
    logger.log_episode_summary(sl, mean_loss, buffer_size, greedy_sl)
    logger.close()

Usage (rollout / evaluation):
    logger.log_rollout_decision(sim_time, vehicle_id, pre_scored, chosen_sim_action, sim_state)
"""

from __future__ import annotations

import statistics as _stat
from pathlib import Path
from typing import Any, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Action classifier
# ─────────────────────────────────────────────────────────────────────────────

ALL_TYPES        = ("REBAL", "ONSITE_REP", "DEPOT_REM", "DEPOT_PICK", "NOTHING")
MAINTENANCE_TYPES = {"ONSITE_REP", "DEPOT_REM", "DEPOT_PICK"}


def classify_mdp_action(mdp_action) -> str:
    """
    Map an MdpAction to one of five canonical type strings.

    Priority: maintenance components take precedence over rebalancing,
    so a combined 'repair + rebalance' action is classified as ONSITE_REP.
    """
    if mdp_action is None:
        return "NOTHING"
    if getattr(mdp_action, "load_from_queue", 0) > 0:
        return "DEPOT_PICK"     # picking up repaired bikes from depot
    if getattr(mdp_action, "depot_removals",  0) > 0:
        return "DEPOT_REM"      # loading broken bikes onto vehicle for depot transport
    if getattr(mdp_action, "onsite_repairs",  0) > 0:
        return "ONSITE_REP"     # repairing bikes in-place
    if getattr(mdp_action, "rebalancing",     0) != 0:
        return "REBAL"          # pure rebalancing (deliver or pick up functional bikes)
    return "NOTHING"            # stay / no-op


# ─────────────────────────────────────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────────────────────────────────────

class MaintenanceDebugLogger:
    """
    Writes structured maintenance-focused debug logs to a dedicated .log file.

    File layout per episode:
        ═══ EPISODE N/M  tau=X  lr=Y ═══
          ┌─ DEC  50 │ t=450min │ broken=12 ...    (every log_detail_every decisions)
          │  Candidates (6): ...
          └─
          ┌── EPISODE N SUMMARY ──
          │ Action dist: REBAL=120(85%) | ONSITE_REP=10(7%) | ...
          │ Mean V:      REBAL=-8.23 | ONSITE_REP=-9.11 | ...
          │ Value gap:   mean +1.34  (rebal consistently scores higher)
          │ Maint explore: 45 decisions had maint option → chosen 10 (22%)
          └─

    Args:
        log_path         : Path to write the .log file.
        log_detail_every : Full candidate table logged every N decisions (default 50).
    """

    def __init__(self, log_path: Path, log_detail_every: int = 50, tee_stdout: bool = True):
        self.log_path          = log_path
        self._log_detail_every = log_detail_every
        self._tee              = tee_stdout
        self._fh               = log_path.open("w", encoding="utf-8", buffering=1)
        self._episode          = 0
        self._decision_count   = 0     # shared counter across all calls (training + rollout)
        self._reset_episode_state()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _w(self, line: str = "") -> None:
        self._fh.write(line + "\n")
        self._fh.flush()
        if self._tee:
            print(line, flush=True)

    def _reset_episode_state(self) -> None:
        self._action_counts      = {t: 0   for t in ALL_TYPES}
        self._values_by_type     = {t: []  for t in ALL_TYPES}
        self._rewards_by_type    = {t: []  for t in ALL_TYPES}
        self._n_maint_available  = 0    # decisions with ≥1 maintenance candidate
        self._n_maint_chosen     = 0    # decisions where maintenance was chosen
        self._maint_value_gaps   = []   # chosen_V - best_maint_V (when rebal wins)
        self._broken_snapshots   = []   # total broken bikes per decision
        self._in_repair_snapshots = []
        self._fixed_q_snapshots  = []
        self._ep_decision_count  = 0    # decisions this episode (reset each episode)

    def _fleet_from_mdp(self, mdp_state) -> Tuple[int, int, int, int]:
        """Extract (n_broken, n_in_repair, n_fixed_queue, total_capacity) from MDPState."""
        n_broken  = sum(s.onsite + s.depot for s in mdp_state.stations.values())
        depot     = mdp_state.depot
        n_in_rep  = depot.in_repair   if depot else 0
        n_fixed_q = depot.fixed_queue if depot else 0
        total_cap = max(sum(s.capacity for s in mdp_state.stations.values()), 1)
        return n_broken, n_in_rep, n_fixed_q, total_cap

    def _fleet_from_sim(self, sim_state) -> Tuple[int, int, int, int, int]:
        """Extract (n_func, n_broken, n_in_repair, n_fixed_q, total_cap) from live sim.State."""
        try:
            stations  = list(sim_state.get_stations())
            n_func    = 0
            n_broken  = 0
            for s in stations:
                for b in s.get_bikes():
                    ds = getattr(b, "damage_status", None)
                    if ds in ("depot", "onsite"):
                        n_broken += 1
                    else:
                        n_func += 1
            total_cap = max(sum(s.capacity for s in stations), 1)
            depots    = list(sim_state.get_depots())
            n_in_rep  = sum(
                sum(len(bikes) for _, bikes in getattr(d, "in_repair", []))
                for d in depots
            )
            n_fixed_q = sum(len(getattr(d, "fixed_queue", [])) for d in depots)
            return n_func, n_broken, n_in_rep, n_fixed_q, total_cap
        except Exception:
            return -1, -1, -1, -1, -1

    # ── Public API ────────────────────────────────────────────────────────────

    def log_episode_start(self, episode: int, total: int, tau: float, lr: float) -> None:
        """Call at the start of each training episode."""
        self._reset_episode_state()
        self._episode = episode
        self._w()
        self._w("═" * 72)
        self._w(f"EPISODE {episode}/{total}   tau={tau:.4f}   lr={lr:.6f}")
        self._w("═" * 72)

    def log_training_decision(
        self,
        sim_time:           float,
        chosen_mdp_action,
        all_candidates:     List[Tuple[Any, float]],  # [(MdpAction, value), ...]
        reward:             float,
        mdp_state,
    ) -> None:
        """
        Called once per learning-phase vehicle decision from NNLearningPolicy.

        Accumulates per-episode stats and, every log_detail_every decisions,
        writes the full ranked candidate table to the log file.
        """
        self._decision_count     += 1
        self._ep_decision_count  += 1

        chosen_type  = classify_mdp_action(chosen_mdp_action)
        chosen_value = next(
            (v for a, v in all_candidates if a is chosen_mdp_action), None
        )

        # Accumulate action / value / reward stats
        self._action_counts[chosen_type] += 1
        if chosen_value is not None:
            self._values_by_type[chosen_type].append(chosen_value)
        self._rewards_by_type[chosen_type].append(reward)

        # Maintenance availability and value gap
        maint_cands = [(a, v) for a, v in all_candidates
                       if classify_mdp_action(a) in MAINTENANCE_TYPES]
        if maint_cands:
            self._n_maint_available += 1
            if chosen_type in MAINTENANCE_TYPES:
                self._n_maint_chosen += 1
            elif chosen_value is not None:
                best_maint_v = max(v for _, v in maint_cands)
                self._maint_value_gaps.append(chosen_value - best_maint_v)

        # Fleet health snapshot
        n_broken, n_in_rep, n_fixed_q, total_cap = self._fleet_from_mdp(mdp_state)
        self._broken_snapshots.append(n_broken)
        self._in_repair_snapshots.append(n_in_rep)
        self._fixed_q_snapshots.append(n_fixed_q)

        # Full candidate table (periodic)
        if self._ep_decision_count % self._log_detail_every == 0:
            self._w()
            cur_station = getattr(chosen_mdp_action, "current_station", "?")
            self._w(
                f"  ┌─ DEC {self._ep_decision_count:4d} │ t={sim_time:.0f}min │ at={cur_station} │ "
                f"broken={n_broken} in_repair={n_in_rep} "
                f"fixed_q={n_fixed_q} (cap={total_cap})"
            )
            self._w(f"  │  Candidates ({len(all_candidates)}):")
            for a, v in sorted(all_candidates, key=lambda x: -x[1]):
                atype   = classify_mdp_action(a)
                mark    = " ◄" if a is chosen_mdp_action else ""
                reb     = getattr(a, "rebalancing",    0)
                onsite  = getattr(a, "onsite_repairs", 0)
                depot_r = getattr(a, "depot_removals", 0)
                depot_p = getattr(a, "load_from_queue", 0)
                next_s  = getattr(a, "next_station",   "?")
                self._w(
                    f"  │    {atype:<12s} → {next_s:<6}  "
                    f"reb={reb:+3d} rep={onsite} rem={depot_r} pick={depot_p}  "
                    f"V={v:8.3f}{mark}"
                )
            self._w(f"  │  chosen={chosen_type}  r_k={reward:.4f}")
            if maint_cands and chosen_type not in MAINTENANCE_TYPES:
                best_mv = max(v for _, v in maint_cands)
                gap     = (chosen_value or 0.0) - best_mv
                note    = "rebal wins" if gap > 0 else "maint undervalued?"
                self._w(
                    f"  │  ⚠ Maint available (best V={best_mv:.3f})  "
                    f"gap={gap:+.3f}  ({note})"
                )
            self._w("  └─")

    def log_episode_summary(
        self,
        sl:          float,
        mean_loss:   float,
        buffer_size: int,
        greedy_sl:   Optional[float] = None,
    ) -> None:
        """Call at the end of each training episode, after sl and mean_loss are known."""
        ep        = self._episode
        total_dec = max(sum(self._action_counts.values()), 1)

        self._w()
        self._w(f"  ┌── EPISODE {ep} SUMMARY {'─'*40}")

        # Action distribution
        dist_parts = [
            f"{t}={self._action_counts[t]}({100 * self._action_counts[t] // total_dec}%)"
            for t in ALL_TYPES
        ]
        self._w(f"  │ Action dist  : {' | '.join(dist_parts)}")

        # Mean Q-value per action type (key signal: is maintenance undervalued?)
        val_parts = []
        for t in ALL_TYPES:
            vs = self._values_by_type[t]
            if vs:
                val_parts.append(f"{t}={_stat.mean(vs):.3f}")
        if val_parts:
            self._w(f"  │ Mean V       : {' | '.join(val_parts)}")

        # Mean reward per action type (do maintenance actions lead to better rewards?)
        rew_parts = []
        for t in ALL_TYPES:
            rs = self._rewards_by_type[t]
            if rs:
                rew_parts.append(f"{t}={_stat.mean(rs):.4f}")
        if rew_parts:
            self._w(f"  │ Mean r       : {' | '.join(rew_parts)}")

        # Maintenance exploration rate
        if self._n_maint_available > 0:
            pct = 100 * self._n_maint_chosen // max(self._n_maint_available, 1)
            self._w(
                f"  │ Maint explore: {self._n_maint_available} decisions had maint option  "
                f"→ chosen {self._n_maint_chosen}x ({pct}%)"
            )
        else:
            self._w("  │ Maint explore: no maintenance candidates generated this episode")

        # Value gap (positive = rebal consistently beats maintenance)
        if self._maint_value_gaps:
            mg = _stat.mean(self._maint_value_gaps)
            if self._maint_value_gaps:
                mg_min = min(self._maint_value_gaps)
                mg_max = max(self._maint_value_gaps)
            diagnosis = (
                "rebal consistently scores higher → NN prefers rebal"
                if mg > 0.05 else
                "maint competitive with rebal"
                if abs(mg) <= 0.05 else
                "maint scores higher than rebal → NN prefers maint"
            )
            self._w(
                f"  │ Value gap    : mean={mg:+.4f}  "
                f"[{mg_min:+.3f}, {mg_max:+.3f}]  → {diagnosis}"
            )

        # Fleet health over episode
        if self._broken_snapshots:
            self._w(
                f"  │ Broken bikes : mean={_stat.mean(self._broken_snapshots):.1f}  "
                f"max={max(self._broken_snapshots)}  "
                f"min={min(self._broken_snapshots)}"
            )
        if self._in_repair_snapshots:
            self._w(
                f"  │ In repair    : mean={_stat.mean(self._in_repair_snapshots):.1f}  "
                f"max={max(self._in_repair_snapshots)}"
            )
        if self._fixed_q_snapshots:
            self._w(
                f"  │ Fixed queue  : mean={_stat.mean(self._fixed_q_snapshots):.1f}  "
                f"max={max(self._fixed_q_snapshots)}"
            )

        # Training metrics
        greedy_str = f"  greedy_sl={greedy_sl:.4f}" if greedy_sl is not None else ""
        self._w(
            f"  │ Training     : sl={sl:.4f}{greedy_str}  "
            f"loss={mean_loss:.5f}  buffer={buffer_size}"
        )
        self._w(f"  └{'─'*50}")

    def log_rollout_decision(
        self,
        sim_time:          float,
        vehicle_id,
        pre_scored:        List[Tuple[float, Any, Any]],  # [(value, MdpAction, sim_action), ...]
        chosen_sim_action,
        sim_state,
        log_every:         int = 1,   # only log every N rollout calls (use >1 to reduce volume)
    ) -> None:
        """
        Called from NNRolloutPolicy.get_best_action() after action selection.

        Logs all pre-scored candidates ranked by Q-value, highlights the
        chosen action, and flags when maintenance was available but rejected.

        Args:
            pre_scored        : list of (pre_score_value, MdpAction, sim_action)
                                — ALL candidates, not just the top-N passed to rollout.
            chosen_sim_action : the sim_action ultimately selected (after full rollout).
            log_every         : only write every N calls to cap log volume during eval.
        """
        self._decision_count += 1
        if self._decision_count % log_every != 0:
            return

        n_func, n_broken, n_in_rep, n_fixed_q, total_cap = self._fleet_from_sim(sim_state)

        self._w()
        self._w(
            f"  [ROLLOUT] t={sim_time:.0f}  v={vehicle_id}  "
            f"func={n_func} broken={n_broken} in_repair={n_in_rep} "
            f"fixed_q={n_fixed_q} (cap={total_cap})"
        )
        self._w(f"  Candidates ({len(pre_scored)}):")

        chosen_v    = None
        chosen_type = None
        for v, mdp_action, sim_action in sorted(pre_scored, key=lambda x: -x[0]):
            atype   = classify_mdp_action(mdp_action)
            is_chosen = (sim_action is chosen_sim_action)
            if is_chosen:
                chosen_v    = v
                chosen_type = atype
            mark    = " ◄CHOSEN" if is_chosen else ""
            reb     = getattr(mdp_action, "rebalancing",    0)
            onsite  = getattr(mdp_action, "onsite_repairs", 0)
            depot_r = getattr(mdp_action, "depot_removals", 0)
            depot_p = getattr(mdp_action, "load_from_queue", 0)
            next_s  = getattr(mdp_action, "next_station",   "?")
            self._w(
                f"    {atype:<12s} → {next_s:<6}  "
                f"reb={reb:+3d} rep={onsite} rem={depot_r} pick={depot_p}  "
                f"V={v:8.2f}{mark}"
            )

        # Flag if maintenance was available but not chosen
        if chosen_type not in MAINTENANCE_TYPES:
            maint = [(v, a) for v, a, _ in pre_scored
                     if classify_mdp_action(a) in MAINTENANCE_TYPES]
            if maint:
                best_mv = max(v for v, _ in maint)
                gap     = (chosen_v or 0.0) - best_mv
                note    = "rebal wins" if gap > 0 else "maint undervalued?"
                self._w(
                    f"  ⚠ Maint available (best V={best_mv:.2f})  "
                    f"gap={gap:+.2f}  ({note})"
                )

    def close(self) -> None:
        """Flush and close the log file. Call once after training completes."""
        self._fh.flush()
        self._fh.close()
