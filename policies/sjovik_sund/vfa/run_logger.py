"""
run_logger.py — Structured per-run output for hybrid rollout evaluations.

One RunLogger instance covers an entire run_batch() call (all experiment × alpha
combinations, all seeds). Each seed appends rows to the shared files under:

    run_logs/run_<timestamp>/
        results.csv          episode-level summary   (one row per seed)
        hourly_metrics.csv   one row per simulated hour per seed
        daily_metrics.csv    one row per simulated day per seed
        decisions.csv        one row per real vehicle decision  (opt-in)
        debug_log.txt        plain-text trace

Usage
─────
    logger = RunLogger(log_decisions=True)
    logger.set_run_label(exp_name, alpha)   # once per experiment combo
    logger.set_seed(seed)                   # once per seed, resets accumulators
    # … simulation runs …
    logger.close()                          # flush + close all handles
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from sim.bike_degradation_modeling import damage_configuration
from policies.sjovik_sund.vfa.vfa_features import get_feature_names as _get_all_feature_names

_FAILURE_CATS = list(damage_configuration.DAMAGE_CATEGORIES.keys())
# Full canonical feature list — used as phi_* columns in decisions.csv so that
# the schema is stable across experiments with different active feature subsets.
_ALL_FEATURE_NAMES = _get_all_feature_names(maintenance_enabled=True, logistics_enabled=True, demand_horizon_enabled=True)


def _cat_key(cat: str) -> str:
    return cat.lower().replace(" & ", "_").replace(" ", "_")


class RunLogger:
    """
    Manages per-run structured CSV output.

    Thread safety: single-threaded use only (use_multiprocessing=False).
    """

    # ── Column definitions ────────────────────────────────────────────────

    @staticmethod
    def _results_columns() -> list[str]:
        return [
            "seed", "exp_name", "alpha", "duration_hours", "total_runtime_s",
            "service_level",
            # Service metrics
            "starvations", "congestions", "total_trips",
            "bike_departures", "bike_arrivals",
            # Maintenance operations
            "total_onsite_repairs", "total_depot_pickups",
            "total_depot_deliveries", "total_depot_visits",
            # Operational movements
            "total_functional_pickups", "total_functional_deliveries",
            # Fleet health – start
            "broken_ratio_start_onsite", "broken_ratio_start_depot",
            "functional_ratio_start",
            # Fleet health – end
            "broken_ratio_end_onsite", "broken_ratio_end_depot",
            "functional_ratio_end",
            # Breakdown events
            "new_breakdowns_onsite", "new_breakdowns_depot",
            # Restorations
            "restored_onsite", "restored_depot",
        ]

    @staticmethod
    def _hourly_columns() -> list[str]:
        base = [
            "seed", "day", "hour",
            # Operational
            "functional_pickups", "functional_deliveries",
            "onsite_repairs", "depot_pickups", "depot_visits", "depot_deliveries",
            "unique_stations_visited",
            # Fleet degradation
            "breakdowns_onsite", "breakdowns_depot",
            "damaged_fraction_onsite", "damaged_fraction_depot",
            "restored_onsite", "restored_depot",
            "total_breakdowns", "total_restored",
            # Demand / service
            "starvations", "congestions",
            "bike_departures", "bike_arrivals", "total_trips",
        ]
        for cat in _FAILURE_CATS:
            base.append(f"new_failures_{_cat_key(cat)}")
        return base

    @staticmethod
    def _daily_columns() -> list[str]:
        return [
            "seed", "day",
            "shift_hour_start", "shift_hour_end",
            "daily_functional_pickups", "daily_functional_deliveries",
            "daily_onsite_repairs", "daily_depot_pickups",
            "daily_depot_visits", "daily_depot_deliveries",
            "daily_starvations", "daily_congestions",
            "daily_trips", "daily_bike_departures", "daily_bike_arrivals",
            "daily_breakdowns_onsite", "daily_breakdowns_depot",
            "daily_total_breakdowns",
            "daily_restored_onsite", "daily_restored_depot",
            "daily_total_restored",
            "eod_damaged_fraction_onsite", "eod_damaged_fraction_depot",
        ]

    @staticmethod
    def _decisions_columns() -> list[str]:
        cols = [
            "seed", "day", "hour", "minute",
            "current_station_id", "is_at_depot",
            # Load before
            "functional_load_before", "depot_load_before", "total_load_before",
            # Action
            "functional_deliveries", "functional_pickups",
            "onsite_repairs", "depot_pickups",
            "depot_deliveries", "load_from_queue",
            "bikes_involved", "action_duration_min",
            # Movement
            "next_station_id", "travel_time_min",
            # Load after
            "functional_load_after", "depot_load_after", "total_load_after",
            # Valuation
            "immediate_reward",
            # VFA standalone: score of chosen post-decision state (empty for hybrid)
            "vfa_value",
            # Hybrid rollout: mean rollout reward and discounted tail (empty for VFA)
            "accumulated_rollout_reward", "tail_value",
            "final_decision_score",
            "maintenance_flag_present", "selected_action_is_maintenance",
            # Rollout diagnostics
            "n_total_candidates", "vfa_top1_next_station",
            "rollout_changed_decision", "winning_candidate_rank",
            "decision_runtime_s",
        ]
        # Feature values — stable schema using full canonical list;
        # inactive features written as empty string for a given experiment.
        cols += [f"phi_{name}" for name in _ALL_FEATURE_NAMES]
        return cols

    # ── Construction ──────────────────────────────────────────────────────

    def __init__(
        self,
        base_dir: str | Path = "run_logs",
        log_decisions: bool = False,
        run_label: str = "",
    ):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{run_label}" if run_label else ""
        self.run_dir = Path(base_dir) / f"run_{ts}{suffix}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.log_decisions = log_decisions

        # Episode context — set by set_run_label / set_seed
        self._exp_name: str = ""
        self._alpha: float = 0.0
        self._policy_type: str = "Hybrid"   # "Hybrid", "VFA", or "DoNothing"
        self._current_seed: int = 0
        self._shift_hour_start: int = 5
        self._results_only: bool = False    # skip hourly/daily files when True

        # Fleet start snapshot
        self._fleet_total_start: int = 0
        self._fleet_func_start: int = 0
        self._fleet_onsite_start: int = 0
        self._fleet_depot_start: int = 0

        # Episode totals (reset by set_seed)
        self._ep_func_pickups = 0
        self._ep_func_deliveries = 0
        self._ep_onsite_repairs = 0
        self._ep_depot_pickups = 0
        self._ep_depot_deliveries = 0
        self._ep_depot_visits = 0
        self._ep_restored_onsite = 0
        self._ep_restored_depot = 0

        # Hourly accumulators (reset after each log_hour call)
        self._hour_func_pickups = 0
        self._hour_func_deliveries = 0
        self._hour_onsite_repairs = 0
        self._hour_depot_pickups = 0
        self._hour_depot_deliveries = 0
        self._hour_depot_visits = 0
        self._hour_restored_onsite = 0
        self._hour_restored_depot = 0
        self._hour_stations: set[str] = set()

        # Buffer of hourly rows for the current day's aggregation
        self._day_hourly_rows: list[dict] = []

        # Per-seed file handles — opened in set_seed(), closed before next seed
        self._results_fh:   Any = None
        self._hourly_fh:    Any = None
        self._daily_fh:     Any = None
        self._decisions_fh: Any = None
        self._results_w:    Any = None
        self._hourly_w:     Any = None
        self._daily_w:      Any = None
        self._decisions_w:  Any = None

        # Top-level debug log shared across all seeds
        self._debug_fh = open(self.run_dir / "debug_log.txt", "w", encoding="utf-8")

        self.debug(f"RunLogger ready → {self.run_dir}")

    # ── Context management ────────────────────────────────────────────────

    def set_run_label(
        self,
        exp_name: str,
        alpha: float,
        policy_type: str = "Hybrid",
        shift_hour_start: int = 5,
        results_only: bool = False,
    ) -> None:
        """Call once per (experiment, alpha, policy_type) combo before test_policies.

        results_only=True: only results.csv is written per seed (no hourly/daily).
        Use for baselines like DoNothing that have no decision-level logging.
        """
        self._exp_name = exp_name
        self._alpha = alpha
        self._policy_type = policy_type
        self._shift_hour_start = shift_hour_start
        self._results_only = results_only
        self.debug(f"Run label: exp={exp_name} alpha={alpha} policy={policy_type} results_only={results_only}")

    def set_seed(self, seed: int) -> None:
        """
        Close current seed's files (if any), create a new per-seed subfolder, and
        open fresh file handles. Call before each simulation seed.
        """
        self._close_seed_files()

        self._current_seed = seed
        self._reset_episode_totals()
        self._reset_hour_accumulators()

        # Build subfolder name: Squared_Temporal_alpha_0.1_seed_9000_Hybrid
        alpha_str = str(self._alpha)
        subdir_name = f"{self._exp_name}_alpha_{alpha_str}_seed_{seed}_{self._policy_type}"
        subdir = self.run_dir / subdir_name
        subdir.mkdir(parents=True, exist_ok=True)

        self._results_fh = open(subdir / "results.csv", "w", newline="", encoding="utf-8")
        self._results_w = csv.DictWriter(
            self._results_fh, fieldnames=self._results_columns(), extrasaction="ignore"
        )
        self._results_w.writeheader()

        if not self._results_only:
            self._hourly_fh = open(subdir / "hourly_metrics.csv", "w", newline="", encoding="utf-8")
            self._daily_fh  = open(subdir / "daily_metrics.csv",  "w", newline="", encoding="utf-8")
            self._hourly_w = csv.DictWriter(
                self._hourly_fh, fieldnames=self._hourly_columns(), extrasaction="ignore"
            )
            self._daily_w = csv.DictWriter(
                self._daily_fh, fieldnames=self._daily_columns(), extrasaction="ignore"
            )
            self._hourly_w.writeheader()
            self._daily_w.writeheader()

        if self.log_decisions and not self._results_only:
            self._decisions_fh = open(subdir / "decisions.csv", "w", newline="", encoding="utf-8")
            self._decisions_w  = csv.DictWriter(
                self._decisions_fh,
                fieldnames=self._decisions_columns(),
                extrasaction="ignore",
                restval="",
            )
            self._decisions_w.writeheader()

        self.debug(f"Seed {seed} → {subdir_name}/")

    def _reset_episode_totals(self) -> None:
        self._ep_func_pickups = 0
        self._ep_func_deliveries = 0
        self._ep_onsite_repairs = 0
        self._ep_depot_pickups = 0
        self._ep_depot_deliveries = 0
        self._ep_depot_visits = 0
        self._ep_restored_onsite = 0
        self._ep_restored_depot = 0
        self._day_hourly_rows = []

    def _close_seed_files(self) -> None:
        """Flush and close all per-seed file handles."""
        for fh in (self._results_fh, self._hourly_fh, self._daily_fh, self._decisions_fh):
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
        self._results_fh = self._hourly_fh = self._daily_fh = self._decisions_fh = None
        self._results_w  = self._hourly_w  = self._daily_w  = self._decisions_w  = None

    def capture_fleet_start(self, state) -> None:
        """
        Snapshot fleet composition before simulator.run().
        Includes bikes in depot repair queues which get_all_bikes() may miss.
        """
        all_bikes = list(state.get_all_bikes())
        n_onsite = sum(1 for b in all_bikes if getattr(b, "damage_status", None) == "onsite")
        n_depot  = sum(1 for b in all_bikes if getattr(b, "damage_status", None) == "depot")
        # Bikes already in depot repair queues
        depot_queue = sum(
            len(bl) for d in state.get_depots() for _, bl in d.in_repair
        )
        n_depot += depot_queue
        total = len(all_bikes) + depot_queue
        self._fleet_total_start  = total
        self._fleet_onsite_start = n_onsite
        self._fleet_depot_start  = n_depot
        self._fleet_func_start   = max(total - n_onsite - n_depot, 0)

    # ── Decision logging (called from HybridRolloutPolicy) ───────────────

    def log_decision(self, row: dict) -> None:
        """
        Update running accumulators from action fields and optionally write a
        decisions.csv row. row keys must match _decisions_columns().
        """
        fp  = int(row.get("functional_pickups", 0))
        fd  = int(row.get("functional_deliveries", 0))
        orr = int(row.get("onsite_repairs", 0))
        dp  = int(row.get("depot_pickups", 0))
        dd  = int(row.get("depot_deliveries", 0))
        dv  = 1 if row.get("is_at_depot", False) else 0
        lfq = int(row.get("load_from_queue", 0))

        # Episode totals
        self._ep_func_pickups     += fp
        self._ep_func_deliveries  += fd
        self._ep_onsite_repairs   += orr
        self._ep_depot_pickups    += dp
        self._ep_depot_deliveries += dd
        self._ep_depot_visits     += dv
        self._ep_restored_onsite  += orr   # onsite repairs → immediately functional
        self._ep_restored_depot   += lfq   # loaded from fixed queue → restored from depot

        # Hour accumulators
        self._hour_func_pickups     += fp
        self._hour_func_deliveries  += fd
        self._hour_onsite_repairs   += orr
        self._hour_depot_pickups    += dp
        self._hour_depot_deliveries += dd
        self._hour_depot_visits     += dv
        self._hour_restored_onsite  += orr
        self._hour_restored_depot   += lfq
        station = row.get("current_station_id")
        if station:
            self._hour_stations.add(str(station))

        if self.log_decisions and self._decisions_w is not None:
            row["seed"] = self._current_seed
            self._decisions_w.writerow(row)
            self._decisions_fh.flush()

    # ── Hourly logging (called from LoggingSimulator.log_hourly_metrics) ─

    def log_hour(self, row: dict) -> None:
        """
        Merge hourly accumulators from decisions into the sim-side metrics row
        and write to hourly_metrics.csv. Resets hour accumulators afterward.

        The caller (LoggingSimulator) provides all sim-side columns; this method
        adds the decision-side columns before writing.
        """
        row.update({
            "seed":                    self._current_seed,
            "functional_pickups":      self._hour_func_pickups,
            "functional_deliveries":   self._hour_func_deliveries,
            "onsite_repairs":          self._hour_onsite_repairs,
            "depot_pickups":           self._hour_depot_pickups,
            "depot_visits":            self._hour_depot_visits,
            "depot_deliveries":        self._hour_depot_deliveries,
            "unique_stations_visited": len(self._hour_stations),
            "restored_onsite":         self._hour_restored_onsite,
            "restored_depot":          self._hour_restored_depot,
            "total_restored":          self._hour_restored_onsite + self._hour_restored_depot,
        })
        self._day_hourly_rows.append(dict(row))
        if self._hourly_w is not None:
            self._hourly_w.writerow(row)
            self._hourly_fh.flush()
        self._reset_hour_accumulators()

    def _reset_hour_accumulators(self) -> None:
        self._hour_func_pickups     = 0
        self._hour_func_deliveries  = 0
        self._hour_onsite_repairs   = 0
        self._hour_depot_pickups    = 0
        self._hour_depot_deliveries = 0
        self._hour_depot_visits     = 0
        self._hour_restored_onsite  = 0
        self._hour_restored_depot   = 0
        self._hour_stations         = set()

    # ── Daily logging (called from LoggingSimulator.log_daily_metrics) ───

    def log_day(self, day: int) -> None:
        """Aggregate this day's hourly rows and write daily_metrics.csv."""
        rows = [r for r in self._day_hourly_rows if r.get("day") == day]
        if not rows:
            return

        def _sum(key: str) -> float:
            return sum(r.get(key, 0) for r in rows)

        eod = rows[-1]   # last hourly snapshot = end-of-day fleet fractions

        daily_row = {
            "seed":                         self._current_seed,
            "day":                          day,
            "shift_hour_start":             self._shift_hour_start,
            "shift_hour_end":               (self._shift_hour_start + 24) % 24,
            "daily_functional_pickups":     _sum("functional_pickups"),
            "daily_functional_deliveries":  _sum("functional_deliveries"),
            "daily_onsite_repairs":         _sum("onsite_repairs"),
            "daily_depot_pickups":          _sum("depot_pickups"),
            "daily_depot_visits":           _sum("depot_visits"),
            "daily_depot_deliveries":       _sum("depot_deliveries"),
            "daily_starvations":            _sum("starvations"),
            "daily_congestions":            _sum("congestions"),
            "daily_trips":                  _sum("total_trips"),
            "daily_bike_departures":        _sum("bike_departures"),
            "daily_bike_arrivals":          _sum("bike_arrivals"),
            "daily_breakdowns_onsite":      _sum("breakdowns_onsite"),
            "daily_breakdowns_depot":       _sum("breakdowns_depot"),
            "daily_total_breakdowns":       _sum("total_breakdowns"),
            "daily_restored_onsite":        _sum("restored_onsite"),
            "daily_restored_depot":         _sum("restored_depot"),
            "daily_total_restored":         _sum("total_restored"),
            "eod_damaged_fraction_onsite":  eod.get("damaged_fraction_onsite", 0.0),
            "eod_damaged_fraction_depot":   eod.get("damaged_fraction_depot", 0.0),
        }
        if self._daily_w is not None:
            self._daily_w.writerow(daily_row)
            self._daily_fh.flush()
        # Discard today's rows; future hours belong to the next day
        self._day_hourly_rows = [r for r in self._day_hourly_rows if r.get("day") != day]

    # ── Episode logging (called from write_simulation_outputs) ────────────

    def log_episode(self, simulator, seed: int, duration: float, solve_time: float) -> None:
        """Write one results.csv row combining sim-side metrics + episode totals."""
        m = simulator.state.metrics

        def ag(key: str):
            return m.get_aggregate_value(key) or 0

        total_trips   = ag("trips")
        failed_events = ag("failed events")
        service_level = (1 - failed_events / total_trips) if total_trips > 0 else 0.0

        # Fleet end state (same correction as log_daily_metrics)
        all_bikes = list(simulator.state.get_all_bikes())
        depot_queue_end = sum(
            len(bl) for d in simulator.state.get_depots() for _, bl in d.in_repair
        )
        total_end   = len(all_bikes) + depot_queue_end
        n_onsite_end = sum(1 for b in all_bikes if getattr(b, "damage_status", None) == "onsite")
        n_depot_end  = sum(1 for b in all_bikes if getattr(b, "damage_status", None) == "depot") + depot_queue_end
        n_func_end   = max(total_end - n_onsite_end - n_depot_end, 0)

        t_start = self._fleet_total_start or 1
        t_end   = total_end or 1

        row = {
            "seed":             seed,
            "exp_name":         self._exp_name,
            "alpha":            self._alpha,
            "duration_hours":   duration,
            "total_runtime_s":  round(solve_time, 2),
            "service_level":    round(service_level, 4),
            "starvations":      ag("starvations"),
            "congestions":      ag("long congestions"),
            "total_trips":      total_trips,
            "bike_departures":  ag("bike departure"),
            "bike_arrivals":    ag("bike arrival"),
            "total_onsite_repairs":        self._ep_onsite_repairs,
            "total_depot_pickups":         self._ep_depot_pickups,
            "total_depot_deliveries":      self._ep_depot_deliveries,
            "total_depot_visits":          self._ep_depot_visits,
            "total_functional_pickups":    self._ep_func_pickups,
            "total_functional_deliveries": self._ep_func_deliveries,
            "broken_ratio_start_onsite":  round(self._fleet_onsite_start / t_start, 4),
            "broken_ratio_start_depot":   round(self._fleet_depot_start  / t_start, 4),
            "functional_ratio_start":     round(self._fleet_func_start   / t_start, 4),
            "broken_ratio_end_onsite":    round(n_onsite_end / t_end, 4),
            "broken_ratio_end_depot":     round(n_depot_end  / t_end, 4),
            "functional_ratio_end":       round(n_func_end   / t_end, 4),
            "new_breakdowns_onsite":      ag("onsite_failures"),
            "new_breakdowns_depot":       ag("depot_failures"),
            "restored_onsite":            self._ep_restored_onsite,
            "restored_depot":             self._ep_restored_depot,
        }
        if self._results_w is not None:
            self._results_w.writerow(row)
            self._results_fh.flush()
        self.debug(
            f"Episode done: seed={seed} service_level={row['service_level']}"
            f" starvations={row['starvations']} congestions={row['congestions']}"
        )

    # ── Utilities ─────────────────────────────────────────────────────────

    def debug(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._debug_fh.write(f"[{ts}] {msg}\n")
        self._debug_fh.flush()

    def close(self) -> None:
        """Flush and close all file handles."""
        self._close_seed_files()
        try:
            self._debug_fh.close()
        except Exception:
            pass
        print(f"[RunLogger] Output written to: {self.run_dir}")
