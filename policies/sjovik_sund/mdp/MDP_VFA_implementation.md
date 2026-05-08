# MDP + VFA Implementation Structure

## Architecture Summary
This codebase is structured as a 3-layer pipeline:

1. **MDP formulation layer (`MDP/`)**
   - Defines canonical state/action/transition abstractions and deterministic post-decision transitions.
2. **VFA learning and policy layer (`VFA/`)**
   - Computes features, learns value-function weights, chooses actions, and orchestrates offline training.
3. **Simulator/environment integration layer (outside these folders)**
   - Provides the live event simulation and concrete executable actions.

In practice, the current implementation is close to this design, but there are interface mismatches between `MDP/` and `VFA/` that should be unified.

---

## 1) Purpose and Responsibility of Each File

## MDP folder (`policies/sjovik_sund/MDP`)

### `mdp_config.py`
- **Primary role:** MDP scenario configuration.
- **Responsibility:** Encodes switches for damage tracking and maintenance actions.
- **Category:** **MDP formulation**.
- **Key outputs:** `MDPConfig.full_maintenance()`, `MDPConfig.no_maintenance()`.

### `mdp_formulation.py`
- **Primary role:** Canonical MDP domain model.
- **Responsibility:**
  - Defines state classes (`MDPState`, `StationInventory`, `DepotInventory`, `VehicleStatus`),
  - Defines action class (`MdpAction`),
  - Defines deterministic transition logic (`PostDecisionState.apply(...)`),
  - Defines stochastic event containers (`StationEvent`, `StochasticTransition`),
  - Defines simulator→MDP extraction functions (`extract_mdp_state`, `extract_station_inventory`, `extract_vehicle_status`, etc.).
- **Category:** **MDP formulation + simulator adapter utilities**.

### `__init__.py`
- **Primary role:** Public API re-export for the MDP package.
- **Responsibility:** Re-exports key classes/functions from `mdp_formulation.py`.
- **Category:** **Supporting utility (package surface)**.

---

## VFA folder (`policies/sjovik_sund/VFA`)

### `vfa_features.py`
- **Primary role:** Feature engineering for value estimation.
- **Responsibility:**
  - Demand forecasting (`DemandForecaster`),
  - Per-station and whole-state feature computation (`VFAFeatures`),
  - Feature normalization (`FeatureNormalizer`).
- **Category:** **VFA/training logic**.

### `vfa_agent.py`
- **Primary role:** Value-function model and weight management.
- **Responsibility:**
  - Holds parameters (`LearningParameters`) and weights `theta`,
  - Evaluates value estimates (`evaluate_value`),
  - Saves/loads model artifacts.
- **Category:** **VFA/training logic**.

### `vfa_policy.py`
- **Primary role:** Online decision policy used by simulator.
- **Responsibility:**
  - Extracts current MDP state,
  - Builds candidate MDP actions,
  - Scores with VFA,
  - Converts selected MDP action to executable simulator action.
- **Category:** **VFA + simulator integration**.

### `offline_experience_collector.py`
- **Primary role:** Offline data collection wrapper around baseline policy.
- **Responsibility:**
  - Runs baseline decisions in simulator,
  - Captures transitions `(pre_state, action, post_state, observed_cost, next_state)`,
  - Serializes/deserializes experience datasets.
- **Category:** **VFA/training data pipeline**.

### `offline_TD0_trainer.py`
- **Primary role:** Offline trainer on collected transitions.
- **Responsibility:**
  - Train/validation loops,
  - TD target/error computation,
  - Iterative weight updates,
  - Early stopping/checkpointing/stats.
- **Category:** **VFA/training logic**.

### `train_offline_vfa.py`
- **Primary role:** End-to-end orchestration script.
- **Responsibility:**
  - CLI entrypoint (`collect`, `train`, `evaluate`, `all`),
  - Wires collector/trainer/policy together,
  - Runs seeds and stores outputs.
- **Category:** **Pipeline orchestration utility**.

### `visualize_training.py`
- **Primary role:** Diagnostics and plotting.
- **Responsibility:** Visualizes training curves/weights from saved model + stats files.
- **Category:** **Supporting utility**.

### `QUICKSTART.py`
- **Primary role:** Example/tutorial snippets.
- **Responsibility:** Demonstrates usage patterns and CLI examples.
- **Category:** **Supporting utility/documentation-in-code**.

### `__init__.py`
- **Primary role:** VFA package export surface.
- **Responsibility:** Re-exports agent/policy/features/collector/trainer for easier imports.
- **Category:** **Supporting utility (package surface)**.

---

## 2) Relationships Between Files

## Core dependency graph (conceptual)

- `VFA/vfa_policy.py` depends on:
  - `MDP/mdp_formulation.py` (state/action/transition abstractions),
  - `VFA/vfa_agent.py` (value scoring),
  - `policies/action.py` (simulator action object).

- `VFA/offline_experience_collector.py` depends on:
  - baseline policy (`Policy`),
  - MDP extraction/transition logic,
  - simulator metrics for observed cost.

- `VFA/offline_TD0_trainer.py` depends on:
  - experience schema from collector,
  - MDP reconstruction + post-decision transitions,
  - `VFAAgent` for value and parameter updates.

- `VFA/train_offline_vfa.py` depends on:
  - collector + trainer + policy + agent,
  - external simulation runner (`run_simulation`, `test_seeds`).

## Who calls what (main paths)

### Offline training path
1. `train_offline_vfa.py::collect_experiences(...)`
2. `CollectorPolicyWrapper.get_best_action(...)`
3. baseline policy decides simulator action
4. collector computes/stores experience row
5. `train_offline_vfa.py::train_offline(...)`
6. `OfflineBatchTrainer.train(...)`
7. `OfflineBatchTrainer._compute_and_update_batch(...)` -> `VFAAgent.evaluate_value(...)`
8. `VFAAgent.save(...)`

### Deployment/evaluation path
1. `train_offline_vfa.py::evaluate_model(...)`
2. `VFAAgent.load(...)`
3. `VFAPolicy.get_best_action(...)`
4. MDP state extraction from simulator
5. candidate actions -> value scoring
6. conversion to simulator action
7. simulator executes action and logs metrics

---

## 3) Data Flow: State, Action, Reward, Transition, Value

## State flow
- **Produced by simulator:** live station/vehicle objects.
- **Projected into MDP:** via extraction utilities (`extract_mdp_state` style flow).
- **Consumed by VFA:** feature builder and policy scoring.

## Action flow
- **Produced by policy (abstract):** `MdpAction` (quantities like rebalancing/repairs/removals/next station).
- **Converted for simulator:** `policies.action.Action` with concrete bike ID lists and timing fields.
- **Consumed by simulator:** executes physical bike operations and vehicle routing.

## Reward/cost flow
- **Produced by simulator metrics:** starvations/congestions over decision intervals.
- **Collected by offline collector:** interval deltas converted to scalar `observed_cost`.
- **Consumed by trainer:** continuous-time TD target
  `gamma^((elapsed_minutes / 2) / 60) * observed_cost + gamma^(elapsed_minutes / 60) * V(next_post_state)`.
  The interval cost/reward is treated as occurring at the interval midpoint; the continuation value is discounted over the full decision interval.

## Transition flow
- **Produced by collector:** serialized transition records (`Experience`).
- **Consumed by trainer:** reconstructs MDP states + action from serialized fields.

## Value estimate flow
- **Produced by VFA agent:** `V_hat(S) = theta^T * phi(S)`.
- **Consumed by trainer:** TD loss and gradient update.
- **Consumed by policy:** route/action ranking at decision time.

---

## 4) How MDP and VFA Are Connected in Practice

## Training flow (offline)
1. Baseline policy drives simulator.
2. Collector builds dataset of MDP transitions and observed costs.
3. Trainer replays transitions and updates `theta` with TD-style updates.
4. Best model checkpoint is selected by validation loss and saved.

## Deployment/decision flow
1. On vehicle arrival, policy extracts `MDPState` from current simulator snapshot.
2. Policy enumerates/selects candidate `MdpAction`s.
3. For each candidate, policy computes post-decision state and VFA value estimate.
4. Best candidate is converted to simulator `Action` and executed.

---

## 5) Code Quality Review (Observed Structural Issues)

## A) Interface mismatches between MDP and VFA
- `MdpAction` in `mdp_formulation.py` uses fields:
  - `current_station`, `rebalancing`, `onsite_repairs`, `depot_removals`, `load_from_queue`, `next_station`
- Several VFA modules use different names (e.g., `station_id`, `rebalance_amount`) and call signatures.
- `PostDecisionState` exposes `apply(...)` but some VFA code calls `apply_action(...)`.

**Impact:** frequent runtime errors and fragile integration.

## B) Missing/incorrect symbol usage
- `vfa_policy.py` calls `StateObservationWrapper.extract_mdp_state(...)`, but `mdp_formulation.py` provides function-based extraction (`extract_mdp_state(...)`), not that class.
- `vfa_policy.py` method signature mismatch observed:
  - `_convert_to_simulator_action(self, mdp_action, vehicle)`
  - but called with `(best_action, vehicle, simul)`.

## C) Inconsistent type/model source
- `vfa_agent.py` imports from `.vfa_state` while policy/trainer import from `MDP/mdp_formulation.py`.
- This suggests two parallel state/action type systems are still mixed.

## D) Internal quality issues
- `vfa_agent.py` has duplicate `import json`.
- `vfa_agent.py::compute_immediate_cost` references `self.params` (likely should be `self.learning_params`).
- `VFA/__init__.py` exports `ExperienceReplayBuffer`, but it is not defined.
- `MDPState` default factory in `mdp_formulation.py` appears to reference `MDPConfig.full` (likely should be `MDPConfig.full_maintenance`).
- `train_offline_vfa.py` creates a dummy `VFAAgent` before immediately replacing it with `VFAAgent.load(...)` (redundant).

## E) Naming drift
- `offline_TD0_trainer.py` class is still named `OfflineBatchTrainer` and docstrings still mention batch gradient descent, while core update loop now performs per-sample TD(0)-style updates.

---

## 6) Suggested Improvements

## 1. Choose one canonical state/action API
Use **only** `MDP/mdp_formulation.py` dataclasses + extraction functions as the source of truth.

- Remove references to `.vfa_state` from VFA modules.
- Standardize all action construction/field names to match `MdpAction`.

## 2. Add an explicit adapter module
Create `VFA/adapters.py` with two focused adapters:
- `sim_to_mdp_state(simul, vehicle_id, config) -> MDPState`
- `mdp_action_to_sim_action(mdp_action, vehicle, simul) -> SimAction`

This isolates conversion logic and prevents leakage across modules.

## 3. Align method names and signatures
- Use one deterministic transition entrypoint (`PostDecisionState.apply(...)`) everywhere.
- Fix `_convert_to_simulator_action` call signature mismatch in policy.

## 4. Tighten package exports
- Remove undefined exports from `VFA/__init__.py`.
- Keep `__all__` synced with actual definitions.

## 5. Rename trainer class/file for clarity
- Rename `OfflineBatchTrainer` -> `OfflineTD0Trainer`.
- Update docstrings/messages to reflect true TD(0) updates.

## 6. Simplify orchestration script
In `train_offline_vfa.py`:
- Remove redundant `VFAAgent(...)` creation before `VFAAgent.load(...)`.
- Remove unused imports/flags (e.g., `ENABLE_COMPONENT_FAILURES` if not used).

## 7. Add minimal contract tests
Create small unit tests for:
- action field compatibility,
- state extraction shape/content,
- adapter conversion (MDP action -> simulator action),
- trainer one-step TD update.

---

## 7) Recommended Clean Module Boundaries

- **`MDP/`**
  - only domain model + deterministic transitions + extraction functions.
- **`VFA/`**
  - only feature/value learning + policy search logic.
- **`VFA/adapters.py`**
  - all simulator↔MDP↔simulator conversions.
- **`train_offline_vfa.py`**
  - orchestration only (no MDP math, no conversion internals).

This split gives a clear dependency direction:

`Simulator` -> `MDP adapters/formulation` -> `VFA scoring/training` -> `Simulator action adapter`

with no circular model drift.

---

## Final Note
Your high-level architecture is good and already close to production-ready. The main gap is not algorithmic; it is **type/interface consistency** between MDP and VFA modules. Unifying those contracts will remove most runtime errors and make the TD(0) workflow much easier to maintain and benchmark.
