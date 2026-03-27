# Hybrid VFA + Rollout Architecture

This document explains the full architecture and workflow of the Sjovik-Sund
hybrid ADP pipeline:

- offline training of a linear value function approximation (VFA),
- feature ablation experiments,
- simulator and state modifications required for maintenance-aware routing,
- and online evaluation of a hybrid rollout policy that uses the trained VFA as
  a terminal value estimator.

The goal is to make the codebase understandable both as an implementation and
as a research pipeline.

## 1. High-Level Idea

The project uses a two-stage approximate dynamic programming design:

1. A **linear VFA** is trained offline from episodic simulation.
2. A **hybrid rollout policy** uses that frozen VFA online to score the tail of
   a short explicit lookahead.

Conceptually:

- `LinearVFAPolicy` learns an approximation of the downstream value of a
  post-decision state.
- `HybridRolloutPolicy` explicitly simulates candidate actions for a short
  horizon, then adds the trained VFA as a terminal continuation value.

This is a standard “horizontal ADP” pattern:

- explicit simulation handles near-term consequences,
- the VFA handles long-term continuation.

## 2. Research Workflow

The intended end-to-end workflow is:

1. Define candidate feature sets in the ablation study.
2. Train one VFA model per feature configuration and random seed.
3. Compare learning curves and final performance across feature sets.
4. Select a trained model.
5. Run the selected model either:
   - as a standalone greedy VFA policy, or
   - as the terminal estimator inside `HybridRolloutPolicy`.
6. Export detailed CSV logs for analysis.

The relevant entry points are:

- `policies/sjovik_sund/ablation_study/run_ablation_study.py`
- `policies/sjovik_sund/vfa/train_vfa.py`
- `policies/sjovik_sund/vfa/evaluate_hybrid_rollout.py`
- `policies/sjovik_sund/run_simulation_ingvild.py`

## 3. Main Modules

### Core VFA modules

- `LinearVFAPolicy.py`
  - feature extraction from post-decision states,
  - value computation `V(S^x) = theta^T phi(S^x)`,
  - candidate generation,
  - Boltzmann exploration during training,
  - TD(0) weight updates,
  - load/save of trained models.

- `vfa_features.py`
  - canonical feature registry,
  - canonical feature computation,
  - single source of truth for active feature names.

- `HybridRolloutPolicy.py`
  - online rollout policy,
  - clones the simulator,
  - applies candidate actions to cloned states,
  - simulates forward for a short horizon,
  - uses the frozen VFA as terminal value.

### Canonical MDP layer

- `policies/sjovik_sund/mdp/mdp_formulation.py`
  - shared representation of state, actions, and post-decision states.

- `policies/sjovik_sund/mdp/action_bridge.py`
  - converts canonical `MdpAction` objects into simulator `sim.Action`
    objects.

- `policies/sjovik_sund/mdp/reward.py`
  - reward accounting based on simulator metrics.

### Experiment / pipeline modules

- `train_vfa.py`
  - offline episodic VFA training.

- `run_ablation_study.py`
  - runs multiple feature subsets and multiple seed offsets.

- `evaluate_hybrid_rollout.py`
  - loads a trained model and evaluates:
    - `DoNothing`,
    - `VFA_Only_Standalone`,
    - `Hybrid_Rollout`.

### Simulator / environment modules modified for this workflow

- `sim/Simulator.py`
- `sim/State.py`
- `sim/Depot.py`
- `sim/Station.py`
- `sim/Area.py`
- `sim/events/VehicleArrival.py`

## 4. Problem Representation

The simulator models a dynamic bike rebalancing and maintenance problem where a
service vehicle moves between stations and depots while customer demand evolves
stochastically over time.

At a decision epoch, the active vehicle chooses:

- how many functional bikes to deliver or pick up,
- how many on-site repairs to perform,
- how many depot-damaged bikes to remove,
- where to drive next.

The canonical MDP state separates:

- **station inventory**
  - functional bikes,
  - onsite-repair bikes,
  - depot-repair bikes,
- **vehicle inventory**
  - functional cargo,
  - depot-damaged cargo,
- **time**
  - current simulation time,
  - optional shift timing information.

The post-decision state is central:

- the VFA is evaluated on the post-decision state,
- rollout uses the post-decision action consequence plus short forward
  simulation.

## 5. Feature System

Feature definitions live only in `vfa_features.py`.

This is intentional:

- feature names,
- feature ordering,
- and feature computation

must stay synchronized across training, loading, ablation, and evaluation.

The file currently contains three categories of features.

### Category A: Rebalancing features

Examples:

- `rebalancing_imbalance`
- `anticipated_demand_shortfall`
- `vehicle_functional_load`
- `squared_starvation_penalty`
- `squared_congestion_penalty`
- `proximity_to_demand_gravity`
- `delivery_potential`
- `pickup_potential`
- `starvation_gravity`
- `congestion_gravity`

These describe system imbalance, expected short-term shortage/congestion, and
how useful the current vehicle inventory is relative to network demand.

### Category B: Maintenance features

Examples:

- `trailer_cannibalization`
- `global_onsite_backlog`
- `demand_weighted_depot_backlog`
- `depot_pull`

These quantify maintenance burden and the pressure to route toward a depot.

### Category C: Shift timing features

Examples:

- `time_remaining_fraction`
- `functional_bikes_time_penalty`

These are optional anticipatory features for end-of-shift behavior.

## 6. Offline VFA Training

Training is implemented in `train_vfa.py`.

### Episode structure

Each training episode is a full simulation with two phases:

1. **Warm-up phase**
   - controlled by `GreedyPolicy`,
   - no TD learning,
   - used to build a realistic non-empty system state before learning starts.

2. **Learning phase**
   - controlled by `LinearVFAPolicy`,
   - uses Boltzmann exploration,
   - updates `theta` via TD(0).

### Why the warm-up exists

Without warm-up, the VFA would learn from unrealistic startup conditions:

- empty or near-empty routing histories,
- unrealistically synchronized station inventories,
- unrepresentative congestion/starvation patterns.

Warm-up makes the learning phase start from a more representative operating
regime.

### TD update

The learned parameter vector `theta` is updated online during the learning
phase:

`theta <- theta + alpha * (r + gamma * V(next) - V(cur)) * phi(cur)`

where:

- `phi(cur)` is the stored feature vector for the last chosen post-decision
  state,
- reward comes from changes in starvation/congestion metrics,
- `phi(next)` is computed from the best downstream candidate.

### Model outputs

Training produces:

- a `.pkl` file containing the trained VFA,
- a learning curve `.npy`,
- a `*_weights_evolution.csv`,
- detailed simulation CSV outputs for each episode under
  `policies/sjovik_sund/simulation_results/csv/...`

## 7. Ablation Study Workflow

Feature ablations are configured in:

- `policies/sjovik_sund/ablation_study/run_ablation_study.py`

The file defines named feature subsets such as:

- `1_Linear_Reactive`
- `2_Non_Linear_Reactive`
- `3_Anticipatory_Spatial_Base`
- `4_Contextual_Interactions`

For each experiment:

1. a subset of feature names is selected,
2. multiple macro-seed runs are launched,
3. each run trains a fresh VFA from scratch,
4. artifacts are saved under `models/ablation_study/<experiment>/`.

This isolates the contribution of different feature families:

- simple reactive imbalance features,
- nonlinear penalties,
- anticipatory/spatial features,
- interaction features between inventory and network context.

## 8. Standalone VFA Decision Logic

`LinearVFAPolicy` is the main deployment policy for a trained model.

At each vehicle decision:

1. lazily initialize caches from the simulator state,
2. generate a tractable set of candidate actions,
3. compute post-decision feature vectors for each candidate,
4. evaluate the linear value function,
5. select:
   - via Boltzmann during training,
   - via greedy argmax during exploitation.

### Candidate generation

Action generation is split into two levels:

- **micro decision**
  - how many bikes to pick up/deliver/repair at the current station,
- **macro decision**
  - which next station/depot to route to.

The routing candidate pool is pruned using:

- nearest stations,
- critical stations,
- a simple tabu mechanism to avoid sending multiple vehicles to the same
  claimed target.

## 9. Hybrid Rollout Logic

`HybridRolloutPolicy` wraps a trained `LinearVFAPolicy`.

### Decision flow

At a real decision epoch:

1. Generate candidate actions using the trained VFA’s candidate generator.
2. For each candidate:
   - clone the simulator,
   - manually apply the candidate action to the clone,
   - simulate forward for a short horizon,
   - let downstream cloned vehicles act using the frozen VFA,
   - accumulate discounted rollout reward,
   - compute a terminal VFA value at the horizon.
3. Choose the candidate with the highest estimated rollout value.

### Why time appears to “jump backward” in debug output

Rollout evaluates multiple branches starting from the same root state.

So log patterns like:

- `07:33 -> candidate A`
- later again `07:33 -> candidate B`

do not mean the real simulator is moving backward. They mean the rollout is
restarting from the same real decision state and evaluating another branch.

### Why the hybrid policy is slow

The rollout policy is expensive by design. For each real decision it performs:

- candidate generation,
- simulator cloning,
- scenario simulation,
- event-by-event forward execution.

Runtime scales roughly with:

`#real decisions x #candidates x #scenarios x #events inside horizon`

The main tuning levers are:

- `lookahead_minutes`
- `num_scenarios`
- `LinearVFAPolicy.N_CANDIDATES`

## 10. Simulator and Environment Modifications

The original simulator was extended to support maintenance-aware ADP and
rollout.

### `sim/Depot.py`

The depot was extended with explicit repair-cycle logic:

- `fixed_queue`
  - repaired bikes ready for pickup,
- `in_repair`
  - bikes currently undergoing the repair cycle,
- `receive_bikes_for_repair(...)`
  - pushes dropped-off broken bikes into the repair queue,
- `tick_repair_queue(...)`
  - moves finished repairs into `fixed_queue`.

This creates a delayed depot-repair process rather than an instantaneous
repair abstraction.

### `sim/Simulator.py`

The simulator was extended with:

- depot queue ticking after each event,
- rollout-safe dummy logging methods,
- `sloppycopy()` to create fast simulator clones for rollout.

The clone path is designed to avoid expensive full `deepcopy()` where
possible.

### `sim/State.py`

The state layer was extended with:

- maintenance-aware action execution,
- depot drop-off and depot pickup logic,
- onsite repair logic,
- fast state cloning via `sloppycopy()`.

The current clone strategy uses a **shared bike map**:

- each `bike_id` is cloned once,
- the same cloned bike object is reused across:
  - station inventories,
  - depot queues,
  - vehicle cargo,
  - bikes-in-use.

This avoids identity mismatches that arise from naive shallow copying.

### `sim/Station.py` and `sim/Area.py`

These classes now support graph-consistent cloning of location inventories.

The key design choice is:

- static network data can be reused,
- dynamic bike objects must be cloned consistently.

### `sim/events/VehicleArrival.py`

The vehicle-arrival event is the operational decision trigger:

- policy chooses an action,
- state executes the action,
- travel and operation time are converted into the next arrival event,
- operational logging is emitted for the real simulation.

This same behavior is mirrored inside rollout when candidate actions are
applied to cloned simulators.

## 11. Sloppy Copy vs Deep Copy

One of the biggest implementation challenges in the hybrid rollout was cloning.

### Why `deepcopy` is too slow

Full simulator deep copies are expensive because the simulator contains:

- a large state graph,
- event objects,
- vehicle objects,
- bike objects,
- depot queues,
- logging structures,
- random generators.

Using `deepcopy()` for every rollout branch quickly becomes the dominant cost.

### Why naive `copy.copy()` is unsafe

Plain shallow copying breaks the simulator graph:

- the same physical bike may be duplicated into multiple unrelated objects,
- vehicle cargo and station inventory may stop referring to the same cloned
  bike,
- queued repair bikes may diverge from depot or station references.

This can produce runtime errors such as bike-ID mismatches during action
execution.

### Current solution

The current strategy is a custom fast clone:

- clone each bike exactly once into a shared `bike_map`,
- rebuild all dynamic containers from those shared cloned bikes,
- relink vehicles and event queue references into the cloned universe.

This is the main reason the rollout can be made workable without relying on
full deep copies.

## 12. Logging and Output Files

The pipeline writes outputs at multiple levels.

### Training and evaluation summaries

Main results CSVs are written by:

- `write_results_to_file(...)`

under:

- `policies/sjovik_sund/simulation_results/csv/`

### Detailed per-run outputs

Additional outputs include:

- bike movement logs,
- component failure logs,
- vehicle cargo logs,
- daily health logs,
- RL/VFA decision logs.

### Model artifacts

Trained models and weight evolution files are written under:

- `models/`
- `models/ablation_study/...`

## 13. How to Run the Full Pipeline

### A. Run ablation training

```bash
python policies/sjovik_sund/ablation_study/run_ablation_study.py
```

This trains multiple feature subsets and writes:

- trained models to `models/ablation_study/...`
- per-run simulation CSVs to `policies/sjovik_sund/simulation_results/csv/...`

### B. Train a single VFA directly

```bash
python policies/sjovik_sund/vfa/train_vfa.py \
  --episodes 200 \
  --save models/my_vfa.pkl \
  --instance TD_W34_old
```

### C. Evaluate a trained VFA and hybrid rollout

```bash
python policies/sjovik_sund/vfa/evaluate_hybrid_rollout.py \
  --model models/ablation_study/4_Contextual_Interactions/vfa_4_Contextual_Interactions_run2.pkl \
  --lookahead 60 \
  --scenarios 1 \
  --episodes 1 \
  --duration 120
```

This compares:

- `DoNothing_Baseline`
- `Hybrid_Rollout_H*_S*`
- `VFA_Only_Standalone`

using the standard simulation evaluation pipeline.

## 14. How Feature Experiments Connect to the Final Hybrid Policy

The full research logic is:

1. Start with a broad candidate feature family.
2. Use ablations to identify which feature combinations produce better learning
   and better downstream policy behavior.
3. Train a stable VFA on the selected feature set.
4. Freeze that VFA.
5. Use the frozen VFA in two ways:
   - standalone deployment,
   - hybrid rollout tail evaluation.

This means the hybrid policy is not a separate learned model. It is a
decision-time wrapper around a trained VFA.

## 15. Practical Notes and Known Tradeoffs

### Benefits

- clean separation between offline learning and online decision support,
- interpretable linear value function,
- explicit simulator-based lookahead,
- straightforward ablation workflow.

### Tradeoffs

- rollout is computationally expensive,
- simulator cloning is delicate and must preserve graph consistency,
- results depend strongly on candidate pruning and feature quality,
- maintenance logic makes the environment substantially more stateful than a
  pure rebalancing model.

## 16. Recommended Reading Order for New Contributors

For someone new to the codebase, the easiest path is:

1. `vfa_features.py`
2. `LinearVFAPolicy.py`
3. `mdp_formulation.py`
4. `train_vfa.py`
5. `run_ablation_study.py`
6. `HybridRolloutPolicy.py`
7. `evaluate_hybrid_rollout.py`
8. `run_simulation_ingvild.py`
9. `sim/State.py`, `sim/Depot.py`, `sim/Simulator.py`

That order moves from conceptual definitions to learning to online rollout to
simulator implementation details.

## 17. Summary

This system is a complete research and implementation stack for dynamic
rebalancing with maintenance:

- a canonical MDP layer,
- a modular feature system,
- a trainable linear VFA,
- an ablation framework for feature-set comparison,
- a simulator extended with maintenance-aware depot logic,
- and a hybrid rollout policy that uses short explicit simulation plus a
  learned tail-value approximation.

The central design principle is separation of responsibilities:

- the simulator handles operational realism,
- the MDP layer defines decision semantics,
- the VFA learns long-horizon structure,
- the rollout handles short-horizon branching,
- and the ablation pipeline supports systematic feature research.
