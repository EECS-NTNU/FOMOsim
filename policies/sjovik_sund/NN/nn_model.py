"""
nn_model.py  —  Deep Sets Value Network

Defines the neural network that approximates V(S^x): the expected future
cost from a post-decision state inside the rollout.

─────────────────────────────────────────────────────────────────────────────
WHY DEEP SETS?
─────────────────────────────────────────────────────────────────────────────
The bike-sharing state has a natural set structure:
  - N stations  (unordered; each described by its own inventory)
  - M vehicles  (unordered; each described by its own load and location)

A flat MLP would treat the position of a station in the input vector as
meaningful (e.g., "slot 3 always means station 3"), which prevents
generalization across different network configurations and breaks if
station ordering changes.

Deep Sets (Zaheer et al., 2017) avoids this by applying a SHARED encoder
to every element independently, then aggregating with a permutation-invariant
pooling function (mean or max):

    V(S^x) = ρ( pool_n φ(s_n),  pool_m ψ(v_m),  g )

where:
    φ  = StationEncoder   shared MLP over each station
    ψ  = VehicleEncoder   shared MLP over each vehicle
    ρ  = ValueMLP         maps combined representation to a scalar
    g  = global context   temporal + system-wide features (passed through)

Permutation invariance holds because mean/max pooling commutes with
any reordering of the input rows.

─────────────────────────────────────────────────────────────────────────────
RELATIONSHIP TO THE LINEAR VFA
─────────────────────────────────────────────────────────────────────────────
The linear VFA computes:  V(S^x) = θᵀ φ(S^x)
This network computes:    V(S^x) = ρ( pool φ(s_n), pool ψ(v_m), g )

Both estimate the same quantity (post-decision state value) and are used
in the same way inside the rollout (as the terminal value estimator after
fast-forwarding the simulator H minutes). The difference is that the linear
VFA uses manually engineered features, while this model learns its own
intermediate representations.

─────────────────────────────────────────────────────────────────────────────
OUTPUT
─────────────────────────────────────────────────────────────────────────────
A scalar V(S^x) ∈ ℝ, unconstrained. The linear VFA clamped θ ≤ 0 because
its features were penalty-oriented. Here, the sign is learned from the
reward signal (which is negative: penalties for starvations and congestions).

─────────────────────────────────────────────────────────────────────────────
REFERENCES
─────────────────────────────────────────────────────────────────────────────
  Zaheer et al. (2017). "Deep Sets". NeurIPS.
  Mnih et al. (2015). "Human-level control through deep RL." (target network)
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Default size hyperparameters
#
# These are deliberately conservative. The bottleneck for this problem is
# the quality of the TD training signal, not model capacity. Scale up only
# after confirming the small model is clearly underfitting on learning curves.
# ─────────────────────────────────────────────────────────────────────────────

STATION_EMBED_DIM = 32   # StationEncoder output dimension
VEHICLE_EMBED_DIM = 16   # VehicleEncoder output dimension
VALUE_HIDDEN_DIM  = 64   # hidden dimension in the final ValueMLP


# ═════════════════════════════════════════════════════════════════════════════
# SUB-NETWORKS
# ═════════════════════════════════════════════════════════════════════════════

class StationEncoder(nn.Module):
    """
    Shared MLP applied independently to every station's feature vector.

    The same weights are reused for every station — the model learns a
    general notion of "what does this inventory pattern mean" rather than
    station-specific behavior. This is the φ function in Deep Sets.

    Input  : [N_stations × station_feature_dim]
    Output : [N_stations × embed_dim]

    After this, the caller pools over the N rows (mean pooling by default)
    to produce a single fixed-size station summary vector.
    """

    def __init__(self, input_dim: int, embed_dim: int = STATION_EMBED_DIM):
        super().__init__()
        # Two-layer MLP. ReLU activations are stable and straightforward.
        # We keep this small: the representational power we care about comes
        # from the ValueMLP that combines all branches, not from encoding
        # individual stations deeply.
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [N_stations × input_dim]  — one row per station
        Returns:
            embeddings : [N_stations × embed_dim]  — same weights, all rows
        """
        # nn.Linear broadcasts naturally over the batch/sequence dimension,
        # so this applies the same transformation to every station row.
        return self.net(x)


class VehicleEncoder(nn.Module):
    """
    Shared MLP applied independently to every vehicle's feature vector.

    Analogous to StationEncoder but for the service fleet. This is the ψ
    function in Deep Sets. With a single vehicle this degenerates to a
    plain MLP; with multiple vehicles the shared weights enforce the
    constraint that vehicles are exchangeable (same rules for all).

    Input  : [M_vehicles × vehicle_feature_dim]
    Output : [M_vehicles × embed_dim]
    """

    def __init__(self, input_dim: int, embed_dim: int = VEHICLE_EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [M_vehicles × input_dim]
        Returns:
            embeddings : [M_vehicles × embed_dim]
        """
        return self.net(x)


class ValueMLP(nn.Module):
    """
    Final value head: combined state embedding → scalar V(S^x).

    Receives the concatenation of:
        mean-pooled station summary  [STATION_EMBED_DIM]
        mean-pooled vehicle summary  [VEHICLE_EMBED_DIM]
        global context vector        [GLOBAL_FEATURE_DIM]

    and maps this to a single real-valued estimate of future cost.
    This is the ρ function in Deep Sets.
    """

    def __init__(self, input_dim: int, hidden_dim: int = VALUE_HIDDEN_DIM):
        super().__init__()
        # Three-layer MLP with a progressive reduction in width.
        # The final layer is Linear(hidden_dim//2, 1) — no activation —
        # so the output is an unconstrained scalar.
        self.net = nn.Sequential(
            nn.Linear(input_dim,       hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,      hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),   # scalar output, unconstrained
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [combined_dim]  or  [batch × combined_dim]
        Returns:
            value : [1]  or  [batch × 1]
        """
        return self.net(x)


# ═════════════════════════════════════════════════════════════════════════════
# FULL MODEL
# ═════════════════════════════════════════════════════════════════════════════

class NNValueNetwork(nn.Module):
    """
    Full Deep Sets value network V(S^x) for post-decision state evaluation.

    This model is used in two roles:
      1. Online model   — the model being trained; gradients flow through it.
      2. Target network — a periodically frozen copy used to compute the TD
                          target (r + γ · V_target(S^x_next)).  Freezing the
                          target stabilizes training by preventing the target
                          from shifting at every gradient step (the "deadly
                          triad" problem in neural TD learning).

    Architecture:

        station_block  [N × 5] ──► StationEncoder ──► mean-pool ──► [32]
                                                                       │
        vehicle_block  [M × 5] ──► VehicleEncoder ──► mean-pool ──► [16]  ──► cat ──► [55]
                                                                       │
        global_context     [7] ──────────────────────────────────── [7]
                                                                       │
                                                                  ValueMLP
                                                                       │
                                                             scalar V(S^x)

    Args:
        station_feature_dim : must match STATION_FEATURE_DIM in nn_state_encoder.py
        vehicle_feature_dim : must match VEHICLE_FEATURE_DIM in nn_state_encoder.py
        global_feature_dim  : must match GLOBAL_FEATURE_DIM in nn_state_encoder.py
        station_embed_dim   : StationEncoder output size
        vehicle_embed_dim   : VehicleEncoder output size
        value_hidden_dim    : hidden size of ValueMLP
    """

    def __init__(
        self,
        station_feature_dim: int,
        vehicle_feature_dim: int,
        global_feature_dim:  int,
        station_embed_dim:   int = STATION_EMBED_DIM,
        vehicle_embed_dim:   int = VEHICLE_EMBED_DIM,
        value_hidden_dim:    int = VALUE_HIDDEN_DIM,
    ):
        super().__init__()

        self.station_encoder = StationEncoder(station_feature_dim, station_embed_dim)
        self.vehicle_encoder = VehicleEncoder(vehicle_feature_dim, vehicle_embed_dim)

        # The ValueMLP input is the concatenation of the three pooled representations.
        combined_dim = station_embed_dim + vehicle_embed_dim + global_feature_dim
        self.value_mlp = ValueMLP(combined_dim, value_hidden_dim)

        # Store dims so checkpoints can be verified for compatibility.
        self.station_feature_dim = station_feature_dim
        self.vehicle_feature_dim = vehicle_feature_dim
        self.global_feature_dim  = global_feature_dim

    def forward(
        self,
        station_block:  torch.Tensor,
        vehicle_block:  torch.Tensor,
        global_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute V(S^x) for one post-decision state.

        Args:
            station_block  : [N_stations × station_feature_dim]
            vehicle_block  : [M_vehicles × vehicle_feature_dim]
            global_context : [global_feature_dim]

        Returns:
            value : scalar tensor, shape [1].

        ── Station branch ────────────────────────────────────────────────────
        Apply shared encoder row-by-row, then mean-pool across stations.
        Mean pooling preserves the magnitude of the global signal: if every
        station is starved, the mean embedding is large. Max pooling would
        only capture the worst station and lose the breadth of the problem.
        ── Vehicle branch ────────────────────────────────────────────────────
        Same logic. Mean over vehicles gives the average fleet state.
        ── Combine ───────────────────────────────────────────────────────────
        Concatenate the two pooled summaries with the global context vector
        and pass through the ValueMLP to get the scalar estimate.
        """
        # --- Station branch ---
        station_embeddings = self.station_encoder(station_block)  # [N, station_embed_dim]
        station_summary    = station_embeddings.mean(dim=0)        # [station_embed_dim]

        # --- Vehicle branch ---
        vehicle_embeddings = self.vehicle_encoder(vehicle_block)   # [M, vehicle_embed_dim]
        vehicle_summary    = vehicle_embeddings.mean(dim=0)        # [vehicle_embed_dim]

        # --- Combine all three representations ---
        # cat produces [station_embed_dim + vehicle_embed_dim + global_feature_dim]
        combined = torch.cat([station_summary, vehicle_summary, global_context], dim=0)

        # --- Scalar value estimate ---
        value = self.value_mlp(combined)   # [1]
        return value


# ═════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═════════════════════════════════════════════════════════════════════════════

def build_nn_value_network() -> NNValueNetwork:
    """
    Construct an NNValueNetwork with input dimensions that match nn_state_encoder.

    This is the canonical way to instantiate the model. Importing the
    dimension constants directly from nn_state_encoder ensures that the
    model's expected input shapes always stay in sync with the encoder's
    output shapes — no manual counting.

    Returns:
        NNValueNetwork with random weights, ready for training.
    """
    from policies.sjovik_sund.NN.nn_state_encoder import (
        STATION_FEATURE_DIM,
        VEHICLE_FEATURE_DIM,
        GLOBAL_FEATURE_DIM,
    )
    return NNValueNetwork(
        station_feature_dim=STATION_FEATURE_DIM,
        vehicle_feature_dim=VEHICLE_FEATURE_DIM,
        global_feature_dim=GLOBAL_FEATURE_DIM,
    )
