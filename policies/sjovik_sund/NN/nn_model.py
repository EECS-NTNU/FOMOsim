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

STATION_EMBED_DIM = 64   # StationEncoder output dimension
VEHICLE_EMBED_DIM = 16   # VehicleEncoder output dimension
VALUE_HIDDEN_DIM  = 128   # hidden dimension in the final ValueMLP


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


'''class ValueMLP(nn.Module):
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
        return self.net(x)'''

class ValueMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = None):
        super().__init__()
        
        # Default to the original funnel if nothing is passed
        if hidden_dims is None:
            hidden_dims = [VALUE_HIDDEN_DIM, VALUE_HIDDEN_DIM // 2]
            
        layers = []
        current_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(current_dim, h))
            layers.append(nn.ReLU())
            current_dim = h
            
        # Final unconstrained scalar output
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
# ═════════════════════════════════════════════════════════════════════════════
# ATTENTION POOLING
# ═════════════════════════════════════════════════════════════════════════════

class AttentionPool(nn.Module):
    """
    Learned weighted sum over a set of embeddings.

    score_i  = w · embedding_i          (single linear, no bias)
    weight_i = softmax(scores)
    summary  = Σ weight_i · embedding_i

    Replaces mean pooling so the network can focus on critical stations
    (e.g., near-starving) rather than weighting all equally.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, D] → [D]"""
        w = torch.softmax(self.score(x), dim=0)   # [N, 1]
        return (w * x).sum(dim=0)                  # [D]

    def forward_batch(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D] → [B, D]"""
        w = torch.softmax(self.score(x), dim=1)   # [B, N, 1]
        return (w * x).sum(dim=1)                  # [B, D]
    
   
# ═════════════════════════════════════════════════════════════════════════════
# Cross-Attention Pooling (Conditioned on Vehicle State)
# ═════════════════════════════════════════════════════════════════════════════
class CrossAttentionPool(nn.Module):
    """
    Learned weighted sum over stations, conditioned dynamically on the vehicle state.
    query = Vehicle Summary
    keys/values = Station Embeddings
    """
    def __init__(self, station_dim: int, vehicle_dim: int):
        super().__init__()
        # Projects the vehicle summary into the station dimension to compute a dot-product
        self.query_proj = nn.Linear(vehicle_dim, station_dim)

    def forward(self, station_emb: torch.Tensor, vehicle_summary: torch.Tensor) -> torch.Tensor:
        """
        station_emb: [N, station_dim]
        vehicle_summary: [vehicle_dim] (Already pooled)
        """
        # 1. Create a search query based on what the vehicle currently needs
        query = self.query_proj(vehicle_summary)  # [station_dim]
        
        # 2. Score stations based on how well they match the vehicle's query (Dot Product)
        scores = (station_emb * query).sum(dim=1) # [N]
        
        # 3. Softmax to get normalized attention weights
        w = torch.softmax(scores, dim=0)          # [N]
        
        # 4. Apply weights to create the final contextual summary
        return (w.unsqueeze(1) * station_emb).sum(dim=0) # [station_dim]
        
    def forward_batch(self, station_emb: torch.Tensor, vehicle_summary: torch.Tensor) -> torch.Tensor:
        """Batched version for _compute_td_loss"""
        # vehicle_summary: [B, vehicle_dim] -> [B, station_dim] -> [B, station_dim, 1]
        query = self.query_proj(vehicle_summary).unsqueeze(2) 
        
        # Matrix multiplication: [B, N, station_dim] @ [B, station_dim, 1] -> [B, N, 1]
        scores = torch.bmm(station_emb, query)
        w = torch.softmax(scores, dim=1) 
        
        return (w * station_emb).sum(dim=1) # [B, station_dim]


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

        station_block  [N × 8] ──► StationEncoder ──► attn+max-pool ──► [64]
                                                                          │
        vehicle_block  [M × 6] ──► VehicleEncoder ──► attn+max-pool ──► [32]  ──► cat ──► [104]
                                                                          │
        global_context     [8] ─────────────────────────────────────── [8]
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
        value_hidden_dims:   list = None,
    ):
        super().__init__()

        self.station_encoder = StationEncoder(station_feature_dim, station_embed_dim)
        self.vehicle_encoder = VehicleEncoder(vehicle_feature_dim, vehicle_embed_dim)
        #self.station_attn    = AttentionPool(station_embed_dim)
        self.vehicle_attn    = AttentionPool(vehicle_embed_dim)
        
        # NEW: Station pooling is now conditioned on the finalized vehicle summary
        vehicle_summary_dim = 2 * vehicle_embed_dim
        self.station_cross_attn = CrossAttentionPool(station_embed_dim, vehicle_summary_dim)

        # Dual pooling: attention + max concatenated → 2× each embed dim.
        # +2*station_feature_dim: destination raw bypass + max-deficit station raw bypass.
        combined_dim = (2 * station_embed_dim + 2 * vehicle_embed_dim
                        + global_feature_dim + 2 * station_feature_dim)
        self.value_mlp = ValueMLP(combined_dim, value_hidden_dims)

        # Store dims so checkpoints can be verified for compatibility.
        self.station_feature_dim = station_feature_dim
        self.vehicle_feature_dim = vehicle_feature_dim
        self.global_feature_dim  = global_feature_dim
        self.value_hidden_dims   = value_hidden_dims


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
        '''# --- Station branch ---
        station_embeddings = self.station_encoder(station_block)   # [N, station_embed_dim]
        station_summary = torch.cat([
            self.station_attn(station_embeddings),          # attention pool  ← revert: station_embeddings.mean(dim=0)
            station_embeddings.max(dim=0).values,
        ], dim=0)  # [2 * station_embed_dim]

        # --- Vehicle branch ---
        vehicle_embeddings = self.vehicle_encoder(vehicle_block)   # [M, vehicle_embed_dim]
        vehicle_summary = torch.cat([
            self.vehicle_attn(vehicle_embeddings),          # attention pool  ← revert: vehicle_embeddings.mean(dim=0)
            vehicle_embeddings.max(dim=0).values,
        ], dim=0)  # [2 * vehicle_embed_dim]

        # --- Combine all three representations ---
        combined = torch.cat([station_summary, vehicle_summary, global_context], dim=0)

        # --- Scalar value estimate ---
        value = self.value_mlp(combined)   # [1]
        return value'''
        
        # --- 1. Vehicle branch (Process FIRST) ---
        vehicle_embeddings = self.vehicle_encoder(vehicle_block)
        vehicle_summary = torch.cat([
            self.vehicle_attn(vehicle_embeddings),
            vehicle_embeddings.max(dim=0).values,
        ], dim=0)

        # --- 2. Station branch (Conditioned on Vehicle) ---
        station_embeddings = self.station_encoder(station_block)
        station_summary = torch.cat([
            self.station_cross_attn(station_embeddings, vehicle_summary),
            station_embeddings.max(dim=0).values,
        ], dim=0)

        # --- 3. Destination spotlight: raw features bypass pooling ---
        dest_mask = station_block[:, -1] > 0.5
        if dest_mask.any():
            dest_raw = station_block[dest_mask][0]
        else:
            dest_raw = torch.zeros(self.station_feature_dim, device=station_block.device)

        # --- 3b. Max-deficit spotlight: worst non-destination station bypasses pooling ---
        # CrossAttentionPool is vehicle-conditioned and may down-weight a critically starved
        # station that is far from the destination. This bypass guarantees the NN always
        # sees the worst station regardless of attention routing.
        non_dest_mask = station_block[:, -1] < 0.5
        if non_dest_mask.any():
            non_dest = station_block[non_dest_mask]
            max_deficit_raw = non_dest[non_dest[:, 5].argmax()]  # deficit_ratio at index 5
        else:
            max_deficit_raw = torch.zeros(self.station_feature_dim, device=station_block.device)

        # --- 4. Combine ---
        combined = torch.cat([station_summary, vehicle_summary, global_context, dest_raw, max_deficit_raw], dim=0)
        value = self.value_mlp(combined)
        return value

    '''def forward_batch(
        self,
        station_blocks:  torch.Tensor,
        vehicle_blocks:  torch.Tensor,
        global_contexts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched forward pass over B states simultaneously.

        Used by _compute_td_loss to replace the Python loop over 128 samples
        with a single GPU kernel call — ~30-50x faster on MPS.

        Args:
            station_blocks  : [B, N_stations, station_feature_dim]
            vehicle_blocks  : [B, M_vehicles, vehicle_feature_dim]
            global_contexts : [B, global_feature_dim]

        Returns:
            values : [B, 1]

        nn.Linear broadcasts over leading batch dimensions naturally, so the
        shared-weight station/vehicle encoders apply identically to each of
        the B states without any explicit looping.
        """
        # [B, N, embed_dim] → attention+max over stations → [B, 2*embed_dim]
        st_emb = self.station_encoder(station_blocks)
        station_summary = torch.cat([
            self.station_attn.forward_batch(st_emb),   # ← revert: st_emb.mean(dim=1)
            st_emb.max(dim=1).values,
        ], dim=1)

        # [B, M, embed_dim] → attention+max over vehicles → [B, 2*embed_dim]
        vh_emb = self.vehicle_encoder(vehicle_blocks)
        vehicle_summary = torch.cat([
            self.vehicle_attn.forward_batch(vh_emb),   # ← revert: vh_emb.mean(dim=1)
            vh_emb.max(dim=1).values,
        ], dim=1)

        # [B, combined_dim]
        combined = torch.cat([station_summary, vehicle_summary, global_contexts], dim=1)

        return self.value_mlp(combined)   # [B, 1]'''
        
    def forward_batch(
        self,
        station_blocks:  torch.Tensor,
        vehicle_blocks:  torch.Tensor,
        global_contexts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched forward pass over B states simultaneously.

        Used by _compute_td_loss to replace the Python loop over 128 samples
        with a single GPU kernel call — ~30-50x faster on MPS.

        Args:
            station_blocks  : [B, N_stations, station_feature_dim]
            vehicle_blocks  : [B, M_vehicles, vehicle_feature_dim]
            global_contexts : [B, global_feature_dim]

        Returns:
            values : [B, 1]
        """
        B = station_blocks.shape[0]

        # --- 1. Vehicle branch (Process FIRST) ---
        vh_emb = self.vehicle_encoder(vehicle_blocks)
        vehicle_summary = torch.cat([
            self.vehicle_attn.forward_batch(vh_emb),
            vh_emb.max(dim=1).values,
        ], dim=1)   # [B, 2*vehicle_embed_dim]

        # --- 2. Station branch (Conditioned on Vehicle) ---
        st_emb = self.station_encoder(station_blocks)
        station_summary = torch.cat([
            self.station_cross_attn.forward_batch(st_emb, vehicle_summary),
            st_emb.max(dim=1).values,
        ], dim=1)   # [B, 2*station_embed_dim]

        # --- 3. Destination spotlight ---
        dest_idx = station_blocks[:, :, -1].argmax(dim=1)   # [B]
        dest_raw = station_blocks[torch.arange(B, device=station_blocks.device), dest_idx, :]  # [B, station_feature_dim]

        # --- 3b. Max-deficit spotlight (non-destination) ---
        deficit_scores = station_blocks[:, :, 5].clone()  # deficit_ratio at index 5, [B, N]
        deficit_scores[station_blocks[:, :, -1] > 0.5] = -float('inf')  # mask out destination
        max_def_idx = deficit_scores.argmax(dim=1)   # [B]
        max_deficit_raw = station_blocks[torch.arange(B, device=station_blocks.device), max_def_idx, :]  # [B, station_feature_dim]

        # --- 4. Combine ---
        combined = torch.cat([station_summary, vehicle_summary, global_contexts, dest_raw, max_deficit_raw], dim=1)

        return self.value_mlp(combined)   # [B, 1]


# ═════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═════════════════════════════════════════════════════════════════════════════

'''def build_nn_value_network() -> NNValueNetwork:
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
    )'''
    
def build_nn_value_network(value_hidden_dims: list = None) -> NNValueNetwork:
    from policies.sjovik_sund.NN.nn_state_encoder import (
        STATION_FEATURE_DIM, VEHICLE_FEATURE_DIM, GLOBAL_FEATURE_DIM,
    )
    return NNValueNetwork(
        station_feature_dim=STATION_FEATURE_DIM,
        vehicle_feature_dim=VEHICLE_FEATURE_DIM,
        global_feature_dim=GLOBAL_FEATURE_DIM,
        value_hidden_dims=value_hidden_dims,
    )
