from __future__ import annotations

import time

import torch
from torch import nn


def aggregate_nodes_to_balls(
    node_features: torch.Tensor,
    node_to_ball: torch.Tensor,
    num_balls: int,
) -> torch.Tensor:
    """Mean-pool node representations within each granular ball."""
    ball_features = node_features.new_zeros(
        node_features.shape[0], num_balls, node_features.shape[2]
    )
    index = node_to_ball.view(1, -1, 1).expand_as(node_features)
    ball_features.scatter_add_(1, index, node_features)
    ball_sizes = (
        torch.bincount(node_to_ball, minlength=num_balls)
        .to(node_features)
        .view(1, -1, 1)
        .clamp_min(1)
    )
    return ball_features / ball_sizes


def project_balls_to_nodes(
    ball_features: torch.Tensor,
    node_to_ball: torch.Tensor,
) -> torch.Tensor:
    """Broadcast each granular-ball representation to its member nodes."""
    return ball_features[:, node_to_ball, :]


class DirectedDiffusionConv(nn.Module):
    """Directed diffusion convolution using forward and reverse random walks."""

    def __init__(self, in_dim: int, out_dim: int, diffusion_steps: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if diffusion_steps < 1:
            raise ValueError(f"diffusion_steps must be positive, got {diffusion_steps}")
        self.diffusion_steps = diffusion_steps
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_dim * (1 + 2 * diffusion_steps), out_dim)

    @staticmethod
    def _row_normalize(A: torch.Tensor, batch_size: int, num_nodes: int,
                       dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if A.ndim == 2:
            if A.shape != (num_nodes, num_nodes):
                raise ValueError(f"A must have shape {(num_nodes, num_nodes)}, got {tuple(A.shape)}")
            A_batch = A.unsqueeze(0).expand(batch_size, -1, -1)
        elif A.ndim == 3:
            if A.shape[-2:] != (num_nodes, num_nodes):
                raise ValueError(f"A must end with shape {(num_nodes, num_nodes)}, got {tuple(A.shape)}")
            A_batch = A.expand(batch_size, -1, -1) if A.shape[0] == 1 else A
            if A_batch.shape[0] != batch_size:
                raise ValueError(f"A batch dimension must be 1 or B={batch_size}, got {A.shape[0]}")
        else:
            raise ValueError(f"A must have shape [N,N] or [B,N,N], got {tuple(A.shape)}")
        A_batch = A_batch.to(device=device, dtype=dtype)
        return A_batch / A_batch.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, N, d], got {tuple(x.shape)}")
        batch_size, num_nodes, _ = x.shape
        P_fwd = self._row_normalize(A, batch_size, num_nodes, x.dtype, x.device)
        P_rev = self._row_normalize(A.transpose(-1, -2), batch_size, num_nodes, x.dtype, x.device)
        supports = [x]
        h_fwd = h_rev = x
        for _ in range(self.diffusion_steps):
            h_fwd = torch.bmm(P_fwd, h_fwd)
            h_rev = torch.bmm(P_rev, h_rev)
            supports.extend([h_fwd, h_rev])
        return self.linear(self.dropout(torch.cat(supports, dim=-1)))


class GBSTGNNDiffusion(nn.Module):
    """Adaptive-ball GB-STGNN using mean-pooled ball representations.

    This variant keeps the original model interface, but replaces the
    hydro-distance softmax aggregation:

        h_ball = sum_i softmax(-d_hydro_i) * h_i

    with simple arithmetic mean pooling inside each granular ball:

        h_ball = mean_{i in ball} h_i

    When use_masked_ball_pool is enabled, the mean is replaced by a mask-weighted
    average:

        h_ball = sum_i M_i h_i / sum_i M_i
    """

    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        hidden_dim: int,
        num_gru_layers: int,
        num_ball_diffusion_layers: int,
        diffusion_steps: int,
        horizon_dim: int,
        dropout: float,
        use_missing_mask_channel: bool = False,
        use_masked_ball_pool: bool = False,
    ) -> None:
        super().__init__()
        if num_ball_diffusion_layers < 1:
            raise ValueError("num_ball_diffusion_layers must be at least 1 for diffusion model.")
        self.use_missing_mask_channel = use_missing_mask_channel
        self.use_masked_ball_pool = use_masked_ball_pool
        gru_dropout = dropout if num_gru_layers > 1 else 0.0
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_gru_layers, batch_first=True, dropout=gru_dropout)
        self.ball_diffusion_layers = nn.ModuleList(
            [
                DirectedDiffusionConv(
                    hidden_dim,
                    hidden_dim,
                    diffusion_steps,
                    dropout=dropout,
                )
                for _ in range(num_ball_diffusion_layers)
            ]
        )
        self.activation = nn.ReLU()
        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3 + static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon_dim),
        )
        self.profile_graph_propagation = False
        self.graph_propagation_seconds = 0.0
        self.graph_propagation_calls = 0

    def reset_graph_propagation_timer(self) -> None:
        self.graph_propagation_seconds = 0.0
        self.graph_propagation_calls = 0

    def set_graph_propagation_profiling(self, enabled: bool) -> None:
        self.profile_graph_propagation = bool(enabled)

    def graph_propagation_profile(self) -> dict[str, float | int]:
        return {
            "seconds": float(self.graph_propagation_seconds),
            "calls": int(self.graph_propagation_calls),
        }

    @staticmethod
    def aggregate_nodes_to_balls_masked_mean(
        H_node: torch.Tensor,
        node_to_ball: torch.Tensor,
        node_weight: torch.Tensor,
        num_balls: int,
    ) -> torch.Tensor:
        if H_node.ndim != 3:
            raise ValueError(f"H_node must have shape [B, N, d], got {tuple(H_node.shape)}")
        if node_weight.ndim != 2:
            raise ValueError(f"node_weight must have shape [B, N], got {tuple(node_weight.shape)}")
        batch_size, num_nodes, hidden_dim = H_node.shape
        if node_weight.shape != (batch_size, num_nodes):
            raise ValueError(f"node_weight must have shape {(batch_size, num_nodes)}, got {tuple(node_weight.shape)}")

        weight = node_weight.to(device=H_node.device, dtype=H_node.dtype).clamp_min(0.0)
        H_ball = H_node.new_zeros(batch_size, num_balls, hidden_dim)
        denom = H_node.new_zeros(batch_size, num_balls, 1)
        expand_index = node_to_ball.view(1, num_nodes, 1).expand(batch_size, -1, hidden_dim)
        H_ball.scatter_add_(1, expand_index, H_node * weight.unsqueeze(-1))
        denom_index = node_to_ball.view(1, num_nodes, 1).expand(batch_size, -1, 1)
        denom.scatter_add_(1, denom_index, weight.unsqueeze(-1))

        mean_ball = aggregate_nodes_to_balls(H_node, node_to_ball, num_balls)
        return torch.where(denom > 0.0, H_ball / denom.clamp_min(1e-8), mean_ball)

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor,
        node_to_ball: torch.Tensor,
        hydro_dist_to_ball: torch.Tensor,
        A_ball: torch.Tensor,
        ball_quality: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B, L, N, F], got {tuple(x.shape)}")
        batch_size, seq_len, num_nodes, feature_dim = x.shape
        if static.ndim == 2:
            static_batch = static.unsqueeze(0).expand(batch_size, -1, -1)
        elif static.ndim == 3:
            static_batch = static
        else:
            raise ValueError(f"static must have shape [N,S] or [B,N,S], got {tuple(static.shape)}")
        if static_batch.shape[:2] != (batch_size, num_nodes):
            raise ValueError(
                f"static station dimensions do not match x: {tuple(static_batch.shape)} vs B,N={(batch_size, num_nodes)}"
            )

        node_to_ball = node_to_ball.to(device=x.device, dtype=torch.long)
        _ = hydro_dist_to_ball
        num_balls = int(node_to_ball.max().item()) + 1
        ball_quality = ball_quality.to(device=x.device, dtype=x.dtype)
        if ball_quality.shape != (num_balls,):
            raise ValueError(
                f"ball_quality must have shape {(num_balls,)}, got {tuple(ball_quality.shape)}"
            )
        if not torch.isfinite(ball_quality).all() or torch.any(ball_quality < 0):
            raise ValueError("ball_quality must contain finite non-negative values")
        quality_outer = ball_quality[:, None] * ball_quality[None, :]
        if A_ball.ndim == 2:
            if A_ball.shape != (num_balls, num_balls):
                raise ValueError(
                    f"A_ball must have shape {(num_balls, num_balls)}, got {tuple(A_ball.shape)}"
                )
            quality_weighted_A_ball = A_ball.to(
                device=x.device, dtype=x.dtype
            ) * quality_outer
        elif A_ball.ndim == 3:
            if A_ball.shape[-2:] != (num_balls, num_balls):
                raise ValueError(
                    f"A_ball must end with shape {(num_balls, num_balls)}, got {tuple(A_ball.shape)}"
                )
            quality_weighted_A_ball = A_ball.to(
                device=x.device, dtype=x.dtype
            ) * quality_outer.unsqueeze(0)
        else:
            raise ValueError(
                f"A_ball must have shape [M,M] or [B,M,M], got {tuple(A_ball.shape)}"
            )

        x_flat = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, feature_dim)
        site_confidence = None
        if self.use_missing_mask_channel:
            missing_mask = x[..., -1].clamp(0.0, 1.0)
            site_confidence = missing_mask.mean(dim=1)
        _, hidden = self.gru(x_flat)
        h_node = hidden[-1].reshape(batch_size, num_nodes, -1)

        if self.use_masked_ball_pool:
            if site_confidence is None:
                raise ValueError("use_masked_ball_pool=True requires use_missing_mask_channel=True.")
            h_ball_local = self.aggregate_nodes_to_balls_masked_mean(h_node, node_to_ball, site_confidence, num_balls)
        else:
            h_ball_local = aggregate_nodes_to_balls(h_node, node_to_ball, num_balls)
        h_ball_diff = h_ball_local
        profile_graph = self.profile_graph_propagation
        if profile_graph and x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
        graph_start = time.perf_counter() if profile_graph else 0.0
        for layer in self.ball_diffusion_layers:
            h_ball_diff = self.activation(
                h_ball_diff + layer(h_ball_diff, quality_weighted_A_ball)
            )
        if profile_graph:
            if x.device.type == "cuda":
                torch.cuda.synchronize(x.device)
            self.graph_propagation_seconds += time.perf_counter() - graph_start
            self.graph_propagation_calls += 1

        h_ball_local_to_node = project_balls_to_nodes(h_ball_local, node_to_ball)
        h_ball_diff_to_node = project_balls_to_nodes(h_ball_diff, node_to_ball)
        h_fused = torch.cat([h_node, h_ball_local_to_node, h_ball_diff_to_node, static_batch.to(h_node.dtype)], dim=-1)
        return self.decoder(h_fused)
