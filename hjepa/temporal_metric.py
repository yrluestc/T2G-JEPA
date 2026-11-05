"""Goal-conditioned temporal metric for long-horizon latent planning.

The metric is trained on trajectory latents to approximate "how many model
steps remain from ``z_a`` to ``z_b``". It is a Euclidean metric in a learned
embedding space, so it satisfies the triangle inequality by construction:

    d(a, b) = scale * || f(a) - f(b) ||

This is a non-hierarchical alternative to goal-space Euclidean distance: the
same flat LeWM latent is used, but the cost is a learned time-to-go.
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class TemporalMetric(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim=256,
        depth=2,
        dropout=0.0,
        eps=1e-6,
        init_scale=1.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)

        layers = []
        in_dim = self.dim
        for _ in range(max(int(depth), 1)):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        self.encoder = nn.Sequential(*layers)

        init_scale = max(float(init_scale), 1e-4)
        self.raw_scale = nn.Parameter(
            torch.tensor(math.log(math.expm1(init_scale)))
        )

    def embed(self, z):
        return self.encoder(z)

    def scale(self):
        return F.softplus(self.raw_scale)

    def forward(self, z_a, z_b):
        ea = self.embed(z_a)
        eb = self.embed(z_b)
        dist = (ea - eb).norm(dim=-1)
        return self.scale() * dist.clamp_min(self.eps)

    def pairwise(self, z_a, z_b):
        """Pairwise distances for ``z_a (B,N,D)`` and ``z_b (B,M,D)``."""
        ea = self.embed(z_a)
        eb = self.embed(z_b)
        if ea.dim() == 2 and eb.dim() == 2:  # (N,D), (M,D)
            return (ea[:, None, :] - eb[None, :, :]).norm(dim=-1)
        if ea.dim() == 3 and eb.dim() == 3:  # (B,N,D), (B,M,D)
            return (ea[:, :, None, :] - eb[:, None, :, :]).norm(dim=-1)
        raise ValueError(
            f"unsupported pairwise shapes: {tuple(ea.shape)}, {tuple(eb.shape)}"
        )


def make_temporal_pairs(
    emb, max_steps=15, min_steps=1, gap=2, generator=None,
    *, sampling="legacy", triples_per_clip=1,
):
    """Sample ``(anchor, near, far)`` latents from trajectories.

    ``near`` is ``k_near`` steps after the anchor and ``far`` is ``k_far``
    steps after it, with ``k_far >= k_near + gap``. The returned latents are
    aligned with their time offsets. Results are flattened to (B * P, D),
    ordered by clip, where P is triples_per_clip (sampling with replacement).
    ``uniform_far`` samples the far offset uniformly, then a legal near offset
    and anchor. It does not make anchors/all possible triples uniform.
    ``legacy`` preserves the original sampling distribution and defaults.
    """
    B, T, D = emb.shape
    max_k = min(int(max_steps), T - 1)
    min_k = max(int(min_steps), 1)
    gap = max(int(gap), 1)
    if max_k < min_k + gap:
        raise ValueError(
            f"trajectory too short for temporal pairs: T={T}, "
            f"max_steps={max_steps}, gap={gap}"
        )

    device = emb.device
    count = int(triples_per_clip)
    if count < 1:
        raise ValueError("triples_per_clip must be positive")
    shape = (B * count,)
    if sampling == "uniform_far":
        k_far = torch.randint(
            min_k + gap, max_k + 1, shape, device=device, generator=generator
        )
        # Per-sample inclusive near range: [min_k, k_far - gap]. No clamping.
        k_near = min_k + (
            torch.rand(shape, device=device, generator=generator)
            * (k_far - gap - min_k + 1)
        ).long()
    elif sampling == "legacy":
        k_near = torch.randint(
            min_k, max_k - gap + 1, shape, device=device, generator=generator
        )
        extra = torch.randint(
            gap, max_k + 1, shape, device=device, generator=generator
        )
        k_far = (k_near + extra).clamp_max(max_k)
    else:
        raise ValueError(f"unknown temporal sampling mode: {sampling!r}")

    # t + k_far <= T - 1
    max_t = T - 1 - k_far
    t = (
        torch.rand(shape, device=device, generator=generator) * (max_t + 1)
    ).long()

    b = torch.arange(B, device=device).repeat_interleave(count)
    return {
        "anchor": emb[b, t],
        "near": emb[b, t + k_near],
        "far": emb[b, t + k_far],
        "k_near": k_near,
        "k_far": k_far,
        "t": t,
        "batch_indices": b,
    }


def temporal_metric_loss(
    d_near,
    d_far,
    margin=1.0,
    d_negative=None,
    negative_weight=0.0,
):
    """Ranking loss: near states should be closer than far states."""
    rank = F.relu(margin + d_near - d_far)
    rank_loss = rank.mean()
    negative_loss = d_near.sum() * 0.0
    loss = rank_loss
    if d_negative is not None and negative_weight > 0.0:
        negative_loss = F.relu(margin + d_near - d_negative).mean()
        loss = loss + float(negative_weight) * negative_loss
    return {
        "loss": loss,
        "rank_loss": rank_loss.detach(),
        "negative_loss": negative_loss.detach(),
        "d_near_mean": d_near.detach().mean(),
        "d_far_mean": d_far.detach().mean(),
        "rank_accuracy": (d_far.detach() > d_near.detach()).float().mean(),
        "margin_accuracy": (d_far.detach() - d_near.detach() >= margin).float().mean(),
        "distance_gap_mean": (d_far.detach() - d_near.detach()).mean(),
    }


def load_temporal_metric(path, map_location="cuda"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"temporal metric checkpoint not found: {path}")
    metric = torch.load(path, map_location=map_location, weights_only=False)
    if not hasattr(metric, "embed"):
        raise RuntimeError(
            f"checkpoint at {path} is not a TemporalMetric; retrain it."
        )
    return metric.to(map_location).eval().requires_grad_(False)
