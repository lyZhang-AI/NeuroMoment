from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    channels: int = 8
    dim: int = 32
    text_dim: int = 32
    sample_rate: float = 128.0
    start: float = -0.2
    end: float = 1.0
    stride: int = 4
    mode: str = "adaptive"  # adaptive, attention, fixed_prior, global_offset, fixed_window, mil
    sigma: float = 0.08
    beta: float = 1.0
    latency_min: float = 0.0
    latency_max: float = 1.0
    latency_bins: int = 26
    fixed_center: float = 0.3
    fine: bool = False
    local_width: float = 0.16
    moments: int = 3
    nms_interval: float = 0.12
    content_temperature: float = 1.0
    selection_temperature: float = 1.0
    selection: str = "straight_through"
    fine_layers: int = 1
    fusion: str = "soft"  # soft (Stage I), all, global, local


class LocalTransformer(nn.Module):
    """Shared within-window self-attention; temporal refinement follows it."""
    def __init__(self, dim, layers):
        super().__init__()
        block = nn.TransformerEncoderLayer(dim, 4, 2*dim, dropout=0., batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)

    def forward(self, x):
        return self.encoder(x.transpose(1, 2)).transpose(1, 2)


class NeuroMoment(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        c = self.config = config
        if c.dim % 4 or c.stride < 1 or c.sigma <= 0 or c.moments < 1 or c.local_width <= 0 or c.sample_rate <= 0:
            raise ValueError("dim must be divisible by 4; stride, sigma, moments must be positive")
        if c.mode not in {"adaptive", "attention", "fixed_prior", "global_offset", "fixed_window", "mil", "fixed_gaussian", "temporal_pool"}:
            raise ValueError(f"Unknown mode {c.mode}")
        if c.mode == "fixed_window" and c.fine:
            raise ValueError("Strict fixed-window baseline does not use a fine branch")
        if c.fusion not in {"soft", "all", "global", "local"} or (c.fusion == "local" and not c.fine):
            raise ValueError("local fusion requires the fine encoder")
        if c.content_temperature <= 0 or c.selection_temperature <= 0 or c.nms_interval <= 0 or c.fine_layers < 1:
            raise ValueError("Invalid content/selection temperature, NMS interval or fine depth")
        if c.selection not in {"straight_through", "hard"}:
            raise ValueError("Unknown moment selection mechanism")
        self.temporal = nn.Sequential(
            nn.Conv1d(c.channels, c.dim, 9, padding=4), nn.GELU(),
            nn.Conv1d(c.dim, c.dim, 5, padding=2, groups=c.dim), nn.GELU(),
            nn.AvgPool1d(c.stride, c.stride),
        )
        layer = nn.TransformerEncoderLayer(c.dim, 4, c.dim * 2, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.pool_score = nn.Linear(c.dim, 1)
        self.latency = nn.Sequential(nn.Linear(c.dim, c.dim), nn.GELU(), nn.Linear(c.dim, c.latency_bins))
        self.shared_prior = nn.Parameter(torch.zeros(c.latency_bins))
        self.offset = nn.Parameter(torch.tensor(0.0))
        self.query = nn.Sequential(nn.Linear(c.dim, c.dim), nn.GELU(), nn.Linear(c.dim, c.dim))
        self.key = nn.Linear(c.dim, c.dim)
        self.register_buffer("delays", torch.linspace(c.latency_min, c.latency_max, c.latency_bins))
        self.fine_encoder = None
        if c.fine:
            self.fine_encoder = nn.Sequential(
                nn.Conv1d(c.channels, c.dim, 7, padding=3), nn.GELU(),
                LocalTransformer(c.dim, c.fine_layers),
                nn.Conv1d(c.dim, c.dim, 5, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(1),
            )
        self.fusion = nn.Sequential(nn.Linear(c.dim * (3 if c.fine else 2), c.dim), nn.GELU())
        self.project = nn.Linear(c.dim, c.text_dim)

    def coordinates(self, n, device, dtype):
        c = self.config
        length = n // c.stride
        # Conv stride is 1, symmetric padding; average-pool centers are exact.
        return c.start + (torch.arange(length, device=device, dtype=dtype) * c.stride
                          + (c.stride - 1) / 2) / c.sample_rate

    def position(self, times):
        frequencies = torch.exp(torch.linspace(0, math.log(30), self.config.dim // 2,
                                                device=times.device, dtype=times.dtype))
        phase = times[..., None] * frequencies
        return torch.cat([phase.sin(), phase.cos()], -1)

    def select(self, alpha, times, n):
        """Deterministic forward indices; surrogate gradients are supplied in local_crops."""
        c = self.config
        width = max(1, round(c.local_width * c.sample_rate))
        if width > n:
            raise ValueError("local window exceeds input")
        centers = torch.round((times - c.start) * c.sample_rate).long()
        legal = (centers - width // 2 >= 0) & (centers - width // 2 + width <= n)
        scores = alpha.detach().masked_fill(~legal[None], -torch.inf).clone()
        selected = []
        for _ in range(c.moments):
            if not torch.isfinite(scores).any(dim=1).all():
                raise ValueError("Too many NMS moments for the input and local window width")
            idx = scores.argmax(-1)
            selected.append(idx)
            distance = (times[None] - times[idx, None]).abs()
            scores = scores.masked_fill(distance < c.nms_interval, -torch.inf)
        return torch.stack(selected, 1), centers, width

    def local_crops(self, x, alpha, times, indices, centers, width):
        c = self.config
        starts = centers - width // 2
        legal = (starts >= 0) & (starts + width <= x.shape[-1])
        legal_ids = legal.nonzero().flatten()
        available = legal[None].expand(len(x), -1).clone()
        candidate = x.unfold(-1, width, 1).index_select(2, starts[legal])
        selections = []
        for j in range(c.moments):
            index = indices[:, j]
            hard = F.one_hot(index, len(times)).to(alpha.dtype)
            if self.training and c.selection == "straight_through":
                logits = alpha.clamp_min(1e-12).log() / c.selection_temperature
                soft = logits.masked_fill(~available, -torch.inf).softmax(-1)
                weight = hard + (soft - soft.detach())
            else:
                weight = hard
            selections.append(weight[:, legal_ids])
            available = available & ((times[None] - times[index, None]).abs() >= c.nms_interval)
        weights = torch.stack(selections, 1)
        crops = torch.einsum("bkl,bclw->bkcw", weights, candidate)
        u = self.fine_encoder(crops.reshape(-1, c.channels, width)).squeeze(-1)
        u = u.reshape(len(x), c.moments, c.dim)
        return u + torch.einsum("bkl,ld->bkd", weights, self.position(times[legal]))

    def forward(self, x, prior_override=None):
        c = self.config
        crop_offset = 0
        if c.mode == "fixed_window":
            crop_offset = round((c.fixed_center - c.local_width / 2 - c.start) * c.sample_rate)
            width = round(c.local_width * c.sample_rate)
            if crop_offset < 0 or crop_offset + width > x.shape[-1] or width < c.stride:
                raise ValueError("Fixed window outside input or too short")
            x = x[..., crop_offset:crop_offset+width]
        times = self.coordinates(x.shape[-1], x.device, x.dtype) + crop_offset / c.sample_rate
        raw_h = self.temporal(x).transpose(1, 2)
        h = self.context(raw_h + self.position(times)[None])
        g = (h * self.pool_score(h).softmax(1)).sum(1)
        p = self.latency(g).softmax(-1)
        if c.mode == "fixed_prior":
            p = self.shared_prior.softmax(-1).expand(x.shape[0], -1)
        if prior_override is not None:
            if isinstance(prior_override, torch.Tensor):
                override = prior_override.to(device=p.device, dtype=p.dtype)
                if override.shape != p.shape or not torch.isfinite(override).all() or (override < 0).any():
                    raise ValueError("Invalid prior override")
                if not torch.allclose(override.sum(-1), torch.ones_like(override[:, 0]), atol=1e-4):
                    raise ValueError("Prior override must sum to one")
                p = override
            elif prior_override == "shuffled":
                if x.shape[0] < 2:
                    raise ValueError("Shuffled prior requires batch size >= 2")
                p = p.roll(1, 0)
            else:
                raise ValueError("Unknown prior intervention")
        log_kernel = -(times[:, None] - self.delays[None]) ** 2 / (2 * c.sigma ** 2)
        log_prior = torch.logsumexp(p.clamp_min(1e-12).log()[:, None] + log_kernel[None], -1)
        if c.mode == "fixed_gaussian":
            log_prior = (-(times-c.fixed_center)**2/(2*c.sigma**2))[None].expand(len(x), -1)
        if c.mode == "global_offset":
            center = c.latency_min + (c.latency_max - c.latency_min) * self.offset.sigmoid()
            log_prior = (-(times - center) ** 2 / (2 * c.sigma ** 2))[None].expand(x.shape[0], -1)
        score = (self.key(h) * self.query(g)[:, None]).sum(-1) / math.sqrt(c.dim)
        logits = score / c.content_temperature + c.beta * log_prior
        if c.mode in {"attention", "mil"}:
            logits = score / c.content_temperature
        if c.mode in {"fixed_window", "temporal_pool"}:
            logits = torch.zeros_like(score)
        alpha = logits.softmax(-1)
        uc = (alpha[..., None] * h).sum(1)
        if c.mode == "mil":
            # Matched-backbone sliding token-bag MIL; overlapping 3-token instances.
            bags = F.avg_pool1d(h.transpose(1, 2), 3, 1, 1).transpose(1, 2)
            bag_score = (self.key(bags) * self.query(g)[:, None]).sum(-1) / math.sqrt(c.dim)
            alpha = bag_score.softmax(-1)
            uc = (alpha[..., None] * bags).sum(1)
        parts = [g, uc]
        indices = None
        if c.fine:
            indices, centers, width = self.select(alpha, times, x.shape[-1])
            u = self.local_crops(x, alpha, times, indices, centers, width)
            parts.append(u.mean(1))
        if c.fusion == "soft":
            z = uc
        elif c.fusion == "global":
            z = g
        elif c.fusion == "local":
            z = parts[-1]
        else:
            z = self.fusion(torch.cat(parts, -1))
        return {"embedding": F.normalize(self.project(z), dim=-1), "alpha": alpha,
                "prior": p, "log_prior": log_prior, "score": score,
                "times": times, "indices": indices}

    def config_dict(self):
        return asdict(self.config)
