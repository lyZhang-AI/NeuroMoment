import torch
from torch.nn import functional as F


def transport_anchor(distribution, times, delta):
    """Anchor t0 -> t0+delta: a_delta(tau) = a(tau+delta).

    Linear interpolation on token centers, zero outside source support.
    Returns a mask for common support. Fractional token offsets are supported.
    """
    if times.numel() < 2:
        raise ValueError("Equivariance requires at least two tokens")
    delta = torch.as_tensor(delta, device=times.device, dtype=times.dtype)
    if delta.ndim == 0:
        delta = delta.expand(distribution.shape[0])
    q = times[None] + delta[:, None]
    index = (q - times[0]) / (times[1] - times[0])
    lo = index.floor().long()
    hi = lo + 1
    frac = index - lo
    last = distribution.shape[-1] - 1
    warped = distribution.gather(1, lo.clamp(0, last)) * (1 - frac)
    warped = warped + distribution.gather(1, hi.clamp(0, last)) * frac
    mask = (q >= times[0] - 1e-6) & (q <= times[-1] + 1e-6)
    return warped * mask, mask


def common_distributions(original, shifted, times, delta):
    target, mask = transport_anchor(original, times, delta)
    a = shifted * mask
    b = target * mask
    if (a.sum(-1) <= 1e-12).any() or (b.sum(-1) <= 1e-12).any():
        raise ValueError("Empty common probability support for requested shift")
    return a / a.sum(-1, keepdim=True), b / b.sum(-1, keepdim=True)


def shift_js(original, shifted, times, delta):
    a, b = common_distributions(original, shifted, times, delta)
    mean = (a + b) / 2
    def kl(p):
        return (p * (p.clamp_min(1e-12).log() - mean.clamp_min(1e-12).log())).sum(-1)
    return ((kl(a) + kl(b)) / 2).mean()


def equivariance_ms(original, shifted, times, delta):
    a, b = common_distributions(original, shifted, times, delta)
    # One-dimensional Wasserstein distance, conditioned on the overlap.
    return (a.cumsum(-1) - b.cumsum(-1)).abs().sum(-1) * (times[1] - times[0]) * 1000


def objective(first, second, labels, embeddings, delta, temperature=0.07,
              lambda_shift=0.2, lambda_inv=0.1, alignment_views="original"):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    embeddings = F.normalize(embeddings, dim=-1)
    # Equation (10) supervises the original view; both-view supervision is optional.
    align = (F.cross_entropy(first["embedding"] @ embeddings.T / temperature, labels)
             + F.cross_entropy(second["embedding"] @ embeddings.T / temperature, labels)) / 2
    if alignment_views == "original":
        align = F.cross_entropy(first["embedding"] @ embeddings.T / temperature, labels)
    elif alignment_views != "both":
        raise ValueError("Unknown alignment view protocol")
    shift = shift_js(first["alpha"], second["alpha"], first["times"], delta)
    inv = (1 - (first["embedding"] * second["embedding"]).sum(-1)).mean()
    loss = align + lambda_shift * shift + lambda_inv * inv
    return loss, {"align": align.detach(), "shift": shift.detach(), "inv": inv.detach()}
