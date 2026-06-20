"""Generate model responses while steering or ablating the residual stream."""
import torch


def apply_intervention(h, u, mode, alpha):
    """Apply a steering or ablation intervention to hidden states ``h``.

    ``h``: [..., H]; ``u``: [H] unit direction. steer -> h + alpha*u;
    ablate -> h - (h.u) u (per position).
    """
    if mode == "steer":
        return h + alpha * u
    if mode == "ablate":
        proj = (h * u).sum(dim=-1, keepdim=True)
        return h - proj * u
    raise ValueError(f"mode must be 'steer' or 'ablate', got {mode!r}")
