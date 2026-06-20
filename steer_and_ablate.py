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


_TOKEN_SCOPES = ("all", "response", "prompt")


def resolve_token_scope(mode, token_scope):
    """Resolve token scope: None -> 'response' (steer) / 'all' (ablate)."""
    if token_scope is None:
        return "response" if mode == "steer" else "all"
    if token_scope in _TOKEN_SCOPES:
        return token_scope
    raise ValueError(f"token_scope must be one of {_TOKEN_SCOPES} or None, "
                     f"got {token_scope!r}")


def should_apply(token_scope, seq_len):
    """Whether to intervene given the forward's sequence length.

    Under KV-cache generation: prefill has seq_len > 1, decode has seq_len == 1.
    """
    if token_scope == "all":
        return True
    if token_scope == "response":
        return seq_len == 1
    return seq_len > 1  # "prompt"
