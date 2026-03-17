from __future__ import annotations

import torch


def diagonal_gaussian_kl(
    mu_p: torch.Tensor,
    std_p: torch.Tensor,
    mu_q: torch.Tensor,
    std_q: torch.Tensor,
) -> torch.Tensor:
    std_p = std_p.clamp_min(1e-8)
    std_q = std_q.clamp_min(1e-8)
    var_p = std_p.pow(2)
    var_q = std_q.pow(2)
    log_ratio = torch.log(std_q) - torch.log(std_p)
    squared_diff = (mu_p - mu_q).pow(2)
    return (log_ratio + (var_p + squared_diff) / (2.0 * var_q) - 0.5).sum(dim=-1)


def role_identifiability_loss(
    role_mu: torch.Tensor,
    role_std: torch.Tensor,
    traj_mu: torch.Tensor,
    traj_std: torch.Tensor,
) -> torch.Tensor:
    return diagonal_gaussian_kl(role_mu, role_std, traj_mu, traj_std).mean()


def role_variance_floor_loss(role_std: torch.Tensor, sigma_floor: float) -> torch.Tensor:
    if sigma_floor < 0.0:
        raise ValueError("sigma_floor must be non-negative.")
    return torch.relu(role_std.clamp_min(1e-8).new_tensor(float(sigma_floor)) - role_std).pow(2).mean()


def role_posterior_diagnostics(
    role_mu: torch.Tensor,
    role_std: torch.Tensor,
    sigma_floor: float,
) -> dict[str, torch.Tensor]:
    if role_mu.shape != role_std.shape:
        raise ValueError("role_mu and role_std must have matching shapes.")
    flat_mu = role_mu.reshape(-1, role_mu.shape[-1])
    flat_std = role_std.reshape(-1, role_std.shape[-1]).clamp_min(1e-8)
    return {
        "role_mu_var_per_dim": flat_mu.var(dim=0, unbiased=False),
        "role_sigma_mean_per_dim": flat_std.mean(dim=0),
        "near_zero_sigma_fraction": (flat_std < sigma_floor).float().mean(),
    }


def role_diversity_loss(role_mu: torch.Tensor) -> torch.Tensor:
    if role_mu.dim() < 2:
        raise ValueError("role_mu must have at least two dimensions.")
    if role_mu.shape[0] < 2:
        return role_mu.new_tensor(0.0)
    pairwise_sq_dist = torch.cdist(role_mu, role_mu, p=2).pow(2)
    mask = ~torch.eye(role_mu.shape[0], dtype=torch.bool, device=role_mu.device)
    if not mask.any():
        return role_mu.new_tensor(0.0)
    return -pairwise_sq_dist[mask].mean()
