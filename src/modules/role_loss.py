from __future__ import annotations

import torch


def diagonal_gaussian_kl(
    mu_p: torch.Tensor,
    std_p: torch.Tensor,
    mu_q: torch.Tensor,
    std_q: torch.Tensor,
) -> torch.Tensor:
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
