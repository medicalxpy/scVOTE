from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .config import VAEConfig


def _build_mlp(in_dim: int, hidden_dims: List[int], dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d, h))
        layers.append(nn.ReLU())
        if dropout and dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        d = h
    return nn.Sequential(*layers)


def _one_hot(batch_index: torch.Tensor, n_batch: int) -> torch.Tensor:
    if n_batch <= 1:
        return torch.zeros((batch_index.shape[0], 0), device=batch_index.device)
    return F.one_hot(batch_index, num_classes=n_batch).to(torch.float32)


def _kl_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def _kl_normal_vs_normal(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    return 0.5 * (
        (logvar_p - logvar_q)
        + (var_q + (mu_q - mu_p).pow(2)) / var_p
        - 1.0
    ).squeeze(-1)


def _nb_log_prob(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    if theta.ndim == 1:
        theta = theta.view(1, -1).expand_as(x)
    elif theta.ndim != 2:
        raise ValueError(f"theta must be 1D or 2D, got {theta.ndim}D")
    log_theta = torch.log(theta + eps)
    log_mu = torch.log(mu + eps)
    log_theta_mu = torch.log(theta + mu + eps)
    return (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (log_theta - log_theta_mu)
        + x * (log_mu - log_theta_mu)
    )


def _zinb_log_prob(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    nb = _nb_log_prob(x, mu, theta)
    if theta.ndim == 1:
        theta_b = theta.view(1, -1).expand_as(x)
    elif theta.ndim == 2:
        theta_b = theta
    else:
        raise ValueError(f"theta must be 1D or 2D, got {theta.ndim}D")
    log_theta = torch.log(theta_b + eps)
    log_theta_mu = torch.log(theta_b + mu + eps)
    log_nb_zero = theta_b * (log_theta - log_theta_mu)

    is_zero = (x < 1e-8)
    log_mix_zero = torch.log(pi + (1.0 - pi) * torch.exp(log_nb_zero) + eps)
    log_mix_non_zero = torch.log(1.0 - pi + eps) + nb
    return torch.where(is_zero, log_mix_zero, log_mix_non_zero)


def _poisson_log_prob(x: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    return x * torch.log(rate + eps) - rate - torch.lgamma(x + 1.0)


class InferenceNet(nn.Module):
    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        in_dim = config.n_input + (config.n_batch if config.n_batch > 1 else 0)
        hidden = config.hidden_dims()
        self.shared = _build_mlp(in_dim, hidden, config.dropout_rate)

        last_dim = hidden[-1] if hidden else in_dim
        self.z_mu = nn.Linear(last_dim, config.n_latent)
        self.z_logvar = nn.Linear(last_dim, config.n_latent)

        self.l_mu = nn.Linear(last_dim, 1)
        self.l_logvar = nn.Linear(last_dim, 1)

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor, n_batch: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_in = torch.log1p(x)
        b = _one_hot(batch_index, n_batch)
        if b.shape[1] > 0:
            x_in = torch.cat([x_in, b], dim=1)
        h = self.shared(x_in)
        return self.z_mu(h), self.z_logvar(h), self.l_mu(h), self.l_logvar(h)


class GenerativeNet(nn.Module):
    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        in_dim = config.n_latent + (config.n_batch if config.n_batch > 1 else 0)
        hidden = config.hidden_dims()
        self.net = _build_mlp(in_dim, list(reversed(hidden)), config.dropout_rate)
        last_dim = hidden[0] if hidden else in_dim

        self.px_scale = nn.Linear(last_dim, config.n_input)
        self.px_dropout = nn.Linear(last_dim, config.n_input)

        if config.dispersion == "gene":
            self.log_theta = nn.Parameter(torch.zeros(config.n_input))
        elif config.dispersion == "gene-batch":
            self.log_theta = nn.Parameter(torch.zeros(config.n_batch, config.n_input))
        else:
            raise ValueError(f"Unsupported dispersion: {config.dispersion}")

    def _theta(self, batch_index: torch.Tensor, n_batch: int) -> torch.Tensor:
        theta = self.log_theta
        if theta.ndim == 1:
            return F.softplus(theta) + 1e-8
        if n_batch <= 1:
            return F.softplus(theta[0]) + 1e-8
        return F.softplus(theta[batch_index]) + 1e-8

    def forward(
        self,
        z: torch.Tensor,
        library_log: torch.Tensor,
        batch_index: torch.Tensor,
        n_batch: int,
        gene_likelihood: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = _one_hot(batch_index, n_batch)
        z_in = torch.cat([z, b], dim=1) if b.shape[1] > 0 else z
        h = self.net(z_in)

        scale = F.softmax(self.px_scale(h), dim=-1)
        library = torch.exp(library_log)
        rate = library * scale

        if gene_likelihood == "poisson":
            return rate, torch.empty(0, device=rate.device), torch.empty(0, device=rate.device)

        theta = self._theta(batch_index, n_batch)
        if theta.ndim == 1:
            theta = theta.view(1, -1).expand_as(rate)

        if gene_likelihood == "nb":
            return rate, theta, torch.empty(0, device=rate.device)

        if gene_likelihood == "zinb":
            pi = torch.sigmoid(self.px_dropout(h))
            return rate, theta, pi

        raise ValueError(f"Unsupported gene_likelihood: {gene_likelihood}")


class VAE(nn.Module):
    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config
        self.inference = InferenceNet(config)
        self.generative = GenerativeNet(config)

        self.register_buffer("library_log_means", torch.zeros(config.n_batch))
        self.register_buffer("library_log_vars", torch.ones(config.n_batch))

    def set_library_priors(self, means: torch.Tensor, vars_: torch.Tensor) -> None:
        self.library_log_means = means
        self.library_log_vars = vars_

    def _sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        give_mean: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mu, z_logvar, l_mu, l_logvar = self.inference(x, batch_index, self.config.n_batch)

        z = z_mu if give_mean else self._sample(z_mu, z_logvar)
        library_log = l_mu if give_mean else self._sample(l_mu, l_logvar)

        rate, theta, pi = self.generative(
            z=z,
            library_log=library_log,
            batch_index=batch_index,
            n_batch=self.config.n_batch,
            gene_likelihood=self.config.gene_likelihood,
        )

        return z_mu, z_logvar, l_mu, l_logvar, rate, theta, pi

    def loss(self, x: torch.Tensor, batch_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mu, z_logvar, l_mu, l_logvar, rate, theta, pi = self.forward(x, batch_index, give_mean=False)

        if self.config.gene_likelihood == "poisson":
            log_px = _poisson_log_prob(x, rate)
        elif self.config.gene_likelihood == "nb":
            log_px = _nb_log_prob(x, rate, theta)
        elif self.config.gene_likelihood == "zinb":
            log_px = _zinb_log_prob(x, rate, theta, pi)
        else:
            raise ValueError(f"Unsupported gene_likelihood: {self.config.gene_likelihood}")

        recon = -log_px.sum(dim=1).mean()

        kl_z = _kl_normal(z_mu, z_logvar).mean()

        prior_mu = self.library_log_means[batch_index].view(-1, 1)
        prior_logvar = torch.log(self.library_log_vars[batch_index].view(-1, 1) + 1e-8)
        kl_l = _kl_normal_vs_normal(l_mu, l_logvar, prior_mu, prior_logvar).mean()

        total = recon + self.config.kl_weight * (kl_z + kl_l)
        return total, recon, kl_z, kl_l
