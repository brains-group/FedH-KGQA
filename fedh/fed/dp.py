"""Differentially private training for FedH-KGQA clients (QA-pair-level DP).

Each client runs DP-SGD on its own questions: every step samples each of its n training questions
independently with probability q (Poisson sampling), clips each sampled question's gradient to L2
norm C, sums them and adds Gaussian noise N(0, sigma^2 C^2 I). The updates a client releases are
then (epsilon, delta)-DP with respect to any one of its QA pairs, against the server and the other
clients, whatever happens to them afterwards (FedAvg and secure aggregation are post-processing).
In deployment the noise can be split across clients and summed under secure aggregation
(distributed discrete Gaussian); the simulation adds it locally, which gives the same guarantee.

Accounting: Renyi DP of the Poisson-subsampled Gaussian at integer orders (Mironov et al., 2019),
composed over all steps of all rounds, converted to (epsilon, delta) with the standard bound.
sigma is calibrated by bisection to hit a target epsilon for the client's step count.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

ORDERS = list(range(2, 257))


def _log_comb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def rdp_subsampled_gaussian(q: float, sigma: float, alpha: int) -> float:
    """RDP of one step at integer order alpha (exact for Poisson subsampling)."""
    if q == 0:
        return 0.0
    if q == 1.0:
        return alpha / (2 * sigma ** 2)
    terms = [_log_comb(alpha, k) + (alpha - k) * math.log1p(-q) + k * math.log(q) + (k * k - k) / (2 * sigma ** 2)
             for k in range(alpha + 1)]
    m = max(terms)
    return (m + math.log(sum(math.exp(t - m) for t in terms))) / (alpha - 1)


def epsilon(q: float, sigma: float, steps: int, delta: float) -> float:
    best = float("inf")
    for a in ORDERS:
        r = steps * rdp_subsampled_gaussian(q, sigma, a)
        eps = r + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)
        best = min(best, eps)
    return best


def calibrate_sigma(q: float, steps: int, delta: float, target_eps: float) -> float:
    lo, hi = 0.3, 1.0
    while epsilon(q, hi, steps, delta) > target_eps:
        hi *= 2
        if hi > 1e4:
            raise ValueError("cannot reach the target epsilon")
    for _ in range(40):
        mid = (lo + hi) / 2
        if epsilon(q, mid, steps, delta) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


def per_example_loss(rel_logits, k_logits, paths, path_mask, k_target):
    """reasoner_loss for one example, written without data-dependent shapes (vmap-friendly)."""
    logp = F.log_softmax(rel_logits, dim=-1)                     # [1, K, R]
    B, P, K = paths.shape
    lp = torch.gather(logp.unsqueeze(1).expand(B, P, K, logp.size(-1)), 3, paths.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    lp = (lp * (paths >= 0)).sum(-1)
    lp = torch.where(path_mask > 0, lp, torch.full_like(lp, -1e4))
    has = (path_mask.sum(1) > 0).float()
    loss_path = -torch.logsumexp(lp, dim=1) * has
    lk = -F.log_softmax(k_logits, -1).gather(1, k_target.clamp(min=0).unsqueeze(1)).squeeze(1) * (k_target >= 0).float()
    return (loss_path + lk).sum()


class DPStep:
    """Per-example clipped gradients for the trainable parameters of a Reasoner; noise is added once
    per step to the clipped sum (`finish`)."""

    def __init__(self, model, clip: float, sigma: float, expected_batch: float):
        self.model, self.clip, self.sigma, self.eb = model, clip, sigma, expected_batch
        self.names = [n for n, p in model.named_parameters() if p.requires_grad]

    def clipped_sum(self, ids, att, paths, pmask, k):
        model = self.model
        params = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
        frozen = {n: p.detach() for n, p in model.named_parameters() if not p.requires_grad}
        buffers = dict(model.named_buffers())

        def loss_fn(p, i, a, pa, pm, kk):
            rl, kl = functional_call(model, ({**frozen, **p}, buffers), (i.unsqueeze(0), a.unsqueeze(0)))
            return per_example_loss(rl.float(), kl.float(), pa.unsqueeze(0), pm.unsqueeze(0), kk.unsqueeze(0))

        g = vmap(grad(loss_fn), in_dims=(None, 0, 0, 0, 0, 0))(params, ids, att, paths, pmask, k)
        norms = torch.sqrt(sum((g[n].reshape(g[n].shape[0], -1).float() ** 2).sum(1) for n in self.names))
        scale = (self.clip / (norms + 1e-12)).clamp(max=1.0)
        out = {n: (g[n].float() * scale.view(-1, *([1] * (g[n].dim() - 1)))).sum(0) for n in self.names}
        return out, norms

    def finish(self, sums, like):
        """Add N(0, sigma^2 C^2) to each clipped sum and divide by the expected batch size."""
        out = {}
        for n in self.names:
            s = sums[n] if sums is not None else torch.zeros_like(like[n], dtype=torch.float32)
            out[n] = ((s + torch.randn_like(s) * self.sigma * self.clip) / self.eb).to(like[n].dtype)
        return out


def poisson_batch(n: int, q: float, rng: np.random.Generator):
    return np.nonzero(rng.random(n) < q)[0]
