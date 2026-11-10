"""Privacy accounting helpers for the optional inference noise (goal G4).

These use the continuous Gaussian mechanism with RDP composition. The final
system uses the distributed discrete Gaussian, whose RDP bound is close to this
one for moderate noise; P0 only needs the order of magnitude of epsilon.
"""
from __future__ import annotations

import math

import numpy as np

ALPHAS = np.concatenate([np.linspace(1.01, 10, 200), np.arange(11, 257, dtype=float)])


def relay_l2_sensitivity(d: int, fixed_mass: bool = True) -> float:
    """Per-round L2 sensitivity of one client's sketch to one triple.

    One triple changes at most two capped candidates (one leaves, one enters),
    each with score <= 1 in d buckets: sqrt(2d). With fixed-mass padding the
    dummy scores absorb a mass change of at most 2, spread over the dummies;
    in the worst case it lands on one dummy's d buckets: 2*sqrt(d).
    Quantisation and randomised rounding add a further term that P0 ignores.
    """
    s = math.sqrt(2 * d)
    if fixed_mass:
        s += 2 * math.sqrt(d)
    return s


def gaussian_rdp_epsilon(noise_multiplier: float, compositions: int, delta: float) -> float:
    """(eps, delta) after `compositions` Gaussian releases with sigma = nm * sensitivity."""
    if noise_multiplier <= 0:
        return float("inf")
    a = ALPHAS
    rdp = compositions * a / (2.0 * noise_multiplier ** 2)
    eps = rdp + np.log((a - 1) / a) - (math.log(delta) + np.log(a)) / (a - 1)
    return float(np.min(eps))


def client_noise_std(sigma_total: float, n_clients: int, curious_client: bool = True) -> float:
    """Per-client noise std so that the shares the adversary does not know sum to sigma_total.

    Against a curious client, which can subtract its own share, the other N-1
    shares must carry the full variance.
    """
    if sigma_total <= 0:
        return 0.0
    k = n_clients - 1 if (curious_client and n_clients > 1) else n_clients
    return sigma_total / math.sqrt(k)
