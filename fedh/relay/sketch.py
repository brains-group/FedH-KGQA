"""Count-min sketch relay used by FedH-KGQA (design revision 4).

Every client turns its candidate list for one relay round into a d x B sketch:
each entry (token, score) adds `score` to one bucket per hash row. The server
only ever sees the element-wise sum of all clients' sketches. A client reads
an entity back by taking the minimum over the d rows at that entity's buckets.

Tokens stand in for OPRF outputs tau(e) = PRF_k(e). In the simulation a keyed
64-bit mix plays the PRF; dummy tokens are uniform random 64-bit values, which
is what a PRF output looks like to anyone without the key.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

_MASK = (1 << 64) - 1


def _mix64_int(x: int) -> int:
    """splitmix64 finaliser on a Python int (used for scalar salts)."""
    x &= _MASK
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _MASK
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _MASK
    x ^= x >> 31
    return x


def mix64(x) -> np.ndarray:
    """Vectorised splitmix64 finaliser on uint64 arrays (wrap-around is intended)."""
    x = np.asarray(x, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        x ^= x >> np.uint64(30)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x *= np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return x


def entity_tokens(entity_ids, key: int) -> np.ndarray:
    """Simulated OPRF tokens tau(e) for integer entity ids."""
    salt = np.uint64(_mix64_int(key ^ 0x5DEECE66D))
    return mix64(np.asarray(entity_ids, dtype=np.uint64) ^ salt)


def bucket_index(tokens: np.ndarray, nonce: int, d: int, B: int) -> np.ndarray:
    """Bucket of every token in every hash row, shape (d, n).

    b_j = H(tau, nonce, j) mod B with a public hash H and a fresh nonce per query.
    """
    tokens = np.asarray(tokens, dtype=np.uint64)
    rows = np.empty((d, tokens.shape[0]), dtype=np.int64)
    for j in range(d):
        salt = np.uint64(_mix64_int((nonce * 0x9E3779B97F4A7C15 + j + 1) & _MASK))
        rows[j] = (mix64(tokens ^ salt) % np.uint64(B)).astype(np.int64)
    return rows


def build_sketch(tokens, scores, nonce: int, d: int, B: int) -> np.ndarray:
    sk = np.zeros((d, B), dtype=np.float64)
    if len(tokens) == 0:
        return sk
    b = bucket_index(tokens, nonce, d, B)
    scores = np.asarray(scores, dtype=np.float64)
    for j in range(d):
        np.add.at(sk[j], b[j], scores)
    return sk


def read_back(sketch: np.ndarray, tokens, nonce: int) -> np.ndarray:
    """Count-min estimate for each token: minimum over the d rows."""
    d, B = sketch.shape
    tokens = np.asarray(tokens, dtype=np.uint64)
    if tokens.shape[0] == 0:
        return np.zeros(0)
    b = bucket_index(tokens, nonce, d, B)
    return sketch[np.arange(d)[:, None], b].min(axis=0)


def randomized_round(x: np.ndarray, gamma: float, rng: np.random.Generator) -> np.ndarray:
    """Quantise to the grid 1/gamma with unbiased randomised rounding."""
    if gamma <= 0:
        return x
    y = x * gamma
    lo = np.floor(y)
    return (lo + (rng.random(y.shape) < (y - lo))) / gamma


@dataclass
class PadConfig:
    L: int                    # entries every client submits per round
    mass_per_slot: float      # fixed total mass M = mass_per_slot * L
    min_dummy_frac: float     # at least this share of the L slots are dummies
    dummy_scores: str = "beta"  # "even", "beta", or "empirical" (drawn from observed real scores)
    beta_a: float = 2.0
    beta_b: float = 5.0
    empirical: Optional[np.ndarray] = None  # sample of real per-entry scores for "empirical"
    fixed_mass: bool = True   # False: dummies drawn i.i.d. from the score distribution, no mass target

    @property
    def M(self) -> float:
        return self.mass_per_slot * self.L

    @property
    def max_real(self) -> int:
        return max(1, self.L - max(1, math.ceil(self.min_dummy_frac * self.L)))


@dataclass
class PadInfo:
    n_real: int
    n_truncated: int
    real_mass: float
    scaled: bool              # real mass exceeded M and was scaled down
    dummy_mean: float


def pad_entries(real_tokens, real_scores, cfg: PadConfig, rng: np.random.Generator):
    """Cap real candidates and pad to exactly L entries.

    fixed_mass=True: total mass M; real scores are scaled down when their mass exceeds M.
    fixed_mass=False (distribution-matched): dummy scores are drawn i.i.d. from the real
    score distribution (cfg.empirical), so every slot looks like a draw from the same law
    whatever the real frontier size; nothing is scaled.
    Returns (tokens, scores, info).
    """
    real_tokens = np.asarray(real_tokens, dtype=np.uint64)
    real_scores = np.asarray(real_scores, dtype=np.float64)
    n_trunc = 0
    if real_tokens.shape[0] > cfg.max_real:
        keep = np.argsort(-real_scores)[: cfg.max_real]
        n_trunc = real_tokens.shape[0] - cfg.max_real
        real_tokens, real_scores = real_tokens[keep], real_scores[keep]
    mass = float(real_scores.sum())
    n_dummy = cfg.L - real_tokens.shape[0]
    if not cfg.fixed_mass:
        dummy_tokens = rng.integers(0, np.iinfo(np.uint64).max, size=n_dummy, dtype=np.uint64)
        if cfg.empirical is not None and cfg.empirical.size:
            dummy_scores = rng.choice(cfg.empirical, size=n_dummy)
        else:
            dummy_scores = rng.beta(cfg.beta_a, cfg.beta_b, size=n_dummy)
        tokens = np.concatenate([real_tokens, dummy_tokens])
        scores = np.concatenate([real_scores, dummy_scores])
        perm = rng.permutation(cfg.L)
        info = PadInfo(int(real_tokens.shape[0]), n_trunc, mass, False,
                       float(dummy_scores.mean()) if n_dummy else 0.0)
        return tokens[perm], scores[perm], info
    scaled = mass > cfg.M
    if scaled:
        real_scores = real_scores * (cfg.M / mass)
        mass = cfg.M
    dummy_tokens = rng.integers(0, np.iinfo(np.uint64).max, size=n_dummy, dtype=np.uint64)
    deficit = cfg.M - mass
    if cfg.dummy_scores == "even":
        dummy_scores = np.full(n_dummy, deficit / n_dummy)
    elif cfg.dummy_scores == "empirical" and cfg.empirical is not None and cfg.empirical.size:
        raw = rng.choice(cfg.empirical, size=n_dummy) + 1e-9
        dummy_scores = raw * (deficit / raw.sum())
    else:
        raw = rng.beta(cfg.beta_a, cfg.beta_b, size=n_dummy) + 1e-9
        dummy_scores = raw * (deficit / raw.sum())
    tokens = np.concatenate([real_tokens, dummy_tokens])
    scores = np.concatenate([real_scores, dummy_scores])
    perm = rng.permutation(cfg.L)
    info = PadInfo(int(real_tokens.shape[0]), n_trunc, mass, scaled,
                   float(dummy_scores.mean()) if n_dummy else 0.0)
    return tokens[perm], scores[perm], info


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (no SciPy dependency)."""
    a, b = np.sort(np.asarray(a)), np.sort(np.asarray(b))
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    ca = np.searchsorted(a, grid, side="right") / a.size
    cb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(ca - cb)))
