"""Systems whose answer you already know.

The whole discipline of this lab is: never trust the operator on a system you cannot check. A
linear system is the check — EDMD with an identity dictionary must recover its matrix exactly,
and the operator's eigenvalues must be that matrix's eigenvalues. If that does not hold to machine
precision, nothing you learn about the pendulum afterwards means anything.
"""

from __future__ import annotations

import numpy as np


def linear_system(seed: int = 0, *, dim: int = 3) -> np.ndarray:
    """A stable linear map. Spectral radius < 1, so trajectories stay bounded."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(dim, dim))
    return a / (1.2 * max(abs(np.linalg.eigvals(a))))


def linear_trajectory(a: np.ndarray, steps: int = 400, seed: int = 0):
    """Snapshot pairs (X, Y) with Y = A X. The answer is `a`."""
    rng = np.random.default_rng(seed + 1)
    x = rng.normal(size=(a.shape[0], steps))
    return x, a @ x


# The pendulum moved to `plant.py`, where MuJoCo integrates it. A hand-rolled Euler pendulum was
# a model of a model: the lab's claim is that a lift beats a linear fit on a *real* plant, and a
# plant you wrote yourself cannot be evidence for that.
