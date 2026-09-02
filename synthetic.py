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


def pendulum(steps: int = 2000, *, dt: float = 0.02, damping: float = 0.1, seed: int = 0):
    """A damped pendulum, integrated. Nonlinear, and the reason a lift is needed at all.

    Returns snapshot pairs (X, Y) of shape (2, steps): rows are angle and angular velocity.
    """
    rng = np.random.default_rng(seed)
    state = np.array([rng.uniform(-1.0, 1.0), rng.uniform(-0.5, 0.5)])
    xs, ys = [], []
    for _ in range(steps):
        theta, omega = state
        nxt = np.array([theta + dt * omega, omega + dt * (-np.sin(theta) - damping * omega)])
        xs.append(state)
        ys.append(nxt)
        state = nxt
        if abs(state[0]) > 10:
            state = np.array([rng.uniform(-1.0, 1.0), rng.uniform(-0.5, 0.5)])
    return np.array(xs).T, np.array(ys).T
