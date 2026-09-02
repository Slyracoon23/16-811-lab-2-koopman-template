"""The technique. This is the only file in the repository you have to write.

What ships here runs and is wrong: it returns a zero operator, which predicts that nothing ever
moves. The harness is proved before you touch it, so your first edit moves a real number.

What you are implementing (Gallier & Quaintance ch. 21; Korda & Mezic 2018):

    Z  = lift(X)                     each snapshot pushed through the dictionary
    Z+ = lift(Y)
    A  = Z+ @ pinv(Z)                ONE least-squares problem. That is the whole of EDMD.

`pinv` is the pseudo-inverse of section 21.1 and it is the algorithm — not a step in it. The
interesting half comes after: eigendecompose A and the slow modes are motion you can see.

Report the condition number of Z alongside. A dictionary big enough to fit anything is also big
enough to be rank-deficient, and chapter 8 is what tells you which side of that line you are on.
"""

from __future__ import annotations

import numpy as np

ASSUMPTIONS: list[str] = [
    "Snapshots are pairs: column i of Y is the state one step after column i of X.",
    "The step is fixed. A variable timestep makes the operator meaningless.",
]


def identity_lift(x: np.ndarray) -> np.ndarray:
    """The dictionary that does nothing. With this, EDMD is plain least squares — and on a linear
    system it must return that system's matrix exactly. Your first test."""
    return x


def rbf_lift(x: np.ndarray, *, centres: np.ndarray | None = None, width: float = 1.0) -> np.ndarray:
    """The state, a constant, radial bases on the first coordinate, and its sine and cosine.

    A starting dictionary, not a good one — tuning it is `extend.py`'s job. The trigonometric pair
    is in here because the pendulum's dynamics contain a sine, and a dictionary that cannot express
    the nonlinearity in the plant cannot linearise it however many bumps you add.
    """
    if centres is None:
        centres = np.linspace(-3, 3, 12)
    angle = x[0:1]
    feats = [x, np.ones((1, x.shape[1]))]
    feats += [np.exp(-((angle - c) ** 2) / width**2) for c in np.asarray(centres).ravel()]
    feats += [np.sin(angle), np.cos(angle)]
    return np.vstack(feats)


def fit_operator(x: np.ndarray, y: np.ndarray, lift=identity_lift) -> np.ndarray:
    """Return the lifted operator A with lift(Y) ~ A @ lift(X)."""
    z, z_next = lift(x), lift(y)

    # PLACEHOLDER — runs, and is wrong. One line of numpy replaces it.
    return np.zeros((z_next.shape[0], z.shape[0]))


def fit_controlled(x: np.ndarray, u: np.ndarray, y: np.ndarray, lift=identity_lift):
    """Return (A, B) with lift(Y) ~ A lift(X) + B U — the same least-squares problem, one block wider.

    This is the half the paper is actually about: a linear model *with an input* is what turns
    nonlinear MPC into a convex program. `mpc.py` is written and waiting for it.
    """
    z, z_next = lift(x), lift(y)
    stacked = np.vstack([z, np.atleast_2d(u)])

    # PLACEHOLDER — runs, and steers nothing.
    return np.zeros((z_next.shape[0], z.shape[0])), np.zeros((z_next.shape[0], np.atleast_2d(u).shape[0]))


def predict(a: np.ndarray, x0: np.ndarray, steps: int, lift=identity_lift) -> np.ndarray:
    """Roll the lifted model forward and return the first len(x0) rows: the state's own prediction."""
    z = lift(x0.reshape(-1, 1))
    out = []
    for _ in range(steps):
        z = a @ z
        out.append(z[: x0.shape[0], 0])
    return np.array(out).T
