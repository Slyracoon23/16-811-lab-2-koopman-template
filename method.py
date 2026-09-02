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
    """Radial basis observables stacked under the state itself. Yours to tune in extend.py."""
    if centres is None:
        centres = np.linspace(-2, 2, 10)[None, :].repeat(x.shape[0], axis=0)
    feats = [x]
    for row in range(centres.shape[1]):
        c = centres[:, row : row + 1]
        feats.append(np.exp(-((x - c) ** 2).sum(axis=0, keepdims=True) / width**2))
    return np.vstack(feats)


def fit_operator(x: np.ndarray, y: np.ndarray, lift=identity_lift) -> np.ndarray:
    """Return the lifted operator A with lift(Y) ~ A @ lift(X)."""
    z, z_next = lift(x), lift(y)

    # PLACEHOLDER — runs, and is wrong. One line of numpy replaces it.
    return np.zeros((z_next.shape[0], z.shape[0]))


def predict(a: np.ndarray, x0: np.ndarray, steps: int, lift=identity_lift) -> np.ndarray:
    """Roll the lifted model forward and return the first len(x0) rows: the state's own prediction."""
    z = lift(x0.reshape(-1, 1))
    out = []
    for _ in range(steps):
        z = a @ z
        out.append(z[: x0.shape[0], 0])
    return np.array(out).T
