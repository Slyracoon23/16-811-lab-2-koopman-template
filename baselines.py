"""A floor and a ceiling, so a number from `method.py` means something on day one."""

from __future__ import annotations

import numpy as np


def zero_operator(x: np.ndarray) -> np.ndarray:
    """Predicts that nothing moves. The floor — anything not beating this is not working."""
    return np.zeros((x.shape[0], x.shape[0]))


def identity_operator(x: np.ndarray) -> np.ndarray:
    """Predicts that nothing changes. A stronger floor than zero on a slow system."""
    return np.eye(x.shape[0])


def oracle(a: np.ndarray) -> np.ndarray:
    """The true operator. The ceiling: your linear-system result should reach it to ~1e-12."""
    return a
