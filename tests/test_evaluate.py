"""The ruler's own tests. These pass on a fresh clone — the evaluator ships finished."""

from __future__ import annotations

import numpy as np

from evaluate import multi_step_error, operator_error, spectrum_error
from synthetic import linear_system, linear_trajectory


def test_perfect_scores_zero():
    a = linear_system(seed=0)
    assert operator_error(a, a) < 1e-12
    assert spectrum_error(a, a) < 1e-12


def test_one_step_of_the_true_operator_is_exact():
    a = linear_system(seed=1)
    x, y = linear_trajectory(a, steps=50, seed=1)
    assert multi_step_error(a @ x, y) < 1e-12


def test_the_floor_is_not_zero():
    """A zero operator must score badly, or "we beat the floor" means nothing."""
    a = linear_system(seed=2)
    assert operator_error(np.zeros_like(a), a) > 0.1
