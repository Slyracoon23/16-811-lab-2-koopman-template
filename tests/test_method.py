"""Your to-do list, as tests.

These fail against the placeholder in `method.py` and pass when it is EDMD. Run them with
`make check`. The order is the order to fix them in.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluate import multi_step_error, operator_error, spectrum_error
from method import fit_operator, identity_lift, predict, rbf_lift
from synthetic import linear_system, linear_trajectory, pendulum


def test_recovers_a_linear_system_exactly():
    """Step 1. With the identity dictionary EDMD is plain least squares, and on a linear system
    the answer is the system's own matrix. Exact, to machine precision."""
    a = linear_system(seed=0)
    x, y = linear_trajectory(a, steps=400, seed=0)
    assert operator_error(fit_operator(x, y, identity_lift), a) < 1e-10


def test_the_spectrum_is_the_systems_spectrum():
    """Step 2. The eigenvalues are the physics. If the operator is right these must match."""
    a = linear_system(seed=1)
    x, y = linear_trajectory(a, steps=400, seed=1)
    assert spectrum_error(fit_operator(x, y, identity_lift), a) < 1e-8


def test_predicts_a_linear_system_over_a_horizon():
    """Step 3. Rolling the fitted operator forward must track the truth, not drift."""
    a = linear_system(seed=2)
    x, y = linear_trajectory(a, steps=400, seed=2)
    fitted = fit_operator(x, y, identity_lift)
    x0 = x[:, 0]
    truth = np.array([np.linalg.matrix_power(a, k + 1) @ x0 for k in range(25)]).T
    assert multi_step_error(predict(fitted, x0, 25, identity_lift), truth) < 1e-8


def test_is_not_fooled_by_more_data():
    """Step 4. Least squares should be stable as columns are added, not wander."""
    a = linear_system(seed=3)
    short = fit_operator(*linear_trajectory(a, steps=100, seed=3), identity_lift)
    long = fit_operator(*linear_trajectory(a, steps=2000, seed=3), identity_lift)
    assert operator_error(short, a) < 1e-8
    assert operator_error(long, a) < 1e-8


@pytest.mark.xfail(reason="Step 5: needs a dictionary that suits the pendulum. Remove the mark when it passes.")
def test_beats_a_linear_fit_on_the_pendulum():
    """The point of the lift. A linear model of a nonlinear system is bad; yours must be better."""
    x, y = pendulum(steps=2000, seed=0)
    plain = fit_operator(x, y, identity_lift)
    lifted = fit_operator(x, y, rbf_lift)
    x0 = x[:, 0]
    truth = x[:, 1:26]
    plain_err = multi_step_error(predict(plain, x0, 25, identity_lift), truth)
    lifted_err = multi_step_error(predict(lifted, x0, 25, rbf_lift), truth)
    assert lifted_err < plain_err
