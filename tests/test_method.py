"""Your to-do list, as tests.

These fail against the placeholder in `method.py` and pass when it is EDMD. The first four are
algebra you can check by hand; the last two are the claim the paper actually makes, on a plant
MuJoCo integrates.
"""

from __future__ import annotations

import numpy as np
import pytest

import plant
from evaluate import multi_step_error, operator_error, spectrum_error
from method import fit_controlled, fit_operator, identity_lift, predict, rbf_lift
from synthetic import linear_system, linear_trajectory


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


def test_the_controlled_fit_recovers_a_known_input_matrix():
    """Step 4. Same least squares, one block wider. `mpc.py` cannot steer until this is right."""
    rng = np.random.default_rng(0)
    a, b = linear_system(seed=3), rng.normal(size=(3, 1))
    x = rng.normal(size=(3, 500))
    u = rng.normal(size=(1, 500))
    y = a @ x + b @ u
    fitted_a, fitted_b = fit_controlled(x, u, y, identity_lift)
    assert operator_error(fitted_a, a) < 1e-8
    assert operator_error(fitted_b, b) < 1e-8


# The paper's claim, on a plant we did not write. Slower than the algebra above: MuJoCo has to
# integrate a few thousand steps first.
def test_the_lift_beats_a_linear_model_on_a_real_pendulum():
    """Step 5. This is why the dictionary exists.

    A pendulum's gravity term is a sine, so one global linear model of it is hopeless however much
    data you give it. Lift the state and the same least-squares problem does far better — that is
    Korda and Mezic's whole argument, and here it is on MuJoCo rather than on paper.
    """
    x, _, y = plant.rollout(plant.PENDULUM, trajectories=60, steps=80, seed=0)
    start = np.array([2.0, 0.0])
    truth = plant.trajectory(plant.PENDULUM, start, 40)

    plain = predict(fit_operator(x, y, identity_lift), start, 40, identity_lift)
    lifted = predict(fit_operator(x, y, rbf_lift), start, 40, rbf_lift)

    plain_error = multi_step_error(plain, truth)
    lifted_error = multi_step_error(lifted, truth)
    assert lifted_error < plain_error / 10, f"lift {lifted_error:.4f} vs linear {plain_error:.4f}"


def test_the_controller_holds_the_arm_where_a_linear_model_cannot():
    """Step 6. The payoff, and none of the controller is your code.

    `mpc.py` is written. It solves a QP against whatever model it is handed, so the arm is a direct
    readout of your operator: get `fit_controlled` right and it holds the setpoint, get it wrong
    and it flails. The linear model is the thing to beat, not gravity.
    """
    from mpc import LiftedMPC

    target, horizon = 1.2, 30
    reference = np.tile(np.array([target, 0.0]), horizon)
    x, u, y = plant.rollout(plant.DRIVEN_PENDULUM, trajectories=150, steps=60, seed=0)

    def error_under(lift):
        a, b = fit_controlled(x, u, y, lift)
        mpc = LiftedMPC(a, b, lift, horizon=horizon, outputs=2, effort=1e-3, limit=3.0)
        states = plant.trajectory(
            plant.DRIVEN_PENDULUM,
            np.array([-1.0, 0.0]),
            250,
            controller=lambda s, k: mpc(s, k, reference=reference),
        )
        return float(np.mean(np.abs(states[0, -50:] - target)))

    assert error_under(rbf_lift) < error_under(identity_lift)
    assert error_under(rbf_lift) < 0.5, "the arm is not holding the setpoint"
