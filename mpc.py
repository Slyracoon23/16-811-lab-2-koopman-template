"""Model predictive control on the lifted model. Written for you — deliberately.

This file is the paper's punchline and none of it is your work: because the Koopman model is
*linear*, the optimal control problem is a convex quadratic program, and a QP solver handles it at
control rate. That is the entire argument of Korda & Mezic, and it is thirty lines once the model
is linear.

It is given rather than assigned so the payoff is reachable. Your job is `fit_controlled` in
`method.py`; when that is right, this steers, and when it is wrong, this flails. The controller
never changes, so the cart is a direct readout of your operator.
"""

from __future__ import annotations

import numpy as np
import osqp
from scipy import sparse


class LiftedMPC:
    """Condensed MPC over a horizon on z+ = A z + B u, tracking the first `outputs` coordinates."""

    def __init__(self, a, b, lift, *, horizon=25, outputs=2, effort=1e-3, limit=10.0):
        self.a, self.b, self.lift = np.asarray(a), np.atleast_2d(b), lift
        self.horizon, self.outputs, self.limit = horizon, outputs, limit

        n, m = self.a.shape[0], self.b.shape[1]
        # Free response and forced response, stacked over the horizon.
        phi = np.zeros((horizon * outputs, n))
        gamma = np.zeros((horizon * outputs, horizon * m))
        power = np.eye(n)
        for k in range(horizon):
            power = self.a @ power
            phi[k * outputs : (k + 1) * outputs] = power[:outputs]
            for j in range(k + 1):
                block = np.linalg.matrix_power(self.a, k - j) @ self.b
                gamma[k * outputs : (k + 1) * outputs, j * m : (j + 1) * m] = block[:outputs]
        self.phi, self.gamma, self.m = phi, gamma, m

        hessian = 2 * (gamma.T @ gamma + effort * np.eye(horizon * m))
        self.problem = osqp.OSQP()
        self.problem.setup(
            P=sparse.csc_matrix(hessian),
            q=np.zeros(horizon * m),
            A=sparse.eye(horizon * m, format="csc"),
            l=-limit * np.ones(horizon * m),
            u=limit * np.ones(horizon * m),
            verbose=False,
        )

    def __call__(self, state, _step=0, reference=None):
        """Return the first control of the optimal sequence, recomputed every step."""
        z = self.lift(np.asarray(state).reshape(-1, 1))[:, 0]
        target = np.zeros(self.horizon * self.outputs) if reference is None else np.asarray(reference)
        gradient = 2 * self.gamma.T @ (self.phi @ z - target)
        self.problem.update(q=gradient)
        result = self.problem.solve()
        if result.x is None or not np.all(np.isfinite(result.x)):
            return np.zeros(self.m)
        return np.clip(result.x[: self.m], -self.limit, self.limit)
