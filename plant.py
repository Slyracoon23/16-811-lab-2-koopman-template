"""The robot: MuJoCo, headless, on the CPU.

The paper's examples are MATLAB toys. This is the same claim on a physics engine that robotics
actually uses — which is the whole point of the lab, and it costs about sixty megabytes and no GPU.

Two plants, and the same arm twice. `PENDULUM` swings freely and is where the lift proves itself:
a linear model of it is hopeless and a lifted one is not. `DRIVEN_PENDULUM` has a motor at the
hinge, and holding it at an angle away from the bottom is a control problem whose difficulty is
gravity's sine — exactly the nonlinearity a single linearisation gets wrong and a lift does not.

A cartpole was the first choice and it is not here, because the honest experiment said no: a
Koopman MPC built from this dictionary does not balance one. Shipping a lab whose payoff does not
happen would have been worse than picking a task that does.
"""

from __future__ import annotations

import mujoco
import numpy as np

PENDULUM = """
<mujoco>
  <option timestep="0.02" integrator="RK4"/>
  <worldbody><body name="arm" pos="0 0 0">
    <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
    <geom type="capsule" fromto="0 0 0 0 0 .5" size=".02" mass=".5"/>
  </body></worldbody>
</mujoco>
"""

DRIVEN_PENDULUM = """
<mujoco>
  <option timestep="0.02" integrator="RK4"/>
  <worldbody><body pos="0 0 0">
    <joint name="hinge" type="hinge" axis="0 1 0" damping="0.15"/>
    <geom type="capsule" fromto="0 0 0 0 0 .5" size=".02" mass=".5"/>
  </body></worldbody>
  <actuator><motor joint="hinge" ctrlrange="-3 3" gear="1"/></actuator>
</mujoco>
"""

def model(xml: str = PENDULUM) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(xml)


def state(data: mujoco.MjData) -> np.ndarray:
    return np.concatenate([data.qpos, data.qvel])


def rollout(xml: str = PENDULUM, *, trajectories: int = 60, steps: int = 80, seed: int = 0):
    """Snapshot pairs (X, U, Y) from random starts. Columns are samples.

    Random control rather than none, on the actuated plant: an operator fitted on a system that was
    never pushed has learned nothing about what pushing does, and the MPC built on it will not
    steer. This is the most common way a Koopman model quietly fails.

    The control is drawn inside the actuator's own `ctrlrange`, and that detail is not fussiness.
    MuJoCo silently clamps anything outside it, so a wider draw records inputs that were never
    applied — the fit then learns a B from a third of its rows being fiction, and the controller
    under-steers for reasons nothing in the mathematics will explain.
    """
    m = model(xml)
    rng = np.random.default_rng(seed)
    low, high = (m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]) if m.nu else (None, None)
    xs, us, ys = [], [], []
    for _ in range(trajectories):
        d = mujoco.MjData(m)
        d.qpos[:] = rng.uniform(-2.5, 2.5, size=m.nq)
        d.qvel[:] = rng.uniform(-3.0, 3.0, size=m.nv)
        for _ in range(steps):
            u = rng.uniform(low, high, size=m.nu) if m.nu else np.zeros(0)
            before = state(d)
            if m.nu:
                d.ctrl[:] = u
            mujoco.mj_step(m, d)
            xs.append(before)
            us.append(u)
            ys.append(state(d))
    return np.array(xs).T, np.array(us).T, np.array(ys).T


def trajectory(xml: str, start: np.ndarray, steps: int, *, controller=None):
    """Run one trajectory and return the states, so you can score a prediction against it."""
    m = model(xml)
    d = mujoco.MjData(m)
    d.qpos[:] = start[: m.nq]
    d.qvel[:] = start[m.nq :]
    out = []
    for step in range(steps):
        if m.nu and controller is not None:
            d.ctrl[:] = np.clip(controller(state(d), step), -10, 10)
        mujoco.mj_step(m, d)
        out.append(state(d))
    return np.array(out).T
