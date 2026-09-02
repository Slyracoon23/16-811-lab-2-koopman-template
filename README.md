# 16-811 · Lab 2 · Make a nonlinear robot linear enough to control

Fit a Koopman operator by least squares, read the robot's dynamics off its eigenvalues, and drive
the thing with linear MPC that beats the state-space baseline the paper compares against.

**Technique:** Korda & Mezic, *Linear predictors for nonlinear dynamical systems: Koopman operator
meets model predictive control*, Automatica 93:149–160, 2018. [arXiv:1611.03537](https://arxiv.org/abs/1611.03537)
**Checked against:** [PyKoopman](https://github.com/dynamicslab/pykoopman) and
[soft-robot-koopman](https://github.com/ramvasudevan/soft-robot-koopman) (Bruder et al., IEEE T-RO 37(3), 2021).
**Built on:** [MuJoCo](https://mujoco.org) for the plant, [OSQP](https://osqp.org) for the
controller, [Weights & Biases](https://wandb.ai) for the sweep.
**From the book:** Gallier & Quaintance ch. 21 (pseudo-inverse), 16 (spectral theorem), 8 (conditioning), 17 (computing eigenvalues).

## Start

Open in the dev container — everything is installed — then:

```bash
make check        # 9 tests. 6 fail. Those 6 are the job.
make reproduce    # runs now, with a deliberately wrong operator. Beat that number.
```

Nothing needs a GPU and nothing needs an account: MuJoCo runs headless on the CPU, and W&B is in
offline mode until you `wandb login`.

Nothing here raises `NotImplementedError` except your own `extend.py`. `method.py` ships a zero
operator: it predicts that nothing ever moves. The harness is proved before you touch it, so your
first edit moves a real number.

## What you write

**`method.py`, one function.** Everything else is scaffolding.

```
Z  = lift(X)              each snapshot through the dictionary
Z⁺ = lift(Y)
A  = Z⁺ · pinv(Z)         ONE least-squares problem. That is the whole of EDMD.
```

Section 21.1 of the book is not background here — it is the algorithm. The interesting half comes
after: eigendecompose `A` and the slow modes are motion you can see in the simulator.

## What the tools are doing here

The paper is from 2018 and its examples are MATLAB toys. Three things sit on top of it so that the
same claim is made where robotics is actually done:

- **MuJoCo** is the plant. `plant.py` has a pendulum that swings freely — where the lift proves
  itself — and the same arm with a motor at the hinge, which is what the controller drives. A
  hand-rolled Euler pendulum was here first and it was a model of a model: the claim is that a lift
  beats a linear fit on a *real* plant, and a plant you wrote yourself cannot be evidence for that.
- **OSQP** is the controller. `mpc.py` is written for you, because it is the paper's punchline
  rather than its exercise: the Koopman model is linear, so the optimal control problem is a convex
  QP. It solves against whatever model it is handed, which makes the arm a direct readout of your
  operator — right, and it holds the setpoint; wrong, and it flails.
- **Weights & Biases** is the notebook. Every question in this lab is a sweep, and `track.py` logs
  each row. Offline by default; `wandb login` and `WANDB_MODE=online` when you want it shared.

## Why the first test is a linear system

Because you can check it. With the identity dictionary, EDMD on a linear system must return that
system's own matrix, exactly, to machine precision — and the operator's eigenvalues must be that
matrix's eigenvalues. If that does not hold, nothing you later conclude about the pendulum means
anything. Never trust an operator on a system whose answer you cannot write down.

## The files

| | |
|---|---|
| `method.py` | **Yours.** `fit_operator` is one line of NumPy when finished |
| `evaluate.py` | The ruler. Complete, and it never imports `method` |
| `plant.py` | The MuJoCo arm, free and driven. Snapshot pairs and trajectories |
| `mpc.py` | The controller. **Given** — it is the payoff, not the exercise |
| `track.py` | W&B, offline by default, degrading to `print` if it is unavailable |
| `synthetic.py` | The linear system whose answer you know. Check here before believing MuJoCo |
| `baselines.py` | A floor (zero, identity) and a ceiling (the true operator) |
| `reproduce.py` | Seeds and horizons → `results.json` |
| `extend.py` | Your own ideas, as switches |
| `tests/` | The to-do list |

## Past the paper

Korda and Mezic choose a dictionary and report that it works. Nobody tells you *which* dictionary,
or how you would know before training that it will fail — and the book has an answer, because a
lift whose data matrix is ill-conditioned cannot be fitted stably however expressive it is.

Three angles, all of them more of the book rather than less:

1. Plot prediction error against the lift's **condition number** rather than against its size,
   across radial-basis, polynomial and time-delay dictionaries.
2. Truncate the operator by its **singular values** and ask how many modes the control actually needs.
3. Run the whole thing on a **MuJoCo arm with contact**, and report where a linear predictor stops
   being enough.

**How you would know:** the same multi-step prediction error `reproduce.py` already reports, so a
dictionary chosen by conditioning has to beat one chosen by size on the axis the paper itself uses.

## What you should see

```
            plant      lift  dims       error   condition
           linear  identity     2   2.500e-15   1.000e+00
  mujoco-pendulum  identity     2   3.231e+00   1.300e+00
  mujoco-pendulum       rbf    17   8.127e-02   1.993e+04
```

Exact on the system whose answer you know. A linear model of the pendulum is useless. Lifting is
forty times better — **and fifteen thousand times worse conditioned**, which is the price chapter 8
charges and the thread the extension pulls.

## One trap worth knowing

MuJoCo silently clamps any control outside an actuator's `ctrlrange`. Sample wider than the range
and the inputs you record were never the inputs applied, so `fit_controlled` learns its `B` partly
from fiction and the controller under-steers for reasons no amount of staring at the mathematics
will explain. `plant.rollout` reads the range off the model for exactly this reason. It cost an
afternoon here before it was a comment.

## Graduating to a real robot

`scripts/fetch_data.py` names the soft-arm data from Bruder et al. A pendulum is a pendulum; the
same three files point at a robot somebody actually built.
