# 16-811 · Lab 2 · Make a nonlinear robot linear enough to control

Fit a Koopman operator by least squares, read the robot's dynamics off its eigenvalues, and drive
the thing with linear MPC that beats the state-space baseline the paper compares against.

**Technique:** Korda & Mezic, *Linear predictors for nonlinear dynamical systems: Koopman operator
meets model predictive control*, Automatica 93:149–160, 2018. [arXiv:1611.03537](https://arxiv.org/abs/1611.03537)
**Checked against:** [PyKoopman](https://github.com/dynamicslab/pykoopman) and
[soft-robot-koopman](https://github.com/ramvasudevan/soft-robot-koopman) (Bruder et al., IEEE T-RO 37(3), 2021).
**From the book:** Gallier & Quaintance ch. 21 (pseudo-inverse), 16 (spectral theorem), 8 (conditioning), 17 (computing eigenvalues).

## Start

Open in the dev container — everything is installed — then:

```bash
make check        # 8 tests. 4 fail. Those 4 are the job.
make reproduce    # runs now, with a deliberately wrong operator. Beat that number.
```

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
| `synthetic.py` | A linear system (known answer) and a damped pendulum (the reason to lift) |
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

## Graduating to a robot

`scripts/fetch_data.py` names the soft-arm data and MuJoCo. Neither is in the image on purpose —
the mathematics is what you are writing and it needs neither, and a multi-gigabyte image is a
slower first open for no gain.
