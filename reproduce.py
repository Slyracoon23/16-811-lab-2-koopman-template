"""Fit the operator, score it, and log the sweep.

Two questions, one script. On the linear system the answer is known, so the numbers should read
~1e-15 and anything else is a bug. On MuJoCo's pendulum there is no closed form, so the question
becomes comparative: does lifting beat not lifting, and by how much?

Every row goes to Weights & Biases (offline unless you log in) because the dictionary choice is a
sweep, and a sweep you cannot go back and look at is a sweep you will run again.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

import plant
import track
from evaluate import multi_step_error, operator_error, spectrum_error
from method import ASSUMPTIONS, fit_operator, identity_lift, predict, rbf_lift
from synthetic import linear_system, linear_trajectory

def jsonable(record: dict) -> dict:
    """A record with every non-finite float replaced by ``None``.

    `json.dump` writes `NaN` and `Infinity` by default. Python reads those back; `JSON.parse`
    refuses them outright, so a single degenerate row makes the whole results file unreadable to
    anything that is not Python — including the course app that imports it. `null` is JSON, and it
    says the true thing: this one was not measured.

    The sanitising happens here, at the boundary, and not in the functions that compute the
    numbers. `float("nan")` is a perfectly good return value for a fit that had too few points, and
    the printed summary below still uses `np.nanmean` over the real values.
    """
    return {
        key: None if isinstance(value, float) and not math.isfinite(value) else value
        for key, value in record.items()
    }


LIFTS = {"identity": identity_lift, "rbf": rbf_lift}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    records = []
    with track.run("reproduce", {"seeds": args.seeds}) as tracker:
        for seed in range(args.seeds):
            # The known-answer half: identity dictionary on a linear system, exact or bust.
            a = linear_system(seed=seed)
            x, y = linear_trajectory(a, steps=400, seed=seed)
            fitted = fit_operator(x, y, identity_lift)
            row = {
                "seed": seed,
                "plant": "linear",
                "lift": "identity",
                "op_err": operator_error(fitted, a),
                "spec_err": spectrum_error(fitted, a),
            }
            records.append(row)
            tracker.log(row)

            # The MuJoCo half: no closed form, so the question is comparative.
            px, _, py = plant.rollout(plant.PENDULUM, trajectories=60, steps=80, seed=seed)
            start = np.array([2.0, 0.0])
            truth = plant.trajectory(plant.PENDULUM, start, 40)
            for name, lift in LIFTS.items():
                operator = fit_operator(px, py, lift)
                lifted = lift(px)
                row = {
                    "seed": seed,
                    "plant": "mujoco-pendulum",
                    "lift": name,
                    "dims": int(lifted.shape[0]),
                    # Chapter 8, and the number that decides whether a bigger dictionary is helping.
                    "condition": float(np.linalg.cond(lifted)),
                    "pred_err": multi_step_error(predict(operator, start, 40, lift), truth),
                }
                records.append(row)
                tracker.log(row)

    with open("results.json", "w") as handle:
        json.dump(
            {"assumptions": ASSUMPTIONS, "records": [jsonable(r) for r in records]},
            handle,
            indent=2,
            allow_nan=False,
        )

    print(f"\n{'plant':>17} {'lift':>9} {'dims':>5} {'error':>11} {'condition':>11}")
    for plant_name in ("linear", "mujoco-pendulum"):
        for name in LIFTS:
            rows = [r for r in records if r["plant"] == plant_name and r["lift"] == name]
            if not rows:
                continue
            err = np.mean([r.get("pred_err", r.get("op_err", np.nan)) for r in rows])
            cond = np.mean([r.get("condition", np.nan) for r in rows])
            print(f"{plant_name:>17} {name:>9} {rows[0].get('dims', 0):>5} {err:>11.3e} {cond:>11.3e}")
    print("\nresults.json written, and the sweep is in ./wandb (run `wandb sync` to upload).")


if __name__ == "__main__":
    main()
