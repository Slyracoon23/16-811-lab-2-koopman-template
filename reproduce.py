"""Fit the operator across seeds and horizons, and write results.json."""

from __future__ import annotations

import argparse
import json

import numpy as np

import baselines
from evaluate import multi_step_error, operator_error, spectrum_error
from method import ASSUMPTIONS, fit_operator, identity_lift, predict
from synthetic import linear_system, linear_trajectory

HORIZONS = [1, 5, 10, 25, 50]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    records = []
    for seed in range(args.seeds):
        a = linear_system(seed=seed)
        x, y = linear_trajectory(a, steps=400, seed=seed)
        fitted = fit_operator(x, y, identity_lift)
        floor = baselines.zero_operator(x)

        for horizon in HORIZONS:
            x0 = x[:, 0]
            truth = np.array([np.linalg.matrix_power(a, k + 1) @ x0 for k in range(horizon)]).T
            records.append(
                {
                    "seed": seed,
                    "horizon": horizon,
                    "pred_err": multi_step_error(predict(fitted, x0, horizon, identity_lift), truth),
                    "op_err": operator_error(fitted, a),
                    "spec_err": spectrum_error(fitted, a),
                    "floor_op_err": operator_error(floor, a),
                }
            )

    with open("results.json", "w") as handle:
        json.dump({"assumptions": ASSUMPTIONS, "records": records}, handle, indent=2)

    print(f"{'horizon':>8} {'pred err':>10} {'op err':>10} {'spec err':>10} {'floor':>8}")
    for horizon in HORIZONS:
        rows = [r for r in records if r["horizon"] == horizon]
        print(
            f"{horizon:>8} {np.mean([r['pred_err'] for r in rows]):>10.3e} "
            f"{np.mean([r['op_err'] for r in rows]):>10.3e} "
            f"{np.mean([r['spec_err'] for r in rows]):>10.3e} "
            f"{np.mean([r['floor_op_err'] for r in rows]):>8.2f}"
        )
    print("\nresults.json written. On a linear system every column should read ~1e-15 once method.py is yours.")


if __name__ == "__main__":
    main()
