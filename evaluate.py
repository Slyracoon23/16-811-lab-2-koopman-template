"""The ruler. Complete, and it does not import `method`.

An evaluator that has seen the answer cannot be trusted with it, so this file knows nothing about
how the operator was fitted.
"""

from __future__ import annotations

import argparse

import numpy as np

from synthetic import linear_system, linear_trajectory


def multi_step_error(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Mean Euclidean error per step over the horizon."""
    predicted, truth = np.asarray(predicted), np.asarray(truth)
    return float(np.mean(np.linalg.norm(predicted - truth, axis=0)))


def operator_error(estimated: np.ndarray, truth: np.ndarray) -> float:
    """Frobenius distance between an estimated operator and the true one."""
    return float(np.linalg.norm(np.asarray(estimated) - np.asarray(truth)))


def spectrum_error(estimated: np.ndarray, truth: np.ndarray) -> float:
    """Distance between sorted eigenvalues. The modes are the physics; this scores them."""
    a = np.sort_complex(np.linalg.eigvals(estimated))
    b = np.sort_complex(np.linalg.eigvals(truth))
    return float(np.max(np.abs(a - b)))


def self_test() -> None:
    a = linear_system(seed=0)
    zero = np.zeros_like(a)
    print(f"perfect operator error: {operator_error(a, a):.3e}   (expect 0)")
    print(f"zero    operator error: {operator_error(zero, a):.3f}     (expect ~1, the size of A)")
    print(f"perfect spectrum error: {spectrum_error(a, a):.3e}   (expect 0)")
    x, y = linear_trajectory(a, steps=50)
    print(f"perfect 1-step error  : {multi_step_error(a @ x, y):.3e}   (expect 0)")
    print("\nIf any 'perfect' line is not ~0 the ruler is wrong and every later number is fiction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    if parser.parse_args().self_test:
        self_test()
