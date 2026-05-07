"""Random tests across input dimensions, with fixed seeds for reproducibility.

Run with `pytest test_random.py -q` or directly: `python test_random.py`.
"""
from __future__ import annotations

import numpy as np
import pytest

from nn_to_mpqp import random_relu_network, verify_network


# (n_input, hidden_widths, n_output, seed)
CASES = [
    (1, [3],         1, 0),
    (1, [4, 4],      1, 1),
    (2, [5, 5],      1, 2),
    (2, [6, 6],      2, 3),
    (3, [6, 6, 6],   2, 4),
    (4, [8, 8, 8],   3, 5),
    (5, [10, 10],    2, 6),
]


@pytest.mark.parametrize("n_in,widths,n_out,seed", CASES)
def test_random_network(n_in, widths, n_out, seed):
    net = random_relu_network(n_in, widths, n_out, seed=seed, scale=0.7)
    summary = verify_network(net, B0=1.0, n_test=30, tol=1e-4, seed=42)
    assert summary["passed"], summary


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    failures = 0
    for n_in, widths, n_out, seed in CASES:
        label = f"n_in={n_in}, hidden={widths}, n_out={n_out}, seed={seed}"
        print(f"\n--- {label} ---")
        net = random_relu_network(n_in, widths, n_out, seed=seed, scale=0.7)
        summary = verify_network(net, B0=1.0, n_test=30, tol=1e-4, seed=42, verbose=True)
        if not summary["passed"]:
            failures += 1
    print()
    print("ALL PASSED" if failures == 0 else f"{failures} FAILED")
    raise SystemExit(failures)
