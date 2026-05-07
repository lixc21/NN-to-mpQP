"""
NN -> mpQP construction (Lemma 4 of the accompanying paper).

Given an L-layer ReLU network
    h_0 = x,
    h_l = ReLU(W_l h_{l-1} + b_l),   l = 1, ..., L,
with bounded input ||x||_inf <= B0, this module builds the equivalent
multi-parametric quadratic program (mpQP)

    min_{z_1, ..., z_L}   sum_{l=1}^L 0.5 * || z_l + M_l * 1 ||^2
    s.t.                  z_l >= 0,                          l = 1, ..., L
                          z_l >= W_l z_{l-1} + b_l,          l = 1, ..., L
    where z_0 := x is the parameter.

The constants {M_l} are determined by a forward bound-propagation pass and a
backward dual-bound recursion so that, by KKT, the unique optimizer satisfies
    z_l^*(x) = h_l(x)   for all l.

The mpQP is returned in the standard form
    min_z   0.5 z' Q z + c' z + x' H z
    s.t.    A z <= b + S x,
i.e. with parameter coupling appearing only in the right-hand side via S, and
no x-z coupling in the objective (H = 0).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------

@dataclass
class ReLUNetwork:
    """A feed-forward ReLU network with a final linear (no-ReLU) output layer.

    weights[l] has shape (d_{l+1}, d_l); biases[l] has shape (d_{l+1},).
    The last entry is the linear output layer.
    """
    weights: List[np.ndarray]
    biases:  List[np.ndarray]

    @property
    def n_input(self) -> int:
        return self.weights[0].shape[1]

    @property
    def n_output(self) -> int:
        return self.weights[-1].shape[0]

    @property
    def n_hidden_layers(self) -> int:
        return len(self.weights) - 1

    @property
    def hidden_dims(self) -> List[int]:
        return [self.weights[l].shape[0] for l in range(self.n_hidden_layers)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.asarray(x, dtype=float)
        for l in range(self.n_hidden_layers):
            h = np.maximum(0.0, self.weights[l] @ h + self.biases[l])
        return self.weights[-1] @ h + self.biases[-1]

    def forward_hidden(self, x: np.ndarray) -> List[np.ndarray]:
        """Return the list [h_1, ..., h_L] of hidden activations for input x."""
        h = np.asarray(x, dtype=float)
        out = []
        for l in range(self.n_hidden_layers):
            h = np.maximum(0.0, self.weights[l] @ h + self.biases[l])
            out.append(h)
        return out


@dataclass
class MPQP:
    """Standard-form mpQP:  min  0.5 z' Q z + c' z + x' H z
                           s.t.  A z <= b + S x.
    """
    Q: np.ndarray   # (n_z, n_z)
    H: np.ndarray   # (n_x, n_z)  -- always zeros for our construction
    c: np.ndarray   # (n_z,)
    A: np.ndarray   # (n_c, n_z)
    b: np.ndarray   # (n_c,)
    S: np.ndarray   # (n_c, n_x)
    layer_dims: List[int]   # dimensions [d_1, ..., d_L] of hidden layers
    M: List[float]          # constants M_l from Lemma 4
    B0: float               # input bound used during construction

    @property
    def n_x(self) -> int:
        return self.S.shape[1]

    @property
    def n_z(self) -> int:
        return self.Q.shape[0]

    @property
    def n_c(self) -> int:
        return self.A.shape[0]


# ----------------------------------------------------------------------------
# The two recursive passes from Lemma 4
# ----------------------------------------------------------------------------

def forward_activation_bounds(
    weights: Sequence[np.ndarray],
    biases:  Sequence[np.ndarray],
    B0: float,
) -> List[np.ndarray]:
    """Compute interval bounds  U_l >= h_l(x)  for all ||x||_inf <= B0.

    Uses the standard interval propagation
        U_l = ReLU(|W_l| U_{l-1} + |b_l|),   U_0 = B0 * 1.
    """
    n_input = weights[0].shape[1]
    U_prev = B0 * np.ones(n_input)
    U: List[np.ndarray] = []
    for W, b in zip(weights, biases):
        U_l = np.maximum(0.0, np.abs(W) @ U_prev + np.abs(b))
        U.append(U_l)
        U_prev = U_l
    return U


def backward_M_constants(
    all_weights: Sequence[np.ndarray],
    U: Sequence[np.ndarray],
) -> Tuple[List[float], List[np.ndarray]]:
    """Backward recursion for M_l and dual upper bounds bar_beta_l.

    all_weights = [W_1, ..., W_L, W_{L+1}], including the linear output layer.
    U           = [U_1, ..., U_L]            from forward_activation_bounds.

    Returns (M, bar_beta) of length L.
    """
    L = len(U)
    n_output = all_weights[-1].shape[0]

    M: List[float] = [0.0] * L
    bar_beta: List[np.ndarray] = [None] * L  # type: ignore[list-item]

    # bar_beta_{L+1} = 0 of dimension equal to output layer.
    bar_beta_next = np.zeros(n_output)

    for l in range(L - 1, -1, -1):
        W_next = all_weights[l + 1]                    # the layer AFTER hidden l
        W_next_T = W_next.T                            # (d_l, d_{l+1})
        W_pos = np.maximum(W_next_T, 0.0)
        W_neg = np.maximum(-W_next_T, 0.0)

        neg_term = W_neg @ bar_beta_next
        # ||(W_{l+1}^T)_- bar_beta_{l+1}||_inf
        worst_neg = float(np.max(neg_term)) if neg_term.size else 0.0
        M_l = max(1.0, worst_neg + 1.0)
        M[l] = M_l

        bar_beta[l] = U[l] + M_l * np.ones_like(U[l]) + W_pos @ bar_beta_next
        bar_beta_next = bar_beta[l]

    return M, bar_beta


# ----------------------------------------------------------------------------
# Main construction
# ----------------------------------------------------------------------------

def relu_network_to_mpqp(network: ReLUNetwork, B0: float = 1.0) -> MPQP:
    """Build the mpQP from Lemma 4 for the *hidden* layers of `network`.

    The mpQP encodes z_l = h_l(x) for every hidden layer l = 1, ..., L.
    The network's final linear layer is *not* embedded as an mpQP variable
    (it has no ReLU); to recover F(x) compose W_{L+1} z_L^* + b_{L+1}.
    """
    W = list(network.weights)
    b = list(network.biases)
    L = network.n_hidden_layers
    n_input = network.n_input
    layer_dims = network.hidden_dims
    n_z = sum(layer_dims)

    # Interval bounds for the hidden activations.
    U = forward_activation_bounds(W[:L], b[:L], B0)
    # Dual constants (need the output layer to compute M_L).
    M, _ = backward_M_constants(W[: L + 1], U)

    # Quadratic part: sum_l 0.5 ||z_l + M_l 1||^2 = 0.5 z'z + (M.*1)'z + const.
    Q = np.eye(n_z)
    c = np.zeros(n_z)
    off = 0
    for l in range(L):
        d = layer_dims[l]
        c[off : off + d] = M[l]
        off += d

    # Constraints: per layer two blocks of d_l rows.
    n_c = 2 * n_z
    A = np.zeros((n_c, n_z))
    b_vec = np.zeros(n_c)
    S = np.zeros((n_c, n_input))

    row = 0
    col = 0
    for l in range(L):
        d = layer_dims[l]

        # Block 1:  z_l >= 0   <=>   -I z_l <= 0.
        A[row : row + d, col : col + d] = -np.eye(d)
        # b_vec[row:row+d] = 0,  S contributes 0
        row += d

        # Block 2:  z_l >= W_l z_{l-1} + b_l   <=>   -I z_l + W_l z_{l-1} <= -b_l
        A[row : row + d, col : col + d] = -np.eye(d)
        if l == 0:
            # z_{l-1} = x is the parameter.
            S[row : row + d, :] = -W[l]
        else:
            prev_col = col - layer_dims[l - 1]
            A[row : row + d, prev_col : prev_col + layer_dims[l - 1]] = W[l]
        b_vec[row : row + d] = -b[l]
        row += d

        col += d

    H = np.zeros((n_input, n_z))  # no x-z coupling in objective

    return MPQP(
        Q=Q, H=H, c=c, A=A, b=b_vec, S=S,
        layer_dims=layer_dims, M=list(M), B0=B0,
    )


# ----------------------------------------------------------------------------
# Reference solver (scipy SLSQP) for verification
# ----------------------------------------------------------------------------

def solve_mpqp_at(
    mpqp: MPQP,
    x: np.ndarray,
    z0: np.ndarray | None = None,
    method: str = "auto",
):
    """Solve the mpQP at a single parameter `x`.

    Parameters
    ----------
    method : {"auto", "quadprog", "trust-constr", "slsqp"}
        ``"auto"`` (default) tries ``quadprog`` first (an exact dense QP
        solver) and falls back to ``trust-constr``. The other names force a
        specific solver. Returns the optimal `z*` or raises ``RuntimeError``.
    """
    Q, c, A, b, S = mpqp.Q, mpqp.c, mpqp.A, mpqp.b, mpqp.S
    rhs = b + S @ x  # constraint:  A z <= rhs

    if method in ("auto", "quadprog"):
        try:
            import quadprog  # type: ignore
            # quadprog solves:  min 0.5 z' G z - a' z  s.t.  C' z >= b
            # -> G = Q (positive definite),  a = -c,  C = -A^T,  b = -rhs
            G = (Q + Q.T) / 2.0  # ensure symmetry for the cholesky inside
            sol = quadprog.solve_qp(G, -c, -A.T, -rhs)
            return np.asarray(sol[0])
        except ImportError:
            if method == "quadprog":
                raise
        except ValueError as exc:
            if method == "quadprog":
                raise RuntimeError(f"quadprog failed: {exc}") from exc
            # else fall through to trust-constr
    if method in ("auto", "trust-constr"):
        from scipy.optimize import LinearConstraint, minimize

        def obj(z):  return 0.5 * z @ Q @ z + c @ z
        def grad(z): return Q @ z + c
        def hess(z): return Q

        if z0 is None:
            z0 = np.zeros(mpqp.n_z)
        lin_con = LinearConstraint(A, lb=-np.inf, ub=rhs)
        res = minimize(
            obj, z0, jac=grad, hess=hess,
            constraints=[lin_con], method="trust-constr",
            options={
                "xtol": 1e-14, "gtol": 1e-12, "maxiter": 5000,
                "initial_barrier_parameter": 1e-10, "verbose": 0,
            },
        )
        if not res.success:
            raise RuntimeError(f"trust-constr failed: {res.message}")
        return res.x

    if method == "slsqp":
        from scipy.optimize import minimize
        def obj(z):  return 0.5 * z @ Q @ z + c @ z
        def grad(z): return Q @ z + c
        if z0 is None:
            z0 = np.zeros(mpqp.n_z)
        cons = {
            "type": "ineq",
            "fun":  lambda z: rhs - A @ z,
            "jac":  lambda z: -A,
        }
        res = minimize(
            obj, z0, jac=grad, constraints=cons, method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not res.success:
            raise RuntimeError(f"SLSQP failed: {res.message}")
        return res.x

    raise ValueError(f"Unknown solver method: {method!r}")


def verify_network(
    network: ReLUNetwork,
    B0: float = 1.0,
    n_test: int = 100,
    tol: float = 1e-4,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Check that the mpQP optimum reproduces every hidden activation h_l(x).

    Returns a dict with max/mean errors and a boolean `passed`.
    """
    mpqp = relu_network_to_mpqp(network, B0=B0)
    rng  = np.random.default_rng(seed)

    layer_max  = np.zeros(network.n_hidden_layers)
    layer_mean = np.zeros(network.n_hidden_layers)
    out_errors = []

    for k in range(n_test):
        x = rng.uniform(-B0, B0, size=network.n_input)
        h_true = network.forward_hidden(x)
        # Warm-start at the network's own activations (the known optimum).
        z0 = np.concatenate(h_true) if h_true else np.zeros(mpqp.n_z)
        z_star = solve_mpqp_at(mpqp, x, z0=z0)

        # Slice z* into per-layer pieces and compare.
        off = 0
        for l, d in enumerate(mpqp.layer_dims):
            z_l = z_star[off : off + d]
            err = float(np.max(np.abs(z_l - h_true[l])))
            layer_max[l]  = max(layer_max[l], err)
            layer_mean[l] += err
            off += d

        # Compose linear output for an end-to-end check.
        z_L = z_star[sum(mpqp.layer_dims[:-1]) :]
        y_mpqp = network.weights[-1] @ z_L + network.biases[-1]
        y_true = network.forward(x)
        out_errors.append(float(np.max(np.abs(y_mpqp - y_true))))

    layer_mean /= max(n_test, 1)
    summary = {
        "n_test":          n_test,
        "B0":              B0,
        "M":               mpqp.M,
        "layer_dims":      mpqp.layer_dims,
        "n_x":             mpqp.n_x,
        "n_z":             mpqp.n_z,
        "n_c":             mpqp.n_c,
        "layer_max_error": layer_max.tolist(),
        "layer_mean_error": layer_mean.tolist(),
        "output_max_error": float(max(out_errors)),
        "output_mean_error": float(np.mean(out_errors)),
        "passed":          bool(max(out_errors) < tol),
    }
    if verbose:
        print(f"[verify] n_x={summary['n_x']}, n_z={summary['n_z']}, "
              f"n_c={summary['n_c']}, M={['%.2f'%m for m in summary['M']]}")
        print(f"[verify] layer_max  = {summary['layer_max_error']}")
        print(f"[verify] output_max = {summary['output_max_error']:.3e}  "
              f"=> {'PASSED' if summary['passed'] else 'FAILED'}")
    return summary


# ----------------------------------------------------------------------------
# Convenience network builders
# ----------------------------------------------------------------------------

def random_relu_network(
    n_input: int,
    hidden_widths: Sequence[int],
    n_output: int,
    seed: int = 0,
    scale: float = 1.0,
) -> ReLUNetwork:
    rng = np.random.default_rng(seed)
    dims = [n_input, *hidden_widths, n_output]
    Ws, bs = [], []
    for i in range(len(dims) - 1):
        fan_in = dims[i]
        Ws.append(rng.standard_normal((dims[i + 1], dims[i])) * scale / np.sqrt(fan_in))
        bs.append(rng.standard_normal(dims[i + 1]) * 0.1)
    return ReLUNetwork(weights=Ws, biases=bs)


def tent_map_network(L: int) -> ReLUNetwork:
    """Deterministic tent-map network of depth L, width 2, scalar input/output.

    Implements the 2^L-piecewise-linear tent map on [0, 1] used in the paper.
    """
    Ws: List[np.ndarray] = []
    bs: List[np.ndarray] = []

    Ws.append(np.array([[2.0], [-2.0]]));  bs.append(np.array([0.0, 1.0]))
    for _ in range(L - 1):
        Ws.append(np.array([[-2.0, -4.0], [2.0, 4.0]]))
        bs.append(np.array([4.0, -3.0]))
    Ws.append(np.array([[-1.0, -2.0]]));   bs.append(np.array([2.0]))

    return ReLUNetwork(weights=Ws, biases=bs)
