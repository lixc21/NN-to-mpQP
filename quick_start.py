"""Minimal usage example for nn_to_mpqp."""
import numpy as np

from nn_to_mpqp import (
    random_relu_network,
    relu_network_to_mpqp,
    solve_mpqp_at,
)


def main() -> None:
    # 1. Build (or load) a ReLU network.
    net = random_relu_network(n_input=2, hidden_widths=[6, 6], n_output=1, seed=0)

    # 2. Convert it to an equivalent mpQP for inputs with ||x||_inf <= 1.
    mpqp = relu_network_to_mpqp(net, B0=1.0)
    print(f"mpQP: n_x={mpqp.n_x}, n_z={mpqp.n_z}, n_c={mpqp.n_c}, M={mpqp.M}")

    # 3. Solve the mpQP at one parameter and recover the network output.
    x = np.array([0.3, -0.2])
    z_star = solve_mpqp_at(mpqp, x)
    z_L = z_star[sum(mpqp.layer_dims[:-1]):]
    y_mpqp = net.weights[-1] @ z_L + net.biases[-1]
    y_nn = net.forward(x)

    print(f"x       = {x}")
    print(f"NN(x)   = {y_nn}")
    print(f"mpQP(x) = {y_mpqp}")
    print(f"|diff|  = {np.max(np.abs(y_nn - y_mpqp)):.2e}")


if __name__ == "__main__":
    main()
