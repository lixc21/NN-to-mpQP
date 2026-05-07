"""2D example: build a ReLU net, convert it to an mpQP, and plot both surfaces."""
import os

import numpy as np

from nn_to_mpqp import random_relu_network, relu_network_to_mpqp, solve_mpqp_at


def main() -> None:
    net = random_relu_network(n_input=2, hidden_widths=[6, 6], n_output=1, seed=7)
    mpqp = relu_network_to_mpqp(net, B0=1.0)
    print(f"mpQP: n_x={mpqp.n_x}, n_z={mpqp.n_z}, n_c={mpqp.n_c}, M={mpqp.M}")

    grid = np.linspace(-1.0, 1.0, 41)
    XX, YY = np.meshgrid(grid, grid)
    F_nn = np.zeros_like(XX)
    F_mpqp = np.zeros_like(XX)
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            x = np.array([XX[i, j], YY[i, j]])
            F_nn[i, j] = net.forward(x)[0]
            z_star = solve_mpqp_at(mpqp, x)
            z_L = z_star[sum(mpqp.layer_dims[:-1]):]
            F_mpqp[i, j] = (net.weights[-1] @ z_L + net.biases[-1])[0]

    diff = float(np.max(np.abs(F_nn - F_mpqp)))
    print(f"max |F_nn - F_mpqp| over the 41x41 grid = {diff:.3e}")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, data, title in zip(
        axes,
        [F_nn, F_mpqp, F_nn - F_mpqp],
        ["NN forward", "mpQP solution", "difference"],
    ):
        im = ax.imshow(data, extent=[-1, 1, -1, 1], origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("x_1")
        ax.set_ylabel("x_2")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_2d.png")
    fig.savefig(out_path, dpi=140)
    print(f"figure saved to {out_path}")


if __name__ == "__main__":
    main()
