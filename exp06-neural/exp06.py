"""Experiment 6: a 2-4-1 neural network trained on XOR with NumPy only."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from student import SEED, start_run, stamp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)
start_run("exp06-neural")

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)
HIDDEN, EPOCHS = 4, 20000
LRS = (0.01, 0.1, 1.0, 10.0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train_xor(lr, seed=SEED):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 1, (2, HIDDEN)); b1 = np.zeros((1, HIDDEN))
    W2 = rng.normal(0, 1, (HIDDEN, 1)); b2 = np.zeros((1, 1))
    loss = np.empty(EPOCHS)
    for e in range(EPOCHS):
        h = sigmoid(X @ W1 + b1)                 # forward pass
        o = sigmoid(h @ W2 + b2)
        loss[e] = np.mean((o - y) ** 2)
        do = (o - y) * o * (1 - o)               # output delta (sigmoid derivative)
        dh = (do @ W2.T) * h * (1 - h)           # back-propagated hidden delta
        W2 -= lr * h.T @ do; b2 -= lr * do.sum(0, keepdims=True)
        W1 -= lr * X.T @ dh; b1 -= lr * dh.sum(0, keepdims=True)
    return (W1, b1, W2, b2), loss


def predict(p, A):
    W1, b1, W2, b2 = p
    return sigmoid(sigmoid(A @ W1 + b1) @ W2 + b2)


runs = {}
for lr in LRS:
    p, loss = train_xor(lr)
    pred = predict(p, X).ravel()
    conv = bool(loss[-1] < 0.01 and np.all((pred > 0.5) == y.ravel().astype(bool)))
    below = np.flatnonzero(loss < 0.01)
    runs[lr] = dict(final_mse=float(loss[-1]), converged=conv, pred=pred.tolist(),
                    epoch_below_0_01=int(below[0]) if below.size else None, params=p, loss=loss)
    print(f"lr = {lr:>5}: final MSE = {loss[-1]:.2e}, converged = {conv}, "
          f"MSE < 0.01 at epoch {runs[lr]['epoch_below_0_01']}, predictions {np.round(pred, 3)}")

fig, ax = plt.subplots(figsize=(9, 3.6))
for lr in LRS:
    ax.semilogy(runs[lr]["loss"], lw=1.0, label=f"learning rate = {lr}")
ax.axhline(0.01, color="0.5", ls=":", lw=0.8, label="convergence threshold 0.01")
ax.set_xlabel("Epoch"); ax.set_ylabel("Mean squared error (log scale)")
ax.set_title("XOR training loss for four learning rates"); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
stamp(fig); fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig6_1_loss.png"), dpi=150); plt.close(fig)

best = min((lr for lr in LRS if runs[lr]["converged"]), key=lambda lr: runs[lr]["final_mse"], default=LRS[2])
gx, gy = np.meshgrid(np.linspace(-0.25, 1.25, 300), np.linspace(-0.25, 1.25, 300))
Z = predict(runs[best]["params"], np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
fig, ax = plt.subplots(figsize=(5.4, 4.4))
cs = ax.contourf(gx, gy, Z, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.8)
ax.contour(gx, gy, Z, levels=[0.5], colors="k", linewidths=1.5)
ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="bwr", edgecolors="k", s=90, zorder=3)
fig.colorbar(cs, ax=ax, label="Network output o")
ax.set_xlabel("Input x1"); ax.set_ylabel("Input x2")
ax.set_title(f"Decision boundary (o = 0.5), learning rate {best}")
stamp(fig); fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig6_2_boundary.png"), dpi=150); plt.close(fig)
print("best run: lr =", best)

with open(os.path.join(HERE, "results.json"), "w") as fh:
    json.dump(dict(seed=SEED, hidden=HIDDEN, epochs=EPOCHS, best_lr=best,
                   runs={str(k): {kk: vv for kk, vv in v.items() if kk not in ("params", "loss")}
                         for k, v in runs.items()}), fh, indent=2)
