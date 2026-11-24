"""Experiment 5: tabular Q-learning on a grid world, effect of epsilon."""
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
start_run("exp05-rl")

H, W = 5, 7
WALLS = {(1, 2), (2, 2), (3, 2), (1, 4), (2, 4)}
GOAL, TRAP, START = (0, W - 1), (2, W - 1), (H - 1, 0)
R_GOAL, R_TRAP, R_STEP = 10.0, -10.0, -0.1
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]          # up, down, left, right
ALPHA, GAMMA, EPISODES, MAX_STEPS = 0.5, 0.95, 800, 120
EPSILONS = (0.01, 0.10, 0.20, 0.50)


def step(s, a):
    r, c = s
    dr, dc = ACTIONS[a]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < H and 0 <= nc < W) or (nr, nc) in WALLS:
        nr, nc = r, c                                   # bump: stay in place
    ns = (nr, nc)
    if ns == GOAL:
        return ns, R_GOAL, True
    if ns == TRAP:
        return ns, R_TRAP, True
    return ns, R_STEP, False


def train(eps, seed):
    rng = np.random.default_rng(seed)
    Q = np.zeros((H, W, 4))
    returns = []
    for _ in range(EPISODES):
        s, G = START, 0.0
        for _ in range(MAX_STEPS):
            if rng.random() < eps:
                a = int(rng.integers(4))                # explore
            else:
                q = Q[s]
                a = int(rng.choice(np.flatnonzero(q == q.max())))   # greedy, random tie-break
            ns, r, done = step(s, a)
            target = r + (0.0 if done else GAMMA * Q[ns].max())    # no future value at terminal
            Q[s][a] += ALPHA * (target - Q[s][a])
            G += r
            s = ns
            if done:
                break
        returns.append(G)
    return Q, np.array(returns)


def greedy_path(Q):
    s, path = START, [START]
    for _ in range(50):
        s, _, done = step(s, int(np.argmax(Q[s])))
        path.append(s)
        if done:
            break
    return path


# value iteration gives the optimal policy to compare against
V = np.zeros((H, W))
for _ in range(500):
    for r in range(H):
        for c in range(W):
            if (r, c) in WALLS or (r, c) in (GOAL, TRAP):
                continue
            V[r, c] = max(rew + (0 if d else GAMMA * V[ns]) for ns, rew, d in (step((r, c), a) for a in range(4)))
opt_len = None
s, n = START, 0
while s not in (GOAL, TRAP) and n < 50:
    s = max((step(s, a) for a in range(4)), key=lambda t: t[1] + (0 if t[2] else GAMMA * V[t[0]]))[0]
    n += 1
opt_len = n

res, curves, Qs = {}, {}, {}
for eps in EPSILONS:
    Q, ret = train(eps, SEED)
    path = greedy_path(Q)
    res[eps] = dict(mean_last100=float(ret[-100:].mean()), greedy_steps=len(path) - 1,
                    greedy_reaches_goal=path[-1] == GOAL)
    curves[eps], Qs[eps] = ret, Q
    print(f"eps = {eps:.2f}: mean return last 100 = {ret[-100:].mean():6.2f}; greedy path "
          f"{len(path) - 1} steps, reaches goal = {path[-1] == GOAL}")
print("optimal path length (value iteration):", opt_len)

k = 25
fig, ax = plt.subplots(figsize=(9, 3.6))
for eps in EPSILONS:
    sm = np.convolve(curves[eps], np.ones(k) / k, mode="valid")
    ax.plot(np.arange(k, EPISODES + 1), sm, lw=1.1, label=f"ε = {eps}")
ax.set_xlabel("Episode"); ax.set_ylabel(f"Return per episode ({k}-episode moving mean)")
ax.set_title("Q-learning learning curves for four exploration rates"); ax.legend(); ax.grid(alpha=0.3)
stamp(fig); fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig5_1_learning.png"), dpi=150); plt.close(fig)

BEST = 0.10
Q = Qs[BEST]
pol = np.argmax(Q, axis=2)
gp = set(greedy_path(Q))
arrows = ["↑", "↓", "←", "→"]
grid_txt = []
fig, ax = plt.subplots(figsize=(6.4, 4.8))
for r in range(H):
    line = []
    for c in range(W):
        if (r, c) in WALLS:
            ax.add_patch(plt.Rectangle((c, r), 1, 1, color="0.25")); line.append("#")
        elif (r, c) == GOAL:
            ax.add_patch(plt.Rectangle((c, r), 1, 1, color="tab:green", alpha=0.6))
            ax.text(c + .5, r + .5, "G\n+10", ha="center", va="center", fontsize=8); line.append("G")
        elif (r, c) == TRAP:
            ax.add_patch(plt.Rectangle((c, r), 1, 1, color="tab:red", alpha=0.6))
            ax.text(c + .5, r + .5, "T\n-10", ha="center", va="center", fontsize=8); line.append("T")
        else:
            if (r, c) in gp:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color="gold", alpha=0.35))
            ax.text(c + .5, r + .5, arrows[pol[r, c]], ha="center", va="center", fontsize=16)
            line.append("^v<>"[pol[r, c]])
    grid_txt.append(" ".join(line))
ax.text(START[1] + .08, START[0] + .9, "S", fontsize=8, color="tab:blue")
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
ax.set_xticks(np.arange(W) + .5, range(W)); ax.set_yticks(np.arange(H) + .5, range(H))
ax.set_xlabel("Column (cells)"); ax.set_ylabel("Row (cells)")
ax.set_title(f"Greedy policy after training (ε = {BEST}); shaded = path from S")
for i in range(W + 1): ax.axvline(i, color="0.7", lw=.5)
for i in range(H + 1): ax.axhline(i, color="0.7", lw=.5)
stamp(fig); fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig5_2_policy.png"), dpi=150); plt.close(fig)
print("\n".join(grid_txt))

with open(os.path.join(HERE, "results.json"), "w") as fh:
    json.dump(dict(grid=[H, W], alpha=ALPHA, gamma=GAMMA, episodes=EPISODES, seed=SEED,
                   results={str(k_): v for k_, v in res.items()}, optimal_steps=opt_len,
                   policy_grid=grid_txt, best_eps=BEST), fh, indent=2)
