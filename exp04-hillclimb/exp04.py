"""Experiment 4: hill climbing, local maxima and random restarts."""
import json
import os
import random
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
start_run("exp04-hillclimb")

# ---- Part 1: a 1-D landscape and greedy climbing from a few starts ----------
xs = np.linspace(0, 10, 2001)
f = lambda x: np.sin(x) + 0.6 * np.sin(2.7 * x) + 0.35 * np.sin(5.1 * x) - 0.05 * (x - 5) ** 2
fx = f(xs)
peaks = [i for i in range(1, len(xs) - 1) if fx[i] > fx[i - 1] and fx[i] > fx[i + 1]]
g = int(np.argmax(fx))


def climb_1d(i):
    while True:
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(xs)]
        j = max(nb, key=lambda k: fx[k])
        if fx[j] <= fx[i]:
            return i
        i = j


rnd = random.Random(SEED)
starts = sorted(rnd.sample(range(len(xs)), 6))
ends = [climb_1d(s) for s in starts]
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(xs, fx, lw=1.1, color="0.15", label="f(x)")
ax.plot(xs[peaks], fx[peaks], "o", mfc="none", mec="tab:orange", ms=8, label=f"local maxima ({len(peaks)})")
ax.plot(xs[g], fx[g], "*", color="tab:red", ms=14, label=f"global maximum x = {xs[g]:.2f}")
for s, e in zip(starts, ends):
    ax.annotate("", xy=(xs[e], fx[e]), xytext=(xs[s], fx[s]),
                arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.8))
ax.set_xlabel("x (dimensionless)"); ax.set_ylabel("Objective f(x)")
ax.set_title("Objective landscape; arrows = greedy climbs from 6 random starts")
ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)
stamp(fig); fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig4_1_landscape.png"), dpi=150); plt.close(fig)
print(f"{len(peaks)} local maxima; global at x = {xs[g]:.2f}; "
      f"{sum(e == g for e in ends)} of 6 climbs reached it")


# ---- Part 2: hill climbing on eight queens --------------------------------
def attacks(board):
    n, bad = len(board), 0
    for i in range(n):
        for j in range(i + 1, n):
            if board[i] == board[j] or abs(board[i] - board[j]) == j - i:
                bad += 1
    return bad


def hill_climb(n, r):
    """Steepest descent on the number of attacking pairs; stops at the first non-improving step."""
    board = [r.randrange(n) for _ in range(n)]
    steps = 0
    while True:
        cur = attacks(board)
        if cur == 0:
            return True, steps
        best, move = cur, None
        for c in range(n):
            old = board[c]
            for row in range(n):
                if row != old:
                    board[c] = row
                    v = attacks(board)
                    if v < best:
                        best, move = v, (c, row)
            board[c] = old
        if move is None:                      # local minimum or plateau: stuck
            return False, steps
        board[move[0]] = move[1]
        steps += 1


def with_restarts(n, k, r):
    return any(hill_climb(n, r)[0] for _ in range(k))


rng = random.Random(SEED * 7919)
STARTS = 500
runs = [hill_climb(8, rng) for _ in range(STARTS)]
succ = sum(ok for ok, _ in runs)
p = succ / STARTS
mean_steps_ok = np.mean([s for ok, s in runs if ok])
mean_steps_fail = np.mean([s for ok, s in runs if not ok])
print(f"plain: {succ}/{STARTS} = {100 * p:.1f}%  (mean steps success {mean_steps_ok:.1f}, failure {mean_steps_fail:.1f})")

TRIALS = 200
table = []
for k in (1, 2, 5, 10, 20):
    ok = sum(with_restarts(8, k, rng) for _ in range(TRIALS))
    pred = 1 - (1 - p) ** k
    table.append(dict(k=k, trials=TRIALS, successes=ok, measured=ok / TRIALS, predicted=pred))
    print(f"k = {k:>2}: {ok}/{TRIALS} = {100 * ok / TRIALS:5.1f}%   predicted {100 * pred:5.1f}%")
k99 = int(np.ceil(np.log(0.01) / np.log(1 - p)))
print("restarts needed for 99%:", k99)

fig, ax = plt.subplots(figsize=(7, 4))
ks = [t["k"] for t in table]
ax.plot(ks, [100 * t["measured"] for t in table], "o-", label=f"measured ({TRIALS} trials each)")
kk = np.arange(1, 21)
ax.plot(kk, 100 * (1 - (1 - p) ** kk), "--", label=f"predicted 1-(1-p)^k, p = {100 * p:.1f}%")
ax.set_xlabel("Number of random restarts k"); ax.set_ylabel("Success rate (%)")
ax.set_title("Eight queens hill climbing: success against restarts"); ax.set_ylim(0, 105)
ax.legend(); ax.grid(alpha=0.3)
stamp(fig); fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig4_2_restarts.png"), dpi=150); plt.close(fig)

with open(os.path.join(HERE, "results.json"), "w") as fh:
    json.dump(dict(n_local_maxima=len(peaks), x_global=float(xs[g]), plain_starts=STARTS, plain_success=succ,
                   p=p, mean_steps_ok=float(mean_steps_ok), mean_steps_fail=float(mean_steps_fail),
                   restarts=table, k99=k99), fh, indent=2)
