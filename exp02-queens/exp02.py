"""Experiment 2: the eight queens problem (generate-and-test vs backtracking)."""
import json
import os
import sys
import time
from itertools import permutations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from student import cpu_name, start_run, stamp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)
start_run("exp02-queens")


def no_diagonal_clash(board):
    """board[c] = row of the queen in column c. Rows are distinct by construction."""
    n = len(board)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(board[i] - board[j]) == j - i:
                return False
    return True


def generate_and_test(n):
    return [p for p in permutations(range(n)) if no_diagonal_clash(p)]


def is_safe(board, col, row):
    """Check the new queen only against the queens already placed (columns < col)."""
    for c in range(col):
        if board[c] == row or abs(board[c] - row) == col - c:
            return False
    return True


def backtracking(n):
    sols, nodes = [], 0

    def place(board, col):
        nonlocal nodes
        if col == n:
            sols.append(tuple(board))
            return
        for row in range(n):
            nodes += 1                      # every (col, row) tried is one node
            if is_safe(board, col, row):
                board[col] = row
                place(board, col + 1)
                board[col] = -1

    place([-1] * n, 0)
    return sols, nodes


rows = []
print(f"{'n':>3}{'solutions':>11}{'nodes':>10}{'backtrack s':>13}{'gen&test s':>12}  same?")
for n in range(4, 11):
    t0 = time.perf_counter(); sb, nodes = backtracking(n); tb = time.perf_counter() - t0
    t0 = time.perf_counter(); sg = generate_and_test(n); tg = time.perf_counter() - t0
    same = sorted(sb) == sorted(sg)
    rows.append(dict(n=n, solutions=len(sb), solutions_gt=len(sg), nodes=nodes, t_bt=tb, t_gt=tg, same=same))
    print(f"{n:>3}{len(sb):>11}{nodes:>10}{tb:>13.4f}{tg:>12.4f}  {same}")

first8 = backtracking(8)[0][0]
print("first solution for n = 8 (row of queen in each column):", first8)
print("CPU:", cpu_name())

fig, ax = plt.subplots(figsize=(7, 4))
ns = [r["n"] for r in rows]
ax.semilogy(ns, [r["t_gt"] for r in rows], "o-", label="generate and test (permutations)")
ax.semilogy(ns, [r["t_bt"] for r in rows], "s-", label="backtracking")
ax.set_xlabel("Board size n (queens)"); ax.set_ylabel("Run time (s, log scale)")
ax.set_title("N-queens: run time against board size"); ax.legend(); ax.grid(alpha=0.3, which="both")
stamp(fig); fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig2_2_runtime.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 4.6))
for r in range(8):
    for c in range(8):
        ax.add_patch(plt.Rectangle((c, r), 1, 1, color="0.85" if (r + c) % 2 else "white"))
for c, r in enumerate(first8):
    ax.text(c + 0.5, r + 0.5, "♛", ha="center", va="center", fontsize=20)
ax.set_xlim(0, 8); ax.set_ylim(8, 0); ax.set_aspect("equal")
ax.set_xticks([i + 0.5 for i in range(8)], range(8)); ax.set_yticks([i + 0.5 for i in range(8)], range(8))
ax.set_xlabel("Column index c"); ax.set_ylabel("Row index r")
ax.set_title(f"First solution, n = 8: {list(first8)}", fontsize=9)
stamp(fig); fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig2_1_board.png"), dpi=150); plt.close(fig)

with open(os.path.join(HERE, "results.json"), "w") as fh:
    json.dump(dict(rows=rows, first8=first8, cpu=cpu_name()), fh, indent=2)
