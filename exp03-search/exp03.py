"""Experiment 3: breadth-first and depth-first search on a grid maze."""
import json
import os
import random
import sys
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from student import SEED, start_run, stamp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)
start_run("exp03-search")
sys.setrecursionlimit(10000)

# ---- build a maze: recursive-backtracker carving, then open some walls -----
H, W = 25, 41
rnd = random.Random(SEED)
maze = np.ones((H, W), dtype=int)          # 1 = wall, 0 = free


def carve(r, c):
    maze[r, c] = 0
    dirs = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    rnd.shuffle(dirs)
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 < nr < H - 1 and 0 < nc < W - 1 and maze[nr, nc] == 1:
            maze[r + dr // 2, c + dc // 2] = 0
            carve(nr, nc)


carve(1, 1)
# knock out ~10% of inner walls so that the maze has loops (several routes)
walls = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1)
         if maze[r, c] == 1 and ((r % 2 == 1) != (c % 2 == 1))]
for r, c in rnd.sample(walls, len(walls) // 10):
    maze[r, c] = 0
START, GOAL = (1, 1), (H - 2, W - 2)


def neighbours(cell):
    r, c = cell
    for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and maze[nr, nc] == 0:   # bounds before walls
            yield (nr, nc)


def search(start, goal, mode):
    """BFS and DFS differ only in which end of the frontier is popped."""
    frontier = deque([start])
    parent = {start: None}                 # marked when enqueued
    expanded, frontier_size = [], []
    while frontier:
        node = frontier.popleft() if mode == "bfs" else frontier.pop()   # the one line
        expanded.append(node)
        if node == goal:
            break
        for nxt in neighbours(node):
            if nxt not in parent:
                parent[nxt] = node
                frontier.append(nxt)
        frontier_size.append(len(frontier))
    path, cur = [], goal
    while cur is not None and goal in parent:
        path.append(cur); cur = parent[cur]
    return expanded, path[::-1], frontier_size


def validate(path):
    ok = path[0] == START and path[-1] == GOAL
    for a, b in zip(path, path[1:]):
        ok &= abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 and maze[b] == 0
    return bool(ok)


res = {}
for mode in ("bfs", "dfs"):
    exp_, path, fr = search(START, GOAL, mode)
    res[mode] = dict(expanded=len(exp_), path_steps=len(path) - 1, valid=validate(path),
                     max_frontier=max(fr), cells=exp_, path=path, frontier=fr)
    print(f"{mode.upper()}: {len(exp_)} cells expanded, path {len(path) - 1} steps, "
          f"max frontier {max(fr)}, valid = {validate(path)}")
open_cells = int((maze == 0).sum())
print(f"maze {H} x {W}, open cells {open_cells}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, mode, col in zip(axes, ("bfs", "dfs"), ("tab:blue", "tab:red")):
    d = res[mode]
    shade = np.full(maze.shape, np.nan)
    for k, (r, c) in enumerate(d["cells"]):
        shade[r, c] = k
    ax.imshow(maze, cmap="binary", extent=(-0.5, W - 0.5, H - 0.5, -0.5))
    ax.imshow(shade, cmap="YlGn", alpha=0.55, extent=(-0.5, W - 0.5, H - 0.5, -0.5))
    ax.plot([p[1] for p in d["path"]], [p[0] for p in d["path"]], color=col, lw=2,
            label=f"{mode.upper()} path ({d['path_steps']} steps)")
    ax.plot(START[1], START[0], "g^", ms=9, label="start"); ax.plot(GOAL[1], GOAL[0], "k*", ms=12, label="goal")
    ax.set_title(f"{mode.upper()}: {d['expanded']} cells expanded (shaded)")
    ax.set_xlabel("Column (cells)"); ax.set_ylabel("Row (cells)")
    ax.legend(fontsize=7, loc="lower left")
stamp(fig); fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig3_1_maze.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.4))
for mode, col in (("bfs", "tab:blue"), ("dfs", "tab:red")):
    ax.plot(res[mode]["frontier"], color=col, lw=1, label=mode.upper())
ax.set_xlabel("Expansion step"); ax.set_ylabel("Frontier size (cells)")
ax.set_title("Frontier size against expansion step"); ax.legend(); ax.grid(alpha=0.3)
stamp(fig); fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig3_2_frontier.png"), dpi=150); plt.close(fig)

with open(os.path.join(HERE, "results.json"), "w") as fh:
    json.dump(dict(H=H, W=W, open_cells=open_cells,
                   **{m: {k: v for k, v in d.items() if k in ("expanded", "path_steps", "valid", "max_frontier")}
                      for m, d in res.items()}), fh, indent=2)
