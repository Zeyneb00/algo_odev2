"""
clustering_core.py  -  Part 1: building blocks (no clustering yet)

Contents
  1. Distance metrics      (Euclidean, Manhattan, Chebyshev)
  2. Feature scaling        (none, min-max, z-score)
  3. Union-Find             (path compression + union by rank)
  4. Data loading           (read two chosen columns from a CSV)

Everything is written by hand. Only the standard library is used.
A "point" is a tuple (x, y).
"""

import csv
import math


# ----------------------------------------------------------------------
# 1. Distance metrics
# ----------------------------------------------------------------------
def euclidean(a, b):
    """Straight-line distance: sqrt(dx^2 + dy^2)."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def manhattan(a, b):
    """City-block distance: |dx| + |dy|."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def chebyshev(a, b):
    """Largest single-axis difference: max(|dx|, |dy|)."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


METRICS = {
    "euclidean": euclidean,
    "manhattan": manhattan,
    "chebyshev": chebyshev,
}


# ----------------------------------------------------------------------
# 2. Feature scaling
# ----------------------------------------------------------------------
def scale_none(points):
    return list(points)


def scale_minmax(points):
    """Map each column to [0, 1]:  (v - min) / (max - min)."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_range = (x_max - x_min) or 1.0   # constant column -> avoid divide by 0
    y_range = (y_max - y_min) or 1.0
    return [((x - x_min) / x_range, (y - y_min) / y_range) for x, y in points]


def scale_zscore(points):
    """Mean 0, standard deviation 1 per column:  (v - mean) / std."""
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sx = math.sqrt(sum((p[0] - mx) ** 2 for p in points) / n) or 1.0
    sy = math.sqrt(sum((p[1] - my) ** 2 for p in points) / n) or 1.0
    return [((x - mx) / sx, (y - my) / sy) for x, y in points]


SCALERS = {
    "none": scale_none,
    "minmax": scale_minmax,
    "zscore": scale_zscore,
}


# ----------------------------------------------------------------------
# 3. Union-Find (disjoint set)
# ----------------------------------------------------------------------
class UnionFind:
    """
    Tracks which points belong to the same group.

    find(i)      -> the "root" (representative) of i's group
    union(i, j)  -> merge the two groups; returns False if they were
                    already the same group (i.e. the edge would form a cycle)
    components   -> current number of groups
    """

    def __init__(self, n):
        self.parent = list(range(n))   # each point starts as its own root
        self.rank = [0] * n            # upper bound on tree height
        self.components = n

    def find(self, i):
        # find the root
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        # path compression: point everything on the path straight to the root
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return False
        # union by rank: attach the shorter tree under the taller one
        if self.rank[ri] < self.rank[rj]:
            ri, rj = rj, ri
        self.parent[rj] = ri
        if self.rank[ri] == self.rank[rj]:
            self.rank[ri] += 1
        self.components -= 1
        return True


# ----------------------------------------------------------------------
# 4. Data loading
# ----------------------------------------------------------------------
def load_points(path, col_x, col_y):
    """
    Read two named columns from a CSV file.
    Rows where either value is empty or not a number are skipped.
    Returns (points, skipped_count).
    """
    points = []
    skipped = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                points.append((float(row[col_x]), float(row[col_y])))
            except (ValueError, TypeError, KeyError):
                skipped += 1
    return points, skipped


def load_manifest(path):
    """Read manifest.csv -> list of dicts with file, group, col_x, col_y."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def column_names(path):
    """Header row of a CSV (used later by the GUI column picker)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        return next(csv.reader(f))
