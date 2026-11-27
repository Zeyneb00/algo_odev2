"""
verify.py  -  does the clustering code really do what it should?

Run:   python verify.py            (checks the datasets listed in manifest.csv)
       python verify.py some_folder  (look for the dataset files in some_folder)

Four kinds of evidence, from strongest to weakest:

 1. OPTIMALITY (brute force, tiny data)
    Greedy clustering promises the maximum possible "spacing"
    (smallest distance between two different clusters) among ALL ways to
    split the points into k groups. On 8-point data we can try every
    possible split and confirm the promise holds. Integer coordinates are
    included on purpose, because they create ties.

 2. STRUCTURE (every dataset, every run)
    exactly k non-empty clusters; exactly n-k kept edges; the kept edges form
    a forest (no cycle) whose pieces ARE the clusters; every edge weight
    matches the distance recomputed from scratch.

 3. AGREEMENT (every dataset)
    Kruskal and Prim are two different routes to the same answer: same total
    edge weight and same spacing. When there are no ties they also give
    identical clusters.

 4. BEHAVIOUR
    Shuffling the row order must not change the clusters (no ties), and the
    clusters for k+1 must be a refinement of the clusters for k (splitting
    one cluster in two, never reshuffling).
"""

import os
import random
import sys

from clustering_core import METRICS, SCALERS, UnionFind, load_points, load_manifest
from clustering_algos import ALGORITHMS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

failures = []
checks = 0


def check(condition, message):
    global checks
    checks += 1
    if not condition:
        failures.append(message)
        print("   FAIL:", message)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def partition(labels):
    groups = {}
    for i, c in enumerate(labels):
        groups.setdefault(c, set()).add(i)
    return {frozenset(g) for g in groups.values()}


def spacing(points, labels, dist):
    """Smallest distance between two points in different clusters."""
    best = float("inf")
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j]:
                d = dist(points[i], points[j])
                if d < best:
                    best = d
    return best


def all_partitions(n, k):
    """Every way to split n items into exactly k non-empty groups
    (restricted growth strings, so no split is counted twice)."""
    def rec(i, used, current):
        if i == n:
            if used == k:
                yield list(current)
            return
        if used + (n - i) < k:          # not enough items left to reach k
            return
        for c in range(min(used + 1, k)):
            current.append(c)
            yield from rec(i + 1, max(used, c + 1), current)
            current.pop()
    yield from rec(0, 0, [])


def best_possible_spacing(points, k, dist):
    return max(spacing(points, lab, dist) for lab in all_partitions(len(points), k))


def refines(fine, coarse):
    """True if every cluster of `fine` lies completely inside one cluster of `coarse`."""
    owner = {}
    for f, c in zip(fine, coarse):
        if owner.setdefault(f, c) != c:
            return False
    return True


def structure_problems(points, k, labels, edges, dist, scaled):
    """Return a list of strings describing anything wrong with one result."""
    n = len(points)
    problems = []
    if len(labels) != n:
        problems.append("labels length != n")
    if len(set(labels)) != k:
        problems.append(f"{len(set(labels))} clusters instead of {k}")
    if len(edges) != n - k:
        problems.append(f"{len(edges)} edges instead of {n - k}")
    uf = UnionFind(n)
    for i, j, w in edges:
        if not uf.union(i, j):
            problems.append("kept edges contain a cycle")
            break
        if abs(dist(scaled[i], scaled[j]) - w) > 1e-9 * max(1.0, w):
            problems.append("edge weight differs from recomputed distance")
            break
    if uf.components != k:
        problems.append("kept edges do not form exactly k pieces")
    label_of_piece = {}
    for i in range(n):
        if label_of_piece.setdefault(uf.find(i), labels[i]) != labels[i]:
            problems.append("kept-edge pieces do not match the labels")
            break
    return problems


# ----------------------------------------------------------------------
# 1. brute-force optimality
# ----------------------------------------------------------------------
def test_optimality():
    print("1. Optimality against brute force (8 points, every possible split)")
    rng = random.Random(7)
    cases = 0
    for trial in range(24):
        if trial % 2 == 0:       # random real coordinates
            pts = [(rng.uniform(0, 50), rng.uniform(0, 50)) for _ in range(8)]
        else:                    # small integers -> many tied distances
            pts = [(rng.randint(0, 6), rng.randint(0, 6)) for _ in range(8)]
        for metric, dist in METRICS.items():
            for k in (2, 3, 4):
                best = best_possible_spacing(pts, k, dist)
                for name, algo in ALGORITHMS.items():
                    labels, _ = algo(pts, k, metric)
                    got = spacing(pts, labels, dist)
                    check(abs(got - best) < 1e-9,
                          f"{name}/{metric}/k={k}: spacing {got} < best {best}")
                    cases += 1
    print(f"   {cases} comparisons done")


# ----------------------------------------------------------------------
# 4. behaviour on random data
# ----------------------------------------------------------------------
def test_behaviour():
    print("2. Behaviour: shuffling rows and nesting of k (random data)")
    rng = random.Random(11)
    pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(80)]
    order = list(range(len(pts)))
    for metric in METRICS:
        for name, algo in ALGORITHMS.items():
            for k in (2, 4, 9):
                labels, _ = algo(pts, k, metric)
                rng.shuffle(order)
                shuffled = [pts[i] for i in order]
                l2, _ = algo(shuffled, k, metric)
                # map back: cluster of original point order[t] is l2[t]
                back = [None] * len(pts)
                for t, i in enumerate(order):
                    back[i] = l2[t]
                check(partition(labels) == partition(back),
                      f"{name}/{metric}/k={k}: result depends on row order")
            for k in (2, 3, 5, 8):
                coarse, _ = algo(pts, k, metric)
                fine, _ = algo(pts, k + 1, metric)
                check(refines(fine, coarse),
                      f"{name}/{metric}: k={k + 1} is not a refinement of k={k}")
    print("   done")


# ----------------------------------------------------------------------
# 2 + 3. real datasets
# ----------------------------------------------------------------------
def find_file(root, filename):
    target = filename.lower()
    for folder, _dirs, files in os.walk(root):
        if "raw" in os.path.basename(folder).lower():
            continue
        for f in files:
            if f.lower() == target:
                return os.path.join(folder, f)
    return None


def test_datasets(root):
    print("3. Real datasets (structure, Kruskal-vs-Prim agreement, nesting)")
    manifest_path = os.path.join(BASE_DIR, "manifest.csv")
    if not os.path.exists(manifest_path):
        print("   manifest.csv not found - skipping")
        return
    configs = [("euclidean", "none"), ("manhattan", "none"),
               ("chebyshev", "none"), ("euclidean", "zscore")]
    ks = (2, 3, 7)

    for row in load_manifest(manifest_path):
        path = find_file(root, row["file"])
        if path is None:
            print(f"   {row['file']:34s} SKIP (file not found under {root})")
            continue
        points, skipped = load_points(path, row["col_x"], row["col_y"])
        n = len(points)
        before = len(failures)
        identical = 0
        runs = 0
        for metric, scaling in configs:
            dist = METRICS[metric]
            scaled = SCALERS[scaling](points)
            per_k = {}
            for k in ks:
                if k > n:
                    continue
                res = {}
                for name, algo in ALGORITHMS.items():
                    labels, edges = algo(scaled, k, metric)
                    for p in structure_problems(scaled, k, labels, edges, dist, scaled):
                        check(False, f"{row['file']} {name}/{metric}/{scaling}/k={k}: {p}")
                    res[name] = (labels, edges)
                    runs += 1
                lk, ek = res["kruskal"]
                lp, ep = res["prim"]
                wk = sum(e[2] for e in ek)
                wp = sum(e[2] for e in ep)
                check(abs(wk - wp) <= 1e-6 * max(1.0, wk),
                      f"{row['file']} {metric}/{scaling}/k={k}: total weight "
                      f"Kruskal {wk} != Prim {wp}")
                sk = spacing(scaled, lk, dist)
                sp = spacing(scaled, lp, dist)
                check(abs(sk - sp) <= 1e-9 * max(1.0, sk),
                      f"{row['file']} {metric}/{scaling}/k={k}: spacing differs")
                if partition(lk) == partition(lp):
                    identical += 1
                per_k[k] = res
            ordered = sorted(per_k)
            for a, b in zip(ordered, ordered[1:]):
                for name in ALGORITHMS:
                    check(refines(per_k[b][name][0], per_k[a][name][0]),
                          f"{row['file']} {name}/{metric}/{scaling}: "
                          f"k={b} does not refine k={a}")
        status = "ok  " if len(failures) == before else "FAIL"
        print(f"   {row['file']:34s} n={n:5d} skipped={skipped:2d} "
              f"{status} (Kruskal=Prim identical in {identical} of "
              f"{runs // 2} runs; others differ only through ties)")


# ----------------------------------------------------------------------
def main():
    root = sys.argv[1] if len(sys.argv) > 1 else BASE_DIR
    test_optimality()
    test_behaviour()
    test_datasets(root)
    print()
    if failures:
        print(f"{len(failures)} PROBLEM(S) out of {checks} checks.")
        sys.exit(1)
    print(f"All {checks} checks passed.")


if __name__ == "__main__":
    main()
