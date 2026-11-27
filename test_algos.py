import random
from clustering_algos import *

def partition(labels):
    groups = {}
    for i, c in enumerate(labels):
        groups.setdefault(c, set()).add(i)
    return {frozenset(g) for g in groups.values()}

# 1. tiny hand-checked example: gaps are 1, 9, 1, 19
pts = [(0, 0), (1, 0), (10, 0), (11, 0), (30, 0)]
for name, algo in ALGORITHMS.items():
    labels, edges = algo(pts, 3)
    assert partition(labels) == {frozenset({0, 1}), frozenset({2, 3}), frozenset({4})}, name
    assert len(edges) == len(pts) - 3, name          # n - k edges kept
    assert cluster_sizes(labels) == [2, 2, 1], name

# 2. edge cases k=1 and k=n
for name, algo in ALGORITHMS.items():
    l, e = algo(pts, 1);  assert set(l) == {0} and len(e) == 4
    l, e = algo(pts, 5);  assert len(set(l)) == 5 and len(e) == 0

# 3. Kruskal and Prim agree on random data (no ties), all metrics
random.seed(1)
for metric in METRICS:
    for n in (30, 120):
        data = [(random.random() * 100, random.random() * 100) for _ in range(n)]
        for k in (1, 2, 3, 5, 10, n):
            lk, ek = kruskal_cluster(data, k, metric)
            lp, ep = prim_cluster(data, k, metric)
            assert partition(lk) == partition(lp), (metric, n, k)
            assert len(set(lk)) == k and len(ek) == n - k and len(ep) == n - k
            assert abs(sum(w for *_, w in ek) - sum(w for *_, w in ep)) < 1e-9

# 4. duplicates / ties: still exactly k non-empty clusters from both
dup = [(1, 1)] * 5 + [(5, 5)] * 5 + [(9, 9)] * 3
for algo in ALGORITHMS.values():
    l, e = algo(dup, 3)
    assert len(set(l)) == 3 and len(e) == len(dup) - 3

# 5. invalid k
for bad in (0, 6):
    try: kruskal_cluster(pts, bad); assert False
    except ValueError: pass

# 6. wrapper with scaling
l, e = run_clustering(pts, 3, "prim", "manhattan", "minmax")
assert len(set(l)) == 3
print("all algorithm tests passed")
