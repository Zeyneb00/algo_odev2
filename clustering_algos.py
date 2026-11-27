"""
clustering_algos.py  -  Part 2: the two greedy clustering algorithms

Both algorithms return the same kind of result:
    labels : list, labels[i] = cluster number (0..k-1) of point i
    edges  : list of (i, j, weight) = the edges the algorithm KEPT
             (these are the lines drawn in the "graph view")

Idea (single-linkage / maximum-spacing clustering):
    The minimum spanning tree (MST) connects all points with the
    cheapest set of edges. If we remove the k-1 heaviest MST edges,
    the tree falls apart into exactly k pieces = k clusters.

    Kruskal does this by STOPPING EARLY (when k pieces remain).
    Prim builds the whole MST first and then CUTS k-1 edges.
"""

from clustering_core import METRICS, SCALERS, UnionFind


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _check_inputs(points, k):
    n = len(points)
    if n == 0:
        raise ValueError("no points")
    if not (1 <= k <= n):
        raise ValueError(f"k must satisfy 1 <= k <= n (k={k}, n={n})")


def _labels_from_uf(uf, n):
    """Turn union-find groups into labels 0..k-1.
    Numbering follows the first appearance by point index, so results
    are stable and easy to compare."""
    label_of_root = {}
    labels = []
    for i in range(n):
        r = uf.find(i)
        if r not in label_of_root:
            label_of_root[r] = len(label_of_root)
        labels.append(label_of_root[r])
    return labels


def cluster_sizes(labels):
    """Number of points per cluster, as a list indexed by cluster number."""
    sizes = [0] * (max(labels) + 1)
    for c in labels:
        sizes[c] += 1
    return sizes


# ----------------------------------------------------------------------
# Kruskal with early stopping
# ----------------------------------------------------------------------
def kruskal_cluster(points, k, metric="euclidean"):
    """
    1. Compute all n(n-1)/2 pairwise distances (the edges).
    2. Sort edges from smallest to largest.
    3. Go through them in order. Take an edge if it joins two different
       groups (union-find says no cycle). Skip it otherwise.
    4. STOP as soon as exactly k groups remain (after n-k accepted edges).
    Time O(n^2 log n), memory O(n^2).
    """
    _check_inputs(points, k)
    dist = METRICS[metric]
    n = len(points)

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((dist(points[i], points[j]), i, j))
    edges.sort()                      # by weight, ties broken by (i, j)

    uf = UnionFind(n)
    kept = []
    for w, i, j in edges:
        if uf.components == k:        # early stop
            break
        if uf.union(i, j):            # greedy choice: cheapest edge, no cycle
            kept.append((i, j, w))

    return _labels_from_uf(uf, n), kept


# ----------------------------------------------------------------------
# Prim (dense, O(n^2)) + cut the k-1 heaviest MST edges
# ----------------------------------------------------------------------
def prim_cluster(points, k, metric="euclidean"):
    """
    1. Start the tree at point 0.
    2. Repeat n-1 times: among points not yet in the tree, pick the one
       closest to the tree, and add that edge. (greedy choice)
       best_dist[v] / best_from[v] remember v's cheapest link to the tree.
    3. The result is the MST (n-1 edges). Remove the k-1 heaviest edges.
    4. The remaining n-k edges connect the clusters.
    Time O(n^2). Memory O(n): no edge list is ever stored.
    """
    _check_inputs(points, k)
    dist = METRICS[metric]
    n = len(points)

    in_tree = [False] * n
    best_dist = [float("inf")] * n
    best_from = [-1] * n

    mst = []                          # (i, j, weight)
    current = 0
    in_tree[0] = True
    for _ in range(n - 1):
        # update every outside point's cheapest link to the tree
        for v in range(n):
            if not in_tree[v]:
                d = dist(points[current], points[v])
                if d < best_dist[v]:
                    best_dist[v] = d
                    best_from[v] = current
        # pick the outside point with the smallest link
        nxt = -1
        for v in range(n):
            if not in_tree[v] and (nxt == -1 or best_dist[v] < best_dist[nxt]):
                nxt = v
        mst.append((best_from[nxt], nxt, best_dist[nxt]))
        in_tree[nxt] = True
        current = nxt

    # cut the k-1 heaviest edges, keep the other n-k
    mst.sort(key=lambda e: e[2])      # ascending by weight
    kept = mst[: n - k]

    uf = UnionFind(n)
    for i, j, w in kept:
        uf.union(i, j)
    return _labels_from_uf(uf, n), kept


ALGORITHMS = {
    "kruskal": kruskal_cluster,
    "prim": prim_cluster,
}


def run_clustering(points, k, algorithm="kruskal", metric="euclidean",
                   scaling="none"):
    """Scale the points, then cluster. Returns (labels, kept_edges)."""
    scaled = SCALERS[scaling](points)
    return ALGORITHMS[algorithm](scaled, k, metric)
