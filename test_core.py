from clustering_core import *

# distances: points (0,0) and (3,4)
a, b = (0, 0), (3, 4)
assert euclidean(a, b) == 5.0
assert manhattan(a, b) == 7
assert chebyshev(a, b) == 4

# scaling
pts = [(0, 10), (5, 20), (10, 30)]
assert scale_minmax(pts) == [(0, 0), (0.5, 0.5), (1, 1)]
z = scale_zscore(pts)
assert abs(sum(p[0] for p in z)) < 1e-9
assert scale_minmax([(1, 1), (1, 1)]) == [(0, 0), (0, 0)]  # constant column

# union-find
uf = UnionFind(5)
assert uf.components == 5
assert uf.union(0, 1) is True
assert uf.union(1, 2) is True
assert uf.union(0, 2) is False          # cycle
assert uf.find(0) == uf.find(2)
assert uf.find(3) != uf.find(0)
assert uf.components == 3

# loading with a blank cell
open("t.csv", "w").write("p,q,r\n1,2,3\n4,,6\n7,8,9\n")
p, s = load_points("t.csv", "p", "q")
assert p == [(1.0, 2.0), (7.0, 8.0)] and s == 1
import os; os.remove("t.csv")
print("all tests passed")
