import os, tempfile
from data_io import *

def make(text, name="t.txt"):
    p = os.path.join(tempfile.mkdtemp(), name)
    open(p, "w", encoding="utf-8").write(text)
    return p

# comma + header + blank cell
p = make("a,b,c\n1,2,x\n3,,y\n5,6,z\n")
h, rows, dc = read_table(p)
assert h == ["a", "b", "c"] and len(rows) == 3 and not dc
assert numeric_columns(h, rows, dc) == ["a", "b"]
pts, sk = extract_points(h, rows, "a", "b", dc)
assert pts == [(1.0, 2.0), (5.0, 6.0)] and sk == 1

# whitespace, no header (UEF style)
p = make("15.55 28.65\n14.9 27.55\n")
h, rows, dc = read_table(p)
assert h == ["col_0", "col_1"] and len(rows) == 2

# tab, no header, 3 columns with label
p = make("1.5\t2.5\t1\n3.5\t4.5\t2\n")
h, rows, dc = read_table(p)
assert len(h) == 3 and extract_points(h, rows, "col_0", "col_1")[0][1] == (3.5, 4.5)

# semicolon with decimal commas (Turkish Excel)
p = make("x;y\n1,5;2,5\n3,25;4\n")
h, rows, dc = read_table(p)
assert dc and extract_points(h, rows, "x", "y", dc)[0] == [(1.5, 2.5), (3.25, 4.0)]

# BOM
p = os.path.join(tempfile.mkdtemp(), "b.csv")
open(p, "wb").write(b"\xef\xbb\xbfx,y\n1,2\n")
assert read_table(p)[0] == ["x", "y"]

# manifest lookup
m = make("file,group,col_x,col_y\nIris.csv,B,pl,pw\n", "m.csv")
assert manifest_columns(m, "C:/foo/iris.csv") == ("pl", "pw")
assert manifest_columns(m, "other.csv") is None
print("io tests passed")
