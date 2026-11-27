import csv, glob, os

print("Working folder:", os.getcwd())
paths = sorted(glob.glob("**/*.csv", recursive=True))
print("CSV files found:", len(paths))

for path in paths:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    bad = 0
    for r in data:
        for cell in r:
            try:
                float(cell)
            except ValueError:
                bad += 1
    print(f"{os.path.basename(path):30s} n={len(data):5d} cols={len(header):2d} bad_cells={bad}")