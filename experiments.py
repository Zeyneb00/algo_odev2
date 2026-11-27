"""
experiments.py  -  produces every table and image needed for the report

Run:   python experiments.py              (looks for datasets next to this file)
       python experiments.py some_folder  (look for the dataset files there)

Creates a folder  results/  with

  summary.csv          all datasets: n, algorithm, metric, k, cluster sizes, runtime
  metric_study.csv     Euclidean vs Manhattan vs Chebyshev on every dataset
  scaling_study.csv    no scaling vs min-max vs z-score on the group B datasets
  k_study.csv          the effect of k on four datasets
  runtime_study.csv    runtime and memory for n = 100, 250, 500, ... points
  runtime_fit.csv      measured growth exponent (runtime ~ n^p)
  images/              pictures for the report

Timing covers ONLY scaling + clustering (no file loading, no drawing),
as the assignment requires. Each time is the median of several repeats.
"""

import csv
import math
import os
import random
import statistics
import sys
import time
import tracemalloc

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

import plotting
from clustering_algos import cluster_sizes, run_clustering
from clustering_core import METRICS, SCALERS, load_manifest, load_points

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "results")
IMG_DIR = os.path.join(OUT_DIR, "images")

# ----------------------------------------------------------------------
# SETTINGS  (edit these, then rerun)
# ----------------------------------------------------------------------
REPEATS = 5          # timing repeats; the median is reported
SEED = 42            # for the fixed random subsets in the runtime study

# k used in the main table. These are the number of groups the datasets are
# known to contain (from their sources). The labels themselves are NEVER given
# to the algorithm - only the count is used to choose k.
# Facebook has no natural grouping, so k=3 is an arbitrary choice.
REFERENCE_K = {
    "aggregation.csv": 7, "compound.csv": 6, "pathbased.csv": 3,
    "spiral.csv": 3, "jain.csv": 2, "flame.csv": 2, "r15.csv": 15,
    "skewed.csv": 6,                       # check against the UEF page
    "iris.csv": 3, "seeds_dataset.csv": 3, "raisin_dataset.csv": 2,
    "wholesale_customers_data.csv": 2, "dataset_facebook.csv": 3,
}
DEFAULT_K = 3

# Group A: both axes use the same units  -> no scaling.
# Group B: features have very different ranges -> z-score.
MAIN_METRIC = "euclidean"
MAIN_SCALING = {"A": "none", "B": "zscore"}

K_STUDY = {                                # file -> list of k
    "aggregation.csv": [3, 5, 7, 9],
    "compound.csv": [3, 6, 9, 12],
    "spiral.csv": [2, 3, 4, 6],
    "iris.csv": [2, 3, 4, 6],
}
RUNTIME_FILES = ["skewed.csv", "raisin_dataset.csv"]
RUNTIME_SIZES = [100, 250, 500, 750, 1000]
RUNTIME_K = 3


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def find_file(root, filename):
    target = filename.lower()
    for folder, _dirs, files in os.walk(root):
        if "raw" in os.path.basename(folder).lower() or \
                os.path.basename(folder) == "results":
            continue
        for f in files:
            if f.lower() == target:
                return os.path.join(folder, f)
    return None


def write_csv(name, header, rows):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"   wrote results/{name}  ({len(rows)} rows)")


def timed_run(points, k, algo, metric, scaling, repeats=REPEATS):
    """Median runtime in ms (scaling + clustering only), and the last result."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        labels, edges = run_clustering(points, k, algo, metric, scaling)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), labels, edges


def peak_memory_mb(points, k, algo, metric, scaling):
    tracemalloc.start()
    run_clustering(points, k, algo, metric, scaling)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def spacing(scaled_points, labels, metric):
    """Smallest distance between two points of different clusters
    (the quantity greedy clustering maximises)."""
    dist = METRICS[metric]
    best = float("inf")
    n = len(scaled_points)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j]:
                d = dist(scaled_points[i], scaled_points[j])
                if d < best:
                    best = d
    return best


def _pairs(x):
    return x * (x - 1) // 2


def rand_index(a, b):
    """Fraction of point pairs on which two clusterings agree
    (both together or both apart). 1.0 = identical partitions."""
    cont, ra, rb = {}, {}, {}
    for x, y in zip(a, b):
        cont[(x, y)] = cont.get((x, y), 0) + 1
        ra[x] = ra.get(x, 0) + 1
        rb[y] = rb.get(y, 0) + 1
    total = _pairs(len(a))
    return (total + 2 * sum(_pairs(v) for v in cont.values())
            - sum(_pairs(v) for v in ra.values())
            - sum(_pairs(v) for v in rb.values())) / total


def sizes_text(labels):
    return " ".join(str(s) for s in sorted(cluster_sizes(labels), reverse=True))


def describe(labels):
    sizes = cluster_sizes(labels)
    n = len(labels)
    return min(sizes), max(sizes) / n


def stem(filename):
    return os.path.splitext(filename)[0]


def save_fig(fig, name):
    fig.savefig(os.path.join(IMG_DIR, name))
    print(f"   wrote results/images/{name}")


def three_views(name, points, labels, edges, xl, yl, title):
    fig = Figure(figsize=(13, 4.3), dpi=110, layout="constrained")
    ax = fig.subplots(1, 3)
    plotting.draw_original(ax[0], points, xl, yl)
    plotting.draw_clusters(ax[1], points, labels, xl, yl)
    plotting.draw_graph(ax[2], points, labels, edges, True, xl, yl)
    fig.suptitle(title, fontsize=11)
    save_fig(fig, name)


# ----------------------------------------------------------------------
# load everything once
# ----------------------------------------------------------------------
def load_datasets(root):
    data = {}
    for row in load_manifest(os.path.join(BASE_DIR, "manifest.csv")):
        key = row["file"].lower()
        path = find_file(root, row["file"])
        if path is None:
            print(f"   SKIP {row['file']} (not found under {root})")
            continue
        points, skipped = load_points(path, row["col_x"], row["col_y"])
        data[key] = {"file": row["file"], "name": stem(row["file"]),
                     "group": row["group"], "x": row["col_x"],
                     "y": row["col_y"], "points": points, "skipped": skipped}
    return data


def main_k(key):
    return REFERENCE_K.get(key, DEFAULT_K)


# ----------------------------------------------------------------------
# study 1: main table
# ----------------------------------------------------------------------
def study_summary(data):
    print("1. Main table (every dataset, both algorithms)")
    rows = []
    for key, d in data.items():
        pts, n = d["points"], len(d["points"])
        k = min(main_k(key), n)
        scaling = MAIN_SCALING[d["group"]]
        ref = None
        for algo in ("kruskal", "prim"):
            ms, labels, edges = timed_run(pts, k, algo, MAIN_METRIC, scaling)
            smallest, _ = describe(labels)
            if algo == "kruskal":
                ref = labels
                same = ""
                three_views(f"dataset_{d['name']}.png", pts, labels, edges,
                            d["x"], d["y"],
                            f"{d['name']}  -  Kruskal, {MAIN_METRIC}, "
                            f"scaling={scaling}, k={k}, n={n}")
            else:
                same = "yes" if rand_index(ref, labels) == 1.0 else "no (ties)"
            rows.append([d["name"], d["group"], n, d["skipped"], d["x"], d["y"],
                         algo, MAIN_METRIC, scaling, k, sizes_text(labels),
                         smallest, len(edges), f"{ms:.2f}", same])
    write_csv("summary.csv",
              ["dataset", "group", "n", "rows_skipped", "x_feature", "y_feature",
               "algorithm", "metric", "scaling", "k", "cluster_sizes",
               "smallest_cluster", "edges_kept", "runtime_ms",
               "same_clusters_as_kruskal"], rows)


# ----------------------------------------------------------------------
# study 2: distance metrics
# ----------------------------------------------------------------------
def study_metrics(data):
    print("2. Distance metric comparison (Kruskal)")
    rows = []
    for key, d in data.items():
        pts, n = d["points"], len(d["points"])
        k = min(main_k(key), n)
        scaling = MAIN_SCALING[d["group"]]
        base = None
        for metric in METRICS:
            ms, labels, _ = timed_run(pts, k, "kruskal", metric, scaling, 1)
            if metric == "euclidean":
                base = labels
            smallest, largest_share = describe(labels)
            rows.append([d["name"], metric, scaling, k, sizes_text(labels),
                         smallest, f"{largest_share:.3f}",
                         f"{rand_index(base, labels):.4f}"])
    write_csv("metric_study.csv",
              ["dataset", "metric", "scaling", "k", "cluster_sizes",
               "smallest_cluster", "largest_cluster_share",
               "rand_index_vs_euclidean"], rows)


# ----------------------------------------------------------------------
# study 3: scaling
# ----------------------------------------------------------------------
def study_scaling(data):
    print("3. Feature scaling comparison (group B, Euclidean, Kruskal)")
    rows = []
    for key, d in data.items():
        if d["group"] != "B":
            continue
        pts, n = d["points"], len(d["points"])
        k = min(main_k(key), n)
        fig = Figure(figsize=(13, 4.3), dpi=110, layout="constrained")
        axes = fig.subplots(1, len(SCALERS))
        base = None
        for ax, scaling in zip(axes, SCALERS):
            _ms, labels, _ = timed_run(pts, k, "kruskal", "euclidean", scaling, 1)
            if scaling == "none":
                base = labels
            smallest, largest_share = describe(labels)
            rows.append([d["name"], scaling, k, sizes_text(labels), smallest,
                         f"{largest_share:.3f}",
                         f"{rand_index(base, labels):.4f}"])
            plotting.draw_clusters(ax, pts, labels, d["x"], d["y"])
            ax.set_title(f"scaling = {scaling}  (k = {k})", fontsize=10,
                         fontweight="bold")
        fig.suptitle(f"{d['name']}: effect of feature scaling", fontsize=11)
        save_fig(fig, f"scaling_{d['name']}.png")
    write_csv("scaling_study.csv",
              ["dataset", "scaling", "k", "cluster_sizes", "smallest_cluster",
               "largest_cluster_share", "rand_index_vs_none"], rows)


# ----------------------------------------------------------------------
# study 4: effect of k
# ----------------------------------------------------------------------
def study_k(data):
    print("4. Effect of k")
    rows = []
    for key, ks in K_STUDY.items():
        d = data.get(key)
        if d is None:
            continue
        pts = d["points"]
        scaling = MAIN_SCALING[d["group"]]
        scaled = SCALERS[scaling](pts)
        fig = Figure(figsize=(14, 3.8), dpi=110, layout="constrained")
        axes = fig.subplots(1, len(ks))
        for ax, k in zip(axes, ks):
            ms, labels, _ = timed_run(pts, k, "kruskal", MAIN_METRIC, scaling, 3)
            smallest, _ = describe(labels)
            rows.append([d["name"], k, sizes_text(labels), smallest,
                         f"{spacing(scaled, labels, MAIN_METRIC):.4f}",
                         f"{ms:.2f}"])
            plotting.draw_clusters(ax, pts, labels, d["x"], d["y"])
        fig.suptitle(f"{d['name']}: effect of k (Kruskal, {MAIN_METRIC}, "
                     f"scaling={scaling})", fontsize=11)
        save_fig(fig, f"k_study_{d['name']}.png")
    write_csv("k_study.csv",
              ["dataset", "k", "cluster_sizes", "smallest_cluster",
               "spacing_between_clusters", "runtime_ms"], rows)


# ----------------------------------------------------------------------
# study 5: runtime and memory vs number of points
# ----------------------------------------------------------------------
def fit_exponent(ns, ts):
    """Least-squares slope of log(time) against log(n)."""
    xs = [math.log(n) for n in ns]
    ys = [math.log(t) for t in ts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            / sum((x - mx) ** 2 for x in xs))


def study_runtime(data):
    print("5. Runtime and memory vs number of points")
    rows, series, fits = [], {}, []
    for key in RUNTIME_FILES:
        d = data.get(key)
        if d is None:
            continue
        pts_all = d["points"]
        scaling = MAIN_SCALING[d["group"]]
        for algo in ("kruskal", "prim"):
            ns, ts, ms_ = [], [], []
            for n in RUNTIME_SIZES:
                if n > len(pts_all):
                    continue
                # fixed random subset: sample n of the rows with a fixed seed
                idx = sorted(random.Random(SEED + n).sample(range(len(pts_all)), n))
                pts = [pts_all[i] for i in idx]
                ms, _l, edges = timed_run(pts, RUNTIME_K, algo, MAIN_METRIC, scaling, 3)
                mem = peak_memory_mb(pts, RUNTIME_K, algo, MAIN_METRIC, scaling)
                all_edges = n * (n - 1) // 2
                rows.append([d["name"], algo, n, all_edges, f"{ms:.2f}",
                             f"{mem:.2f}"])
                ns.append(n); ts.append(ms); ms_.append(mem)
            if len(ns) >= 3:
                fits.append([d["name"], algo, f"{fit_exponent(ns, ts):.2f}",
                             f"{fit_exponent(ns, ms_):.2f}"])
            series[(d["name"], algo)] = (ns, ts, ms_)
    write_csv("runtime_study.csv",
              ["dataset", "algorithm", "n", "possible_edges", "runtime_ms",
               "peak_memory_mb"], rows)
    write_csv("runtime_fit.csv",
              ["dataset", "algorithm", "time_exponent_p (time ~ n^p)",
               "memory_exponent_q (memory ~ n^q)"], fits)

    fig = Figure(figsize=(12, 4.3), dpi=110, layout="constrained")
    a1, a2 = fig.subplots(1, 2)
    style = {"kruskal": "o-", "prim": "s--"}
    for (name, algo), (ns, ts, mems) in series.items():
        a1.plot(ns, ts, style[algo], label=f"{name} - {algo}")
        a2.plot(ns, mems, style[algo], label=f"{name} - {algo}")
    for ax, ylabel, title in ((a1, "runtime (ms)", "Runtime vs number of points"),
                              (a2, "peak memory (MB)", "Memory vs number of points")):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("n (points)"); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    save_fig(fig, "runtime_memory.png")


# ----------------------------------------------------------------------
def main():
    root = sys.argv[1] if len(sys.argv) > 1 else BASE_DIR
    os.makedirs(IMG_DIR, exist_ok=True)
    print(f"Searching for datasets under {root}")
    data = load_datasets(root)
    if not data:
        print("No datasets found. Check manifest.csv and the folder.")
        sys.exit(1)
    print(f"{len(data)} datasets loaded\n")
    study_summary(data)
    study_metrics(data)
    study_scaling(data)
    study_k(data)
    study_runtime(data)
    print("\nDone. Everything is in the 'results' folder.")


if __name__ == "__main__":
    main()
