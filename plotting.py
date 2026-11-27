"""
plotting.py  -  the three views required by the assignment (section 2.3)

  1. draw_original : the points, one colour (no cluster information)
  2. draw_clusters : same coordinates, coloured by cluster number
  3. draw_graph    : the edges the greedy algorithm selected (can be hidden)

All functions draw on a matplotlib Axes in the ORIGINAL coordinates,
even when the clustering used scaled values.
"""

from matplotlib import colormaps
from matplotlib.collections import LineCollection

POINT_COLOR = "#5b6b7c"
EDGE_COLOR = "#2b2b2b"


def _marker_size(n):
    return 16 if n <= 400 else 10 if n <= 800 else 7


def cluster_colors(labels):
    """One colour per label. Up to 20 clusters use a standard palette; for
    more clusters, hues are spread with the golden ratio so that
    neighbouring numbers still get clearly different colours."""
    k = max(labels) + 1
    if k <= 10:
        cmap = colormaps["tab10"]
        return [cmap(l) for l in labels]
    if k <= 20:
        cmap = colormaps["tab20"]
        return [cmap(l) for l in labels]
    cmap = colormaps["hsv"]
    return [cmap((l * 0.6180339887) % 1.0) for l in labels]


def _prepare(ax, title, xlabel, ylabel):
    ax.clear()
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)


def draw_original(ax, points, xlabel="x", ylabel="y"):
    _prepare(ax, "Original data", xlabel, ylabel)
    if not points:
        return
    ax.scatter([p[0] for p in points], [p[1] for p in points],
               s=_marker_size(len(points)), c=POINT_COLOR, linewidths=0)


def draw_clusters(ax, points, labels, xlabel="x", ylabel="y"):
    k = max(labels) + 1
    _prepare(ax, f"Clustering result (k = {k})", xlabel, ylabel)
    ax.scatter([p[0] for p in points], [p[1] for p in points],
               s=_marker_size(len(points)), c=cluster_colors(labels),
               linewidths=0)


def draw_graph(ax, points, labels, edges, show_edges=True,
               xlabel="x", ylabel="y"):
    title = f"Graph view ({len(edges)} selected edges)" if show_edges \
        else "Graph view (edges hidden)"
    _prepare(ax, title, xlabel, ylabel)
    if show_edges and edges:
        segments = [(points[i], points[j]) for i, j, _ in edges]
        ax.add_collection(LineCollection(segments, colors=EDGE_COLOR,
                                         linewidths=0.8, alpha=0.7, zorder=1))
    ax.scatter([p[0] for p in points], [p[1] for p in points],
               s=_marker_size(len(points)), c=cluster_colors(labels),
               linewidths=0, zorder=2)
    ax.autoscale_view()


def draw_empty(ax, title, message="Run the clustering to see this view"):
    ax.clear()
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, fontsize=9, color="#888888")
