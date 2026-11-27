"""
app.py  -  Greedy Clustering Studio (desktop application)

Run with:   python app.py

Needs:  Python 3.9+, matplotlib   (pip install matplotlib)
Tkinter ships with the standard Windows/macOS Python installer.

Files of this project
    clustering_core.py   distances, scaling, union-find
    clustering_algos.py  Kruskal and Prim clustering
    data_io.py           CSV / TXT reading
    plotting.py          the three views
    manifest.csv         prescribed X / Y columns of the group B datasets
"""

import os
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

import data_io
import plotting
from clustering_core import METRICS, SCALERS
from clustering_algos import run_clustering, cluster_sizes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.csv")
MAX_TABLE_ROWS = 5000          # the table shows at most this many rows


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Greedy Clustering Studio")
        self.geometry("1320x820")
        self.minsize(1100, 700)
        try:
            ttk.Style(self).theme_use("clam")
        except tk.TclError:
            pass

        # ---- data state -------------------------------------------------
        self.path = None
        self.header = []
        self.rows = []
        self.decimal_comma = False
        self.points = []            # (x, y) pairs actually used
        self.skipped = 0            # rows dropped (blank / non-numeric)
        self.labels = None          # clustering result
        self.edges = None

        # ---- tk variables ----------------------------------------------
        self.file_var = tk.StringVar(value="(no file loaded)")
        self.x_var = tk.StringVar()
        self.y_var = tk.StringVar()
        self.algo_var = tk.StringVar(value="kruskal")
        self.metric_var = tk.StringVar(value="euclidean")
        self.scale_var = tk.StringVar(value="none")
        self.k_var = tk.StringVar(value="3")
        self.edges_var = tk.BooleanVar(value=True)
        self.info_var = tk.StringVar(value="")
        self.stat_vars = {name: tk.StringVar(value="-") for name in
                          ("Points", "Clusters", "Selected edges", "Runtime")}

        self._build_ui()
        self._show_empty_plots()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ==================================================================
    # UI construction
    # ==================================================================
    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # ---------------- left: control panel ---------------------------
        panel = ttk.Frame(self, padding=12)
        panel.grid(row=0, column=0, sticky="ns")

        ttk.Label(panel, text="CONTROL PANEL",
                  font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        ttk.Separator(panel).pack(fill="x", pady=6)

        ttk.Label(panel, text="Dataset").pack(anchor="w")
        ttk.Button(panel, text="Open CSV / TXT...",
                   command=self.open_file).pack(fill="x", pady=(2, 2))
        ttk.Label(panel, textvariable=self.file_var, wraplength=210,
                  foreground="#555555").pack(anchor="w", pady=(0, 8))

        ttk.Label(panel, text="X feature").pack(anchor="w")
        self.x_box = ttk.Combobox(panel, textvariable=self.x_var,
                                  state="readonly", width=28)
        self.x_box.pack(fill="x", pady=(2, 6))
        ttk.Label(panel, text="Y feature").pack(anchor="w")
        self.y_box = ttk.Combobox(panel, textvariable=self.y_var,
                                  state="readonly", width=28)
        self.y_box.pack(fill="x", pady=(2, 8))
        self.x_box.bind("<<ComboboxSelected>>", self._on_columns_changed)
        self.y_box.bind("<<ComboboxSelected>>", self._on_columns_changed)

        ttk.Label(panel, text="Algorithm").pack(anchor="w")
        row = ttk.Frame(panel)
        row.pack(fill="x", pady=(2, 8))
        ttk.Radiobutton(row, text="Kruskal", value="kruskal",
                        variable=self.algo_var).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(row, text="Prim", value="prim",
                        variable=self.algo_var).pack(side="left")

        ttk.Label(panel, text="Distance metric").pack(anchor="w")
        ttk.Combobox(panel, textvariable=self.metric_var, state="readonly",
                     values=list(METRICS)).pack(fill="x", pady=(2, 8))

        ttk.Label(panel, text="Feature scaling").pack(anchor="w")
        ttk.Combobox(panel, textvariable=self.scale_var, state="readonly",
                     values=list(SCALERS)).pack(fill="x", pady=(2, 8))

        ttk.Label(panel, text="Number of clusters (k)").pack(anchor="w")
        self.k_spin = ttk.Spinbox(panel, from_=1, to=1000,
                                  textvariable=self.k_var, width=8)
        self.k_spin.pack(anchor="w", pady=(2, 10))

        ttk.Button(panel, text="Run clustering",
                   command=self.run).pack(fill="x", ipady=4)
        ttk.Checkbutton(panel, text="Show graph edges",
                        variable=self.edges_var,
                        command=self._redraw_graph).pack(anchor="w", pady=10)
        ttk.Label(panel, textvariable=self.info_var, wraplength=210,
                  foreground="#555555").pack(anchor="w")

        # ---------------- right: plots + summary + tables ---------------
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=8)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=3)
        right.rowconfigure(2, weight=2)

        plot_frame = ttk.Frame(right)
        plot_frame.grid(row=0, column=0, sticky="nsew")
        self.fig = Figure(figsize=(12, 4), dpi=100, layout="constrained")
        self.axes = self.fig.subplots(1, 3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # summary cards
        summary = ttk.Frame(right, padding=(0, 6))
        summary.grid(row=1, column=0, sticky="ew")
        for i, (name, var) in enumerate(self.stat_vars.items()):
            summary.columnconfigure(i, weight=1)
            card = ttk.Frame(summary, relief="groove", padding=6)
            card.grid(row=0, column=i, sticky="ew", padx=3)
            ttk.Label(card, text=name.upper(), foreground="#666666",
                      font=("TkDefaultFont", 8, "bold")).pack(anchor="w")
            ttk.Label(card, textvariable=var,
                      font=("TkDefaultFont", 15, "bold")).pack(anchor="w")

        # tabs: data table | cluster sizes
        notebook = ttk.Notebook(right)
        notebook.grid(row=2, column=0, sticky="nsew")

        table_tab = ttk.Frame(notebook)
        notebook.add(table_tab, text="Data table")
        self.table = ttk.Treeview(table_tab, show="headings")
        vsb = ttk.Scrollbar(table_tab, orient="vertical",
                            command=self.table.yview)
        hsb = ttk.Scrollbar(table_tab, orient="horizontal",
                            command=self.table.xview)
        self.table.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.table.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        table_tab.rowconfigure(0, weight=1)
        table_tab.columnconfigure(0, weight=1)

        sizes_tab = ttk.Frame(notebook)
        notebook.add(sizes_tab, text="Cluster sizes")
        self.sizes = ttk.Treeview(sizes_tab, show="headings",
                                  columns=("cluster", "points", "share"))
        for col, text, width in (("cluster", "Cluster", 100),
                                 ("points", "Points", 100),
                                 ("share", "Share", 100)):
            self.sizes.heading(col, text=text)
            self.sizes.column(col, width=width, anchor="center")
        svsb = ttk.Scrollbar(sizes_tab, orient="vertical",
                             command=self.sizes.yview)
        self.sizes.configure(yscrollcommand=svsb.set)
        self.sizes.grid(row=0, column=0, sticky="nsew")
        svsb.grid(row=0, column=1, sticky="ns")
        sizes_tab.rowconfigure(0, weight=1)
        sizes_tab.columnconfigure(0, weight=1)

    # ==================================================================
    # loading data
    # ==================================================================
    def open_file(self):
        start = os.path.join(BASE_DIR, "datasets")
        if not os.path.isdir(start):
            start = BASE_DIR
        path = filedialog.askopenfilename(
            title="Open dataset", initialdir=start,
            filetypes=[("CSV / TXT files", "*.csv *.txt"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            header, rows, decimal_comma = data_io.read_table(path)
        except Exception as exc:                       # noqa: BLE001
            messagebox.showerror("Could not read file", str(exc))
            return

        numeric = data_io.numeric_columns(header, rows, decimal_comma)
        if len(numeric) < 2:
            messagebox.showerror(
                "Not enough numeric columns",
                "The file needs at least two numeric columns.")
            return

        self.path = path
        self.header, self.rows, self.decimal_comma = header, rows, decimal_comma
        self.file_var.set(os.path.basename(path))
        self.x_box.configure(values=numeric)
        self.y_box.configure(values=numeric)

        # prescribed columns from manifest.csv, otherwise the first two
        chosen = data_io.manifest_columns(MANIFEST_PATH, path)
        if chosen and chosen[0] in numeric and chosen[1] in numeric:
            self.x_var.set(chosen[0])
            self.y_var.set(chosen[1])
        else:
            self.x_var.set(numeric[0])
            self.y_var.set(numeric[1])

        self._fill_table()
        self._update_points()
        self._clear_result()
        self._redraw_all()

    def _fill_table(self):
        self.table.delete(*self.table.get_children())
        self.table.configure(columns=self.header)
        for name in self.header:
            self.table.heading(name, text=name)
            self.table.column(name, width=max(80, 9 * len(name)),
                              stretch=False, anchor="e")
        for r in self.rows[:MAX_TABLE_ROWS]:
            self.table.insert("", "end", values=r)

    def _update_points(self):
        self.points, self.skipped = data_io.extract_points(
            self.header, self.rows, self.x_var.get(), self.y_var.get(),
            self.decimal_comma)
        n = len(self.points)
        self.stat_vars["Points"].set(str(n))
        self.k_spin.configure(to=max(n, 1))
        try:
            if int(self.k_var.get()) > n:
                self.k_var.set(str(max(n, 1)))
        except ValueError:
            self.k_var.set("3")
        note = f"{len(self.rows)} rows in file, {n} points used."
        if self.skipped:
            note += f" {self.skipped} rows skipped (blank value in X or Y)."
        if len(self.rows) > MAX_TABLE_ROWS:
            note += f" Table shows the first {MAX_TABLE_ROWS} rows."
        self.info_var.set(note)

    def _on_columns_changed(self, _event=None):
        if not self.header:
            return
        self._update_points()
        self._clear_result()
        self._redraw_all()

    # ==================================================================
    # clustering
    # ==================================================================
    def run(self):
        n = len(self.points)
        if n == 0:
            messagebox.showinfo("No data", "Open a dataset first.")
            return
        try:
            k = int(self.k_var.get())
        except ValueError:
            messagebox.showerror("Invalid k", "k must be a whole number.")
            return
        if not 1 <= k <= n:
            messagebox.showerror("Invalid k",
                                 f"k must be between 1 and {n}.")
            return

        self.configure(cursor="watch")
        self.update_idletasks()
        try:
            t0 = time.perf_counter()
            labels, edges = run_clustering(
                self.points, k, self.algo_var.get(),
                self.metric_var.get(), self.scale_var.get())
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        finally:
            self.configure(cursor="")

        self.labels, self.edges = labels, edges
        self.stat_vars["Clusters"].set(str(k))
        self.stat_vars["Selected edges"].set(str(len(edges)))
        self.stat_vars["Runtime"].set(f"{elapsed_ms:.1f} ms")
        self._fill_sizes()
        self._redraw_all()

    def _clear_result(self):
        self.labels = self.edges = None
        for name in ("Clusters", "Selected edges", "Runtime"):
            self.stat_vars[name].set("-")
        self.sizes.delete(*self.sizes.get_children())

    def _fill_sizes(self):
        self.sizes.delete(*self.sizes.get_children())
        sizes = cluster_sizes(self.labels)
        total = sum(sizes)
        for c, s in enumerate(sizes):
            self.sizes.insert("", "end",
                              values=(c + 1, s, f"{100.0 * s / total:.1f} %"))

    # ==================================================================
    # drawing
    # ==================================================================
    def _labels(self):
        return self.x_var.get() or "x", self.y_var.get() or "y"

    def _show_empty_plots(self):
        plotting.draw_empty(self.axes[0], "Original data",
                            "Open a dataset to begin")
        plotting.draw_empty(self.axes[1], "Clustering result")
        plotting.draw_empty(self.axes[2], "Graph view")
        self.canvas.draw_idle()

    def _redraw_all(self):
        xl, yl = self._labels()
        plotting.draw_original(self.axes[0], self.points, xl, yl)
        if self.labels is None:
            plotting.draw_empty(self.axes[1], "Clustering result")
            plotting.draw_empty(self.axes[2], "Graph view")
        else:
            plotting.draw_clusters(self.axes[1], self.points, self.labels,
                                   xl, yl)
            plotting.draw_graph(self.axes[2], self.points, self.labels,
                                self.edges, self.edges_var.get(), xl, yl)
        self.canvas.draw_idle()

    def _redraw_graph(self):
        """Only the graph panel changes when the checkbox is toggled."""
        if self.labels is None:
            return
        xl, yl = self._labels()
        plotting.draw_graph(self.axes[2], self.points, self.labels,
                            self.edges, self.edges_var.get(), xl, yl)
        self.canvas.draw_idle()

    def _on_close(self):
        self.quit()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
