#!/usr/bin/env python3
"""
FRET Monte Carlo Simulator - desktop application (Tkinter + Matplotlib)
Engine: exifret.py (ExiFRET-style simulation, Corry 2005 / Deplazes 2012)

Run:  python exifret_gui.py
"""
from __future__ import annotations

import csv
import dataclasses
import itertools
import json
import math
import os
import queue
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Circle

import exifret
from exifret import (ACCEPTOR, DONOR, EMPTY, FretParams, assign_types, generate_coordinates, run_simulation)

APP_TITLE = "FRET Monte Carlo Simulator"
NA = 6.02214076e23

# --------------------------------------------------------------------------- #
# parameter description: (section, field, label, kind, help)
# kind: "int", "float", "bool", "choice:2,3", "str", "seed", "density"
# --------------------------------------------------------------------------- #
SPEC = [
    ("Geometry", "dimension", "System dimension", "choice:2,3", "Fluorophores arranged in 2-D (membrane/surface) or 3-D (solution)."),
    ("Geometry", "nmer_size", "n-mer size (n)", "int", "n=1: single random fluorophores; n=2: linked pairs; n>=3: ring / sphere / hexagonal oligomer."),
    ("Geometry", "radius", "Radius (Å)", "float", "Radius of the circle/sphere carrying the fluorophores (half the separation for pairs; nearest-neighbour spacing for hexagonal packing)."),
    ("Geometry", "second_radius", "Use separate host radius", "bool", "Use a different radius (below) for excluded volume than for fluorophore positions."),
    ("Geometry", "host_radius", "Host radius (Å)", "float", "Radius of the host used to avoid overlap when 'separate host radius' is ticked."),
    ("Geometry", "hexagonal_oligomers", "Hexagonal oligomers (2-D)", "bool", "Hexagonally packed n-mers instead of a ring (2-D only)."),
    ("Geometry", "planar_nmers", "Planar n-mers (3-D)", "bool", "3-D only: n-mers on a circle instead of a sphere."),
    ("Geometry", "particles", "Number of n-mers", "int", "Particles per configuration (hundreds is usually enough)."),
    ("Geometry", "density", "Density", "density", "n-mer density. Unit selectable on the right."),
    ("Geometry", "avoid_overlap", "Avoid overlap (excluded volume)", "bool", "No two n-mers closer than twice the host radius."),
    ("Geometry", "clusters", "Clustered distribution (2-D)", "bool", "Arrange n-mers in circular clusters instead of homogeneously."),
    ("Geometry", "n_clusters", "Number of clusters", "int", ""),
    ("Geometry", "cluster_radius", "Cluster radius (Å)", "float", ""),
    ("Geometry", "periodic", "Periodic boundaries", "bool", "Minimum-image convention at the box edges (recommended)."),
    ("Labelling", "p_donor", "Donor probability (Pdonor)", "float", "Fraction of labelled sites that are donors (0-1)."),
    ("Labelling", "labeling_efficiency", "Labelling efficiency (0-1)", "float", "Fraction of sites that carry a fluorophore; the rest are empty."),
    ("Labelling", "da_stoichiometry", "Fixed D/A per n-mer", "bool", "Enforce the donors/acceptors-per-n-mer given below."),
    ("Labelling", "donors_per_nmer", "Donors per n-mer", "int", ""),
    ("Labelling", "acceptors_per_nmer", "Acceptors per n-mer", "int", ""),
    ("Labelling", "da_separate", "Donor-only / acceptor-only n-mers", "bool", "Each n-mer holds only donors or only acceptors."),
    ("Labelling", "independent_types", "Independent Bernoulli assignment", "bool", "Each site is a donor with probability Pdonor (Corry 2005). Off = exact system-wide ratio (Deplazes 2012)."),
    ("Labelling", "zero_if_no_donor", "No donor in config -> E = 0", "bool", "Needed for single-molecule averages (e.g. isolated pentamers)."),
    ("Photophysics", "R0", "Förster distance R0 (Å)", "float", "Includes the orientation factor κ²."),
    ("Photophysics", "donor_lifetime", "Donor lifetime (ns)", "float", "Without acceptor."),
    ("Photophysics", "acceptor_lifetime", "Acceptor lifetime (ns)", "float", "Time an acceptor stays excited after accepting energy."),
    ("Photophysics", "irradiance", "Irradiance (W/µm²)", "float", "~0 (e.g. 1e-15) = no excitation competition."),
    ("Photophysics", "wavelength", "Laser wavelength (nm)", "float", ""),
    ("Photophysics", "extinction_coefficient", "Donor ε (cm⁻¹ M⁻¹)", "float", ""),
    ("Simulation", "configurations", "Configurations", "int", "Random arrangements averaged (≥1000 for publication quality)."),
    ("Simulation", "photons", "Excitons per configuration", "int", "Excitation events played per configuration."),
    ("Simulation", "seed", "Random seed (blank = random)", "seed", ""),
    ("Simulation", "user_coordinates", "Coordinate file (optional)", "str", "Text file 'x y z D|A' per line. If set, geometry is taken from the file and a single configuration is computed."),
]

DENSITY_UNITS = {
    2: [("×10⁻³ n-mers/Å²", 1e-3), ("n-mers/Å²", 1.0)],
    3: [("mM n-mers", 1e-3 * NA / 1e27), ("n-mers/Å³", 1.0)],
}

PRESETS = {
    "Fluorophores in 3-D solution (R0 = 60 Å, 4 mM)": dict(dimension=3, nmer_size=1, avoid_overlap=False, particles=800, R0=60,
        p_donor=0.5, irradiance=1e-15, configurations=25, photons=4000, _density=(4.0, "mM n-mers")),
    "Fluorophores in a plane (R0 = 30 Å)": dict(dimension=2, nmer_size=1, avoid_overlap=False, particles=800, R0=30,
        p_donor=0.5, irradiance=1e-15, configurations=30, photons=4000, _density=(0.5, "×10⁻³ n-mers/Å²")),
    "Linked D–A pairs in 3-D (sep 50 Å, R0 = 50 Å)": dict(dimension=3, nmer_size=2, radius=25, da_stoichiometry=True, particles=300,
        R0=50, irradiance=1e-15, configurations=30, photons=3000, _density=(1.0, "mM n-mers")),
    "D–A hosts in 2-D (Deplazes Fig 3b)": dict(dimension=2, nmer_size=2, radius=18, da_stoichiometry=True, particles=300, R0=30,
        irradiance=1e-15, configurations=40, photons=3000, _density=(0.3, "×10⁻³ n-mers/Å²")),
    "Pentamers in a plane (Corry Fig 12)": dict(dimension=2, nmer_size=5, radius=30, particles=300, R0=50, p_donor=0.5,
        irradiance=1e-15, configurations=30, photons=3000, _density=(0.02, "×10⁻³ n-mers/Å²")),
    "Tetramers, labelling efficiency 60 %": dict(dimension=2, nmer_size=4, radius=10, particles=300, R0=30, labeling_efficiency=0.6,
        irradiance=1e-15, configurations=30, photons=3000, _density=(0.05, "×10⁻³ n-mers/Å²")),
    "Clustered fluorophores in 2-D": dict(dimension=2, nmer_size=1, avoid_overlap=False, particles=300, R0=50, clusters=True,
        n_clusters=5, cluster_radius=100, irradiance=1e-15, configurations=30, photons=3000, _density=(0.2, "×10⁻³ n-mers/Å²")),
}

SWEEPABLE = {
    "Density": "density", "R0 (Å)": "R0", "Radius (Å)": "radius", "Donor probability": "p_donor",
    "Labelling efficiency": "labeling_efficiency", "n-mer size": "nmer_size", "Irradiance (W/µm²)": "irradiance",
    "Acceptor lifetime (ns)": "acceptor_lifetime", "Donor lifetime (ns)": "donor_lifetime", "Host radius (Å)": "host_radius",
}
INT_FIELDS = {"nmer_size"}


class Tooltip:
    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        if text:
            widget.bind("<Enter>", self.show)
            widget.bind("<Leave>", self.hide)

    def show(self, _e=None):
        if self.tip:
            return
        x, y = self.widget.winfo_rootx() + 20, self.widget.winfo_rooty() + 22
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background="#ffffe0", relief="solid", borderwidth=1,
                 wraplength=320, padx=4, pady=2).pack()

    def hide(self, _e=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x800")
        self.minsize(1000, 650)
        ttk.Style(self).theme_use("clam")
        self.vars: dict[str, tk.Variable] = {}
        self.widgets: dict[str, list] = {}
        self.q: queue.Queue = queue.Queue()
        self.cancel = threading.Event()
        self.worker: threading.Thread | None = None
        self.last_result = None
        self.last_params = None
        self.sweep_data = None
        self._build_menu()
        self._build_layout()
        self.load_defaults()
        self.after(100, self._poll)

    # ------------------------------------------------------------------ menu
    def _build_menu(self):
        m = tk.Menu(self)
        f = tk.Menu(m, tearoff=0)
        f.add_command(label="Load settings…", command=self.load_settings)
        f.add_command(label="Save settings…", command=self.save_settings)
        f.add_separator()
        f.add_command(label="Export last result (CSV)…", command=self.export_result)
        f.add_command(label="Export sweep (CSV)…", command=self.export_sweep)
        f.add_command(label="Save current figure…", command=self.save_figure)
        f.add_separator()
        f.add_command(label="Quit", command=self.destroy)
        m.add_cascade(label="File", menu=f)
        e = tk.Menu(m, tearoff=0)
        for name in PRESETS:
            e.add_command(label=name, command=lambda n=name: self.apply_preset(n))
        m.add_cascade(label="Examples", menu=e)
        h = tk.Menu(m, tearoff=0)
        h.add_command(label="Self-test (isolated pentamer vs analytic)", command=self.self_test)
        h.add_command(label="About", command=self.about)
        m.add_cascade(label="Help", menu=h)
        self.config(menu=m)

    # ---------------------------------------------------------------- layout
    def _build_layout(self):
        pw = ttk.PanedWindow(self, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw, width=390)
        right = ttk.Frame(pw)
        pw.add(left, weight=0)
        pw.add(right, weight=1)

        # buttons
        bar = ttk.Frame(left)
        bar.pack(fill="x", padx=6, pady=6)
        self.btn_run = ttk.Button(bar, text="▶  Run simulation", command=self.run_simulation_clicked)
        self.btn_run.pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.btn_prev = ttk.Button(bar, text="Preview geometry", command=self.preview_clicked)
        self.btn_prev.pack(side="left", expand=True, fill="x", padx=3)
        self.btn_stop = ttk.Button(bar, text="■ Stop", command=self.cancel.set, state="disabled")
        self.btn_stop.pack(side="left", padx=(3, 0))

        # scrollable parameter area
        canvas = tk.Canvas(left, highlightthickness=0)
        sb = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))
        self._build_params(inner)

        # right: notebook
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill="both", expand=True)
        self.tab_geom = ttk.Frame(self.nb)
        self.tab_res = ttk.Frame(self.nb)
        self.tab_sweep = ttk.Frame(self.nb)
        self.nb.add(self.tab_geom, text="Geometry preview")
        self.nb.add(self.tab_res, text="Result")
        self.nb.add(self.tab_sweep, text="Parameter sweep")
        self.fig_geom, self.cv_geom = self._make_canvas(self.tab_geom)
        self._build_result_tab()
        self._build_sweep_tab()

        # status bar
        sbar = ttk.Frame(self)
        sbar.pack(fill="x", side="bottom")
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(sbar, textvariable=self.status).pack(side="left", padx=8, pady=3)
        self.progress = ttk.Progressbar(sbar, length=260, mode="determinate")
        self.progress.pack(side="right", padx=8, pady=3)

    def _make_canvas(self, parent):
        fig = Figure(figsize=(6, 5), dpi=100)
        cv = FigureCanvasTkAgg(fig, master=parent)
        cv.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(cv, parent).update()
        return fig, cv

    def _build_params(self, parent):
        section = None
        frame = None
        for sec, name, label, kind, hlp in SPEC:
            if sec != section:
                section = sec
                frame = ttk.LabelFrame(parent, text=sec)
                frame.pack(fill="x", padx=6, pady=4)
                frame.columnconfigure(1, weight=1)
                r = 0
            if kind == "bool":
                v = tk.BooleanVar()
                w = ttk.Checkbutton(frame, text=label, variable=v)
                w.grid(row=r, column=0, columnspan=3, sticky="w", padx=4, pady=1)
                Tooltip(w, hlp)
                self.widgets[name] = [w]
            else:
                lb = ttk.Label(frame, text=label)
                lb.grid(row=r, column=0, sticky="w", padx=4, pady=1)
                Tooltip(lb, hlp)
                v = tk.StringVar()
                if kind.startswith("choice:"):
                    w = ttk.Combobox(frame, textvariable=v, values=kind.split(":")[1].split(","), width=8, state="readonly")
                    w.bind("<<ComboboxSelected>>", lambda e: self.on_dimension_change())
                else:
                    w = ttk.Entry(frame, textvariable=v, width=14)
                w.grid(row=r, column=1, sticky="ew", padx=4, pady=1)
                self.widgets[name] = [lb, w]
                if kind == "density":
                    self.density_unit = tk.StringVar()
                    self.cb_unit = ttk.Combobox(frame, textvariable=self.density_unit, width=16, state="readonly")
                    self.cb_unit.grid(row=r, column=2, padx=4)
                if name == "user_coordinates":
                    b = ttk.Button(frame, text="…", width=3, command=self.browse_coords)
                    b.grid(row=r, column=2, padx=4)
            self.vars[name] = v
            r += 1

    def _build_result_tab(self):
        top = ttk.Frame(self.tab_res)
        top.pack(fill="x", padx=8, pady=6)
        self.big = tk.StringVar(value="E = –")
        ttk.Label(top, textvariable=self.big, font=("TkDefaultFont", 26, "bold")).pack(side="left")
        self.res_text = tk.Text(self.tab_res, height=9, wrap="word")
        self.res_text.pack(fill="x", padx=8)
        self.fig_res, self.cv_res = self._make_canvas(self.tab_res)

    def _build_sweep_tab(self):
        c = ttk.LabelFrame(self.tab_sweep, text="Sweep one parameter (all other settings from the left panel)")
        c.pack(fill="x", padx=8, pady=6)
        self.sw_param = tk.StringVar(value="Density")
        self.sw_from, self.sw_to, self.sw_n = tk.StringVar(value="0.05"), tk.StringVar(value="1.0"), tk.StringVar(value="8")
        self.sw_log = tk.BooleanVar(value=False)
        ttk.Label(c, text="Parameter").grid(row=0, column=0, padx=4, pady=3)
        ttk.Combobox(c, textvariable=self.sw_param, values=list(SWEEPABLE), state="readonly", width=20).grid(row=0, column=1)
        for i, (t, v) in enumerate((("from", self.sw_from), ("to", self.sw_to), ("points", self.sw_n))):
            ttk.Label(c, text=t).grid(row=0, column=2 + 2 * i, padx=(10, 2))
            ttk.Entry(c, textvariable=v, width=8).grid(row=0, column=3 + 2 * i)
        ttk.Checkbutton(c, text="log spacing", variable=self.sw_log).grid(row=0, column=8, padx=8)
        self.btn_sweep = ttk.Button(c, text="Run sweep", command=self.run_sweep_clicked)
        self.btn_sweep.grid(row=0, column=9, padx=8)
        ttk.Label(c, text="(Density sweeps use the unit selected in the left panel.)").grid(row=1, column=0, columnspan=10, sticky="w", padx=4)
        self.fig_sw, self.cv_sw = self._make_canvas(self.tab_sweep)

    # ------------------------------------------------------------- settings
    def load_defaults(self):
        self.set_params(FretParams())
        self.apply_preset(list(PRESETS)[0])

    def on_dimension_change(self, keep_unit=None):
        dim = int(self.vars["dimension"].get() or 2)
        units = [u for u, _ in DENSITY_UNITS[dim]]
        self.cb_unit["values"] = units
        if keep_unit in units:
            self.density_unit.set(keep_unit)
        elif self.density_unit.get() not in units:
            self.density_unit.set(units[0])
        self.update_enabled()

    def update_enabled(self, *_):
        pass  # all fields remain editable; irrelevant ones are simply ignored by the engine

    def density_factor(self, dim, unit):
        for u, f in DENSITY_UNITS[dim]:
            if u == unit:
                return f
        return DENSITY_UNITS[dim][0][1]

    def set_params(self, p: FretParams, density_display=None):
        for name, v in self.vars.items():
            if name == "density":
                continue
            val = getattr(p, name)
            if isinstance(v, tk.BooleanVar):
                v.set(bool(val))
            else:
                v.set("" if val is None else str(val))
        self.vars["dimension"].set(str(p.dimension))
        self.on_dimension_change()
        if density_display is not None:
            val, unit = density_display
            self.on_dimension_change(keep_unit=unit)
            self.density_unit.set(unit)
            self.vars["density"].set(str(val))
        else:
            f = self.density_factor(p.dimension, self.density_unit.get())
            self.vars["density"].set(f"{p.density / f:g}")

    def apply_preset(self, name):
        d = dict(PRESETS[name])
        dens = d.pop("_density")
        p = dataclasses.replace(FretParams(), **d)
        self.set_params(p, density_display=dens)
        self.status.set(f"Loaded example: {name}")

    def get_params(self) -> FretParams:
        kw = {}
        types = {f.name: f.type for f in dataclasses.fields(FretParams)}
        for sec, name, label, kind, _ in SPEC:
            v = self.vars[name]
            try:
                if kind == "bool":
                    kw[name] = bool(v.get())
                elif kind.startswith("choice"):
                    kw[name] = int(v.get())
                elif kind == "int":
                    kw[name] = int(float(v.get()))
                elif kind in ("float", "density"):
                    kw[name] = float(v.get())
                elif kind == "seed":
                    kw[name] = int(v.get()) if v.get().strip() else None
                else:
                    kw[name] = v.get().strip() or None
            except ValueError:
                raise ValueError(f"'{label}' must be a valid number.")
        dim = kw["dimension"]
        kw["density"] = kw["density"] * self.density_factor(dim, self.density_unit.get())
        p = FretParams(**kw)
        self.validate(p)
        return p

    @staticmethod
    def validate(p: FretParams):
        if p.density <= 0: raise ValueError("Density must be > 0.")
        if p.particles < 1: raise ValueError("Number of n-mers must be ≥ 1.")
        if p.nmer_size < 1: raise ValueError("n-mer size must be ≥ 1.")
        if p.R0 <= 0: raise ValueError("R0 must be > 0.")
        if not (0 < p.p_donor < 1): raise ValueError("Donor probability must be between 0 and 1 (exclusive).")
        if not (0 < p.labeling_efficiency <= 1): raise ValueError("Labelling efficiency must be in (0, 1].")
        if p.configurations < 1 or p.photons < 1: raise ValueError("Configurations and excitons must be ≥ 1.")
        if p.donor_lifetime <= 0 or p.acceptor_lifetime <= 0: raise ValueError("Lifetimes must be > 0.")
        if p.irradiance <= 0 or p.wavelength <= 0 or p.extinction_coefficient <= 0:
            raise ValueError("Irradiance, wavelength and extinction coefficient must be > 0.")
        if p.da_stoichiometry and p.donors_per_nmer + p.acceptors_per_nmer > p.nmer_size:
            raise ValueError("Donors + acceptors per n-mer cannot exceed the n-mer size.")
        if p.clusters and p.dimension != 2: raise ValueError("Clusters are only available in 2-D.")
        if p.user_coordinates and not os.path.exists(p.user_coordinates): raise ValueError("Coordinate file not found.")


    @staticmethod
    def packing_warning(p: FretParams):
        """Excluded-volume packing fraction check (hard discs/spheres can't be packed beyond ~0.6 / ~0.4 by annealing)."""
        if not p.avoid_overlap or p.clusters or p.user_coordinates or p.particles < 2:
            return None
        R = p.host_radius if p.second_radius else p.radius
        vol = math.pi * R ** 2 if p.dimension == 2 else 4 / 3 * math.pi * R ** 3
        phi = p.density * vol
        limit = 0.60 if p.dimension == 2 else 0.40
        if phi > limit:
            dmax = limit / vol
            return (f"The excluded regions would fill {phi*100:.0f}% of the space (limit that can usually be reached: "
                    f"{limit*100:.0f}%). Reduce the density (to below ≈ {dmax:.3g} n-mers/Å{'²' if p.dimension == 2 else '³'}), "
                    f"reduce the radius, or untick 'Avoid overlap'.\n\nTry anyway?")
        return None

    def confirm_packing(self, p):
        w = self.packing_warning(p)
        return True if w is None else messagebox.askyesno("Density very high", w)

    def save_settings(self):
        try:
            p = self.get_params()
        except ValueError as e:
            return messagebox.showerror("Invalid input", str(e))
        fn = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if fn:
            d = dataclasses.asdict(p)
            with open(fn, "w") as f:
                json.dump(d, f, indent=2)

    def load_settings(self):
        fn = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not fn:
            return
        try:
            with open(fn) as f:
                p = FretParams(**json.load(f))
            self.set_params(p)
        except Exception as e:
            messagebox.showerror("Could not load settings", str(e))

    def browse_coords(self):
        fn = filedialog.askopenfilename(filetypes=[("Text", "*.txt *.dat *.csv"), ("All", "*.*")])
        if fn:
            self.vars["user_coordinates"].set(fn)

    # --------------------------------------------------------------- workers
    def _start(self, target, *args):
        if self.worker and self.worker.is_alive():
            return messagebox.showinfo(APP_TITLE, "A job is already running.")
        self.cancel.clear()
        self._busy(True)
        self.progress["value"] = 0

        def run():
            try:
                target(*args)
            except Exception as e:  # report in GUI thread
                self.q.put(("error", f"{type(e).__name__}: {e}"))

        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def _busy(self, busy):
        st = "disabled" if busy else "normal"
        for b in (self.btn_run, self.btn_prev, self.btn_sweep):
            b.config(state=st)
        self.btn_stop.config(state="normal" if busy else "disabled")

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, i, n, mean, label = msg
                    self.progress["value"] = 100 * i / max(n, 1)
                    self.status.set(f"{label}  {i}/{n}   running mean E = {mean:.4f}" if mean == mean else f"{label} {i}/{n}")
                elif kind == "status":
                    self.status.set(msg[1])
                elif kind == "geometry":
                    self._draw_geometry(*msg[1:])
                    self._busy(False)
                    self.status.set("Geometry preview ready.")
                elif kind == "result":
                    self._show_result(*msg[1:])
                    self._busy(False)
                elif kind == "sweep":
                    self._show_sweep(*msg[1:])
                    self._busy(False)
                elif kind == "message":
                    messagebox.showinfo(APP_TITLE, msg[1])
                    self._busy(False)
                elif kind == "error":
                    self._busy(False)
                    self.status.set("Error.")
                    messagebox.showerror("Simulation error", msg[1])
        except queue.Empty:
            pass
        self.after(100, self._poll)

    # ---- preview
    def preview_clicked(self):
        try:
            p = self.get_params()
        except ValueError as e:
            return messagebox.showerror("Invalid input", str(e))
        if not self.confirm_packing(p):
            return
        if p.user_coordinates:
            self.status.set("Loading coordinate file…")
        else:
            self.status.set("Generating geometry (annealing dense systems can take a few seconds)…")
        self._start(self._preview_worker, p)

    def _preview_worker(self, p):
        rng = np.random.default_rng(p.seed)
        if p.user_coordinates:
            coords, types = exifret.load_user_coordinates(p.user_coordinates)
            self.q.put(("geometry", coords, types, None, None, p))
            return
        coords, nmer_id, L = generate_coordinates(p, rng)
        types = assign_types(p, len(coords), nmer_id, rng)
        self.q.put(("geometry", coords, types, nmer_id, L, p))

    def _draw_geometry(self, coords, types, nmer_id, L, p):
        self.nb.select(self.tab_geom)
        fig = self.fig_geom
        fig.clear()
        three = (p.dimension == 3) if not p.user_coordinates else bool(np.ptp(coords[:, 2]) > 0)
        ax = fig.add_subplot(111, projection="3d" if three else None)
        colors = {DONOR: "#2e9e44", ACCEPTOR: "#d62728", EMPTY: "#999999"}
        names = {DONOR: "donor", ACCEPTOR: "acceptor", EMPTY: "empty site"}
        for t in (DONOR, ACCEPTOR, EMPTY):
            m = types == t
            if not m.any():
                continue
            if three:
                ax.scatter(coords[m, 0], coords[m, 1], coords[m, 2], c=colors[t], s=18, label=f"{names[t]} ({m.sum()})")
            else:
                ax.scatter(coords[m, 0], coords[m, 1], c=colors[t], s=22, edgecolors="k", linewidths=0.3,
                           label=f"{names[t]} ({m.sum()})", zorder=3)
        if (not three) and nmer_id is not None and p.avoid_overlap and p.particles <= 600:
            R = (p.host_radius if p.second_radius else p.radius)
            n = p.nmer_size
            centres = np.array([coords[nmer_id == k][:, :2].mean(axis=0) for k in range(p.particles)])
            ax.add_collection(PatchCollection([Circle(c, R) for c in centres], facecolor="none",
                                              edgecolor="#4477aa", linewidth=0.5, alpha=0.6, zorder=1))
        if not three:
            ax.set_aspect("equal")
            if L:
                ax.set_xlim(0, L); ax.set_ylim(0, L)
        ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
        if three:
            ax.set_zlabel("z (Å)")
        ttl = "User coordinates" if L is None else f"One configuration: {p.particles} n-mers (n={p.nmer_size}) in a {L:.0f} Å box, {p.dimension}-D"
        ax.set_title(ttl, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        self.cv_geom.draw()

    # ---- single run
    def run_simulation_clicked(self):
        try:
            p = self.get_params()
        except ValueError as e:
            return messagebox.showerror("Invalid input", str(e))
        if not self.confirm_packing(p):
            return
        self.status.set("Running… (first run compiles the fast kernel, ~10 s)")
        self._start(self._sim_worker, p)

    def _sim_worker(self, p):
        def prog(i, n, mean):
            self.q.put(("progress", i, n, mean, "Simulating"))
        res = run_simulation(p, progress=prog, cancel=self.cancel)
        self.q.put(("result", res, p))

    def _show_result(self, res, p):
        self.last_result, self.last_params = res, p
        self.nb.select(self.tab_res)
        E = res["efficiency"]
        self.big.set(f"E = {E:.4f}" + (f"  ± {res['sem']:.4f}" if res["configurations"] > 1 else ""))
        t = self.res_text
        t.delete("1.0", "end")
        lines = [
            f"FRET efficiency (mean over {res['configurations']} configuration(s)): {E:.5f}",
            f"Standard deviation between configurations: {res['std']:.5f}   standard error of mean: {res['sem']:.5f}",
            f"Donors / acceptors (last configuration): {res['n_donors']} / {res['n_acceptors']}",
            f"System: {p.dimension}-D, n={p.nmer_size}, {p.particles} n-mers, R0 = {p.R0} Å, labelling efficiency {p.labeling_efficiency}, Pdonor {p.p_donor}",
        ]
        if res.get("cancelled"):
            lines.append("** Run was stopped early - partial result. **")
        t.insert("end", "\n".join(lines))
        fig = self.fig_res
        fig.clear()
        ax = fig.add_subplot(111)
        vals = np.array(res["per_configuration"], float)
        if len(vals) > 1:
            ax.hist(vals[~np.isnan(vals)], bins=min(30, max(5, len(vals) // 3)), color="#4477aa", edgecolor="white")
            ax.axvline(E, color="k", ls="--", label=f"mean = {E:.3f}")
            ax.set_xlabel("FRET efficiency of one configuration")
            ax.set_ylabel("count")
            ax.legend()
            ax.set_title("Distribution over configurations", fontsize=10)
        else:
            ax.bar(["E"], [E], color="#4477aa"); ax.set_ylim(0, 1)
        fig.tight_layout()
        self.cv_res.draw()
        self.status.set("Done." if not res.get("cancelled") else "Stopped.")

    # ---- sweep
    def run_sweep_clicked(self):
        try:
            p = self.get_params()
            a, b, n = float(self.sw_from.get()), float(self.sw_to.get()), int(float(self.sw_n.get()))
            if n < 2 or n > 200:
                raise ValueError("Points must be between 2 and 200.")
            if self.sw_log.get() and (a <= 0 or b <= 0):
                raise ValueError("Log spacing needs positive limits.")
        except ValueError as e:
            return messagebox.showerror("Invalid input", str(e))
        label = self.sw_param.get()
        name = SWEEPABLE[label]
        vals = np.geomspace(a, b, n) if self.sw_log.get() else np.linspace(a, b, n)
        if name == "density":
            f = self.density_factor(p.dimension, self.density_unit.get())
            xs_display = vals
            vals = vals * f
        else:
            xs_display = vals
        if name in INT_FIELDS:
            vals = np.round(vals).astype(int)
            xs_display = vals
        self.status.set("Sweep running…")
        self._start(self._sweep_worker, p, name, label, list(vals), list(xs_display))

    def _sweep_worker(self, p, name, label, vals, shown):
        out = []
        n = len(vals)
        for i, (v, x) in enumerate(zip(vals, shown)):
            if self.cancel.is_set():
                break
            q = dataclasses.replace(p, **{name: (int(v) if name in INT_FIELDS else float(v))})
            if name == "density":
                q = dataclasses.replace(p, density=float(v))

            def prog(ci, cn, mean, i=i):
                self.q.put(("progress", i * 1.0 + ci / cn, n, mean, f"Sweep point {i+1}/{n}"))
            try:
                r = run_simulation(q, progress=prog, cancel=self.cancel)
                out.append((float(x), r["efficiency"], r["sem"]))
            except Exception as e:
                out.append((float(x), float("nan"), float("nan")))
                self.q.put(("status", f"Point {x:g}: {e}"))
        self.q.put(("sweep", out, label))

    def _show_sweep(self, data, label):
        self.sweep_data = (label, data)
        self.nb.select(self.tab_sweep)
        fig = self.fig_sw
        fig.clear()
        ax = fig.add_subplot(111)
        arr = np.array(data, float)
        if len(arr):
            ax.errorbar(arr[:, 0], arr[:, 1], yerr=arr[:, 2], fmt="o-", color="#4477aa", capsize=3)
        unit = f" [{self.density_unit.get()}]" if label == "Density" else ""
        ax.set_xlabel(label + unit)
        ax.set_ylabel("FRET efficiency")
        ax.set_ylim(0, 1.02)
        if self.sw_log.get():
            ax.set_xscale("log")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        self.cv_sw.draw()
        self.status.set("Sweep finished." if not self.cancel.is_set() else "Sweep stopped.")

    # ----------------------------------------------------------------- export
    def export_result(self):
        if not self.last_result:
            return messagebox.showinfo(APP_TITLE, "Run a simulation first.")
        fn = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not fn:
            return
        with open(fn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["# parameters"] + [f"{k}={v}" for k, v in dataclasses.asdict(self.last_params).items()])
            w.writerow(["mean_E", self.last_result["efficiency"], "std", self.last_result["std"], "sem", self.last_result["sem"]])
            w.writerow(["configuration", "E"])
            for i, e in enumerate(self.last_result["per_configuration"], 1):
                w.writerow([i, e])

    def export_sweep(self):
        if not self.sweep_data:
            return messagebox.showinfo(APP_TITLE, "Run a sweep first.")
        label, data = self.sweep_data
        fn = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if fn:
            with open(fn, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([label, "E_mean", "E_sem"])
                w.writerows(data)

    def save_figure(self):
        fig = {0: self.fig_geom, 1: self.fig_res, 2: self.fig_sw}[self.nb.index(self.nb.select())]
        fn = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if fn:
            fig.savefig(fn, dpi=300)

    # ------------------------------------------------------------------ help
    def self_test(self):
        self.status.set("Self-test running…")
        self._start(self._self_test_worker)

    def _self_test_worker(self):
        def analytic(r, R0, Pd):
            ang = 2 * np.pi * np.arange(5) / 5
            xy = r * np.c_[np.cos(ang), np.sin(ang)]
            dist = np.linalg.norm(xy[:, None] - xy[None], axis=2)
            tot = 0.0
            for conf in itertools.product([0, 1], repeat=5):
                conf = np.array(conf); nd = conf.sum()
                if nd in (0, 5):
                    continue
                D, A = np.where(conf == 1)[0], np.where(conf == 0)[0]
                s = np.array([(R0 / dist[d, A]) ** 6 for d in D]).sum(axis=1)
                tot += Pd ** nd * (1 - Pd) ** (5 - nd) * np.mean(s / (1 + s))
            return tot
        lines = []
        ok = True
        for r in (30, 60):
            p = FretParams(dimension=2, nmer_size=5, radius=r, particles=1, density=1e-9, avoid_overlap=False,
                           independent_types=True, R0=60, configurations=3000, photons=300, irradiance=1e-15, seed=5)
            res = run_simulation(p, cancel=self.cancel)
            ref = analytic(r, 60, 0.5)
            good = abs(res["efficiency"] - ref) < 0.02
            ok &= good
            lines.append(f"Pentamer radius {r} Å:  simulated {res['efficiency']:.3f}   analytic {ref:.3f}   {'OK' if good else 'FAIL'}")
        self.q.put(("message", ("SELF-TEST PASSED\n\n" if ok else "SELF-TEST FAILED\n\n") + "\n".join(lines)))

    def about(self):
        messagebox.showinfo("About", f"{APP_TITLE}\n\nMonte Carlo calculation of FRET efficiency for fluorophores in arbitrary 2-D/3-D "
            "arrangements (n-mers, clusters, labelling efficiency, user coordinates).\n\nMethod: Corry, Jayatilaka & Rigby, "
            "Biophys. J. 89:3822 (2005); Deplazes, Jayatilaka & Corry, J. Biomed. Opt. 17:011005 (2012).\n\n"
            "This is an independent re-implementation, validated against the analytic results in those papers "
            "(see validate.py), not the original ExiFRET program.")


if __name__ == "__main__":
    App().mainloop()
