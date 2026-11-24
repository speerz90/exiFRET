#!/usr/bin/env python3
"""
exifret.py - Monte Carlo FRET simulation (ExiFRET-style, Deplazes/Jayatilaka/Corry 2012)

Pipeline (matches the flow-chart):
  1. Generate fluorophore coordinates (2-D / 3-D, n-mers, clusters, or user file)
  2. Assign fluorophore types (donor / acceptor / empty)
  3. Calculate transfer probability (rate) for every donor-acceptor pair
  4. Generate exciton (photon) schedule from irradiance, wavelength, extinction coeff.
  5. Play excitons: pick donor -> available? -> relax by fluorescence or transfer
  6. FRET efficiency = N_transfer / (N_transfer + N_fluorescence), averaged over configurations

Usage
-----
  As a library:
      from exifret import FretParams, run_simulation
      res = run_simulation(FretParams(dimension=2, particles=200, density=1e-4, R0=50))
      print(res["efficiency"])

  From the command line:
      python exifret.py --dimension 2 --nmer 1 --particles 200 --density 1e-4 --R0 50
      python exifret.py --config my_input.json
      python exifret.py --user-coords coords.txt --R0 50       (single configuration)

  user-coords file: one fluorophore per line:  x  y  z  type     (type = D or A; '#' = comment)
Units: distances in Angstrom, lifetimes in ns, irradiance W/um^2, wavelength nm,
       extinction coefficient cm^-1 M^-1, density in n-mers/A^2 (2-D) or n-mers/A^3 (3-D).
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict, fields
from typing import Optional

import numpy as np

try:  # optional speed-up: pip install numba
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*a, **k):
        if len(a) == 1 and callable(a[0]):
            return a[0]
        return lambda f: f

# physical constants
H = 6.62607015e-34   # J s
C = 2.99792458e8     # m / s

DONOR, ACCEPTOR, EMPTY = 0, 1, 2


# --------------------------------------------------------------------------- #
# Input parameters (Table 1 of the paper)
# --------------------------------------------------------------------------- #
@dataclass
class FretParams:
    # geometry
    user_coordinates: Optional[str] = None  # path to coordinate file (else generated)
    dimension: int = 2                      # 2 or 3
    hexagonal_oligomers: bool = False       # 2-D only: hexagonally packed n-mers
    nmer_size: int = 1                      # n in n-mer
    radius: float = 20.0                    # radius of the fluorophore ring/sphere (A)
    second_radius: bool = False             # use a separate host radius for overlap
    host_radius: float = 20.0               # host radius used if second_radius=True (A)
    density: float = 1e-4                   # n-mers per A^2 (2-D) or A^3 (3-D)
    particles: int = 200                  # number of n-mers
    avoid_overlap: bool = True              # excluded volume around each n-mer
    planar_nmers: bool = False              # 3-D only: n-mers on a circle instead of sphere
    clusters: bool = False                  # 2-D only: clustered distribution
    n_clusters: int = 5
    cluster_radius: float = 100.0           # A
    # labelling / stoichiometry
    p_donor: float = 0.5                    # probability a fluorophore is a donor
    independent_types: bool = False         # True: each site is a donor with prob p_donor
                                            #   independently (Corry 2005). False: system-wide
                                            #   ratio enforced exactly (Deplazes 2012)
    zero_if_no_donor: bool = True           # configuration with no donors counts as E = 0
    labeling_efficiency: float = 1.0        # 0 < l <= 1
    da_stoichiometry: bool = False          # fixed donors/acceptors per n-mer
    donors_per_nmer: int = 1
    acceptors_per_nmer: int = 1
    da_separate: bool = False               # n-mers contain only donors OR only acceptors
    # photophysics
    R0: float = 50.0                        # Forster distance (A)
    donor_lifetime: float = 4.0             # ns
    acceptor_lifetime: float = 4.0          # ns
    irradiance: float = 1e-4                # W / um^2
    wavelength: float = 488.0               # nm
    extinction_coefficient: float = 8.0e4   # cm^-1 M^-1 (donor)
    # simulation
    configurations: int = 100               # paper uses >1000 for publication quality
    photons: int = 5000                     # excitation events per configuration
    periodic: bool = True                   # periodic boundary conditions (generated systems)
    seed: Optional[int] = None


# --------------------------------------------------------------------------- #
# Step 1: coordinates
# --------------------------------------------------------------------------- #
def _random_rotation(rng, dim):
    if dim == 2:
        a = rng.uniform(0, 2 * np.pi)
        R = np.eye(3)
        R[0, 0], R[0, 1], R[1, 0], R[1, 1] = np.cos(a), -np.sin(a), np.sin(a), np.cos(a)
        return R
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _hex_offsets(n, spacing):
    pts = []
    for q in range(-4, 5):
        for r in range(-4, 5):
            x = spacing * (q + r / 2.0)
            y = spacing * (math.sqrt(3) / 2.0 * r)
            pts.append((x, y, 0.0))
    pts = np.array(pts)
    pts = pts[np.argsort(np.linalg.norm(pts, axis=1), kind="stable")][:n]
    return pts - pts.mean(axis=0)


def nmer_offsets(p: FretParams):
    """Fluorophore positions of one n-mer relative to its centre (unrotated)."""
    n, r = p.nmer_size, p.radius
    if n == 1:
        return np.zeros((1, 3))
    if n == 2:  # pair separated by one diameter
        return np.array([[-r, 0, 0], [r, 0, 0]], float)
    if p.dimension == 2 and p.hexagonal_oligomers:
        return _hex_offsets(n, r)       # r = nearest-neighbour spacing
    if p.dimension == 2 or p.planar_nmers:   # circle
        ang = 2 * np.pi * np.arange(n) / n
        return np.c_[r * np.cos(ang), r * np.sin(ang), np.zeros(n)]
    # sphere (3-D): Fibonacci lattice
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return r * np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def _min_image(d, L, periodic, dim):
    if periodic:
        d = d - L * np.round(d / L)
    return d


@njit(cache=True)
def _anneal(centres, L, dim, min_d, steps, a0, seed):
    """Annealing placement (Corry 2005): move an overlapping n-mer to a random spot; keep the move if it
    does not increase its overlaps, or with probability a (a0 -> 0 over `steps`). Returns True when overlap-free."""
    np.random.seed(seed)
    P = centres.shape[0]
    md2 = min_d * min_d
    ov = np.zeros(P, np.int64)
    for i in range(P):
        for j in range(i + 1, P):
            d2 = 0.0
            for k in range(dim):
                d = centres[i, k] - centres[j, k]
                d -= L * np.round(d / L)
                d2 += d * d
            if d2 < md2:
                ov[i] += 1
                ov[j] += 1
    total = ov.sum()
    newpos = np.zeros(3)
    for step in range(steps):
        if total == 0:
            return True
        a = a0 * (1.0 - step / steps)
        i = np.random.randint(P)
        while ov[i] == 0:
            i = np.random.randint(P)
        if np.random.random() < 0.5:     # local move (needed at high packing) ...
            for k in range(dim):
                newpos[k] = (centres[i, k] + (np.random.random() - 0.5) * min_d) % L
        else:                            # ... or a global random re-placement
            for k in range(dim):
                newpos[k] = np.random.random() * L
        cnew = 0
        for j in range(P):
            if j == i:
                continue
            d2 = 0.0
            for k in range(dim):
                d = newpos[k] - centres[j, k]
                d -= L * np.round(d / L)
                d2 += d * d
            if d2 < md2:
                cnew += 1
        if cnew <= ov[i] or np.random.random() < a:
            for j in range(P):
                if j == i:
                    continue
                d2o = 0.0
                d2n = 0.0
                for k in range(dim):
                    d = centres[i, k] - centres[j, k]
                    d -= L * np.round(d / L)
                    d2o += d * d
                    d = newpos[k] - centres[j, k]
                    d -= L * np.round(d / L)
                    d2n += d * d
                if d2o < md2:
                    ov[j] -= 1
                if d2n < md2:
                    ov[j] += 1
            total += 2 * (cnew - ov[i])
            ov[i] = cnew
            for k in range(dim):
                centres[i, k] = newpos[k]
    return total == 0


def generate_coordinates(p: FretParams, rng):
    """Random sequential placement of n-mers with excluded volume (periodic box)."""
    dim = p.dimension
    P = p.particles
    L = (P / p.density) ** (1.0 / dim)
    R_host = p.host_radius if p.second_radius else p.radius
    if p.nmer_size == 2 and not p.second_radius:
        R_host = p.radius
    min_d = 2 * R_host if p.avoid_overlap else 0.0
    offs = nmer_offsets(p)

    # cluster centres (2-D)
    cl_centres = None
    if p.clusters:
        if dim != 2:
            raise ValueError("Clusters are only supported for 2-D systems")
        cl_centres = []
        for _ in range(p.n_clusters):
            for _try in range(20000):
                c = rng.uniform(0, L, 2)
                if all(np.linalg.norm(_min_image(c - o, L, True, 2)) >= 2 * p.cluster_radius
                       for o in cl_centres):
                    cl_centres.append(c)
                    break
            else:
                raise RuntimeError("Could not place clusters without overlap; reduce n_clusters/cluster_radius")
        cl_centres = np.array(cl_centres)

    centres = np.zeros((P, 3))
    vol_excl = (math.pi * (min_d / 2) ** 2) if dim == 2 else (4 / 3 * math.pi * (min_d / 2) ** 3)
    phi = P * vol_excl / L ** dim                       # packing fraction of the excluded regions
    if (not p.clusters) and min_d > 0 and phi > 0.25 and P > 1:   # dense -> annealing procedure
        centres[:, :dim] = rng.uniform(0, L, (P, dim))
        ok = _anneal(centres, L, dim, min_d, 4_000_000, 0.3, int(rng.integers(1 << 31)))
        if not ok:
            raise RuntimeError(f"Annealing could not remove all overlaps (packing fraction {phi:.2f}); "
                               "density too high for this radius.")
        P_range = range(0)
    else:
        P_range = range(P)
    for i in P_range:
        for _try in range(50000):
            if p.clusters:
                c = cl_centres[rng.integers(p.n_clusters)]
                rr = p.cluster_radius * math.sqrt(rng.uniform())
                a = rng.uniform(0, 2 * np.pi)
                cand = np.array([(c[0] + rr * math.cos(a)) % L, (c[1] + rr * math.sin(a)) % L, 0.0])
            else:
                cand = np.zeros(3)
                cand[:dim] = rng.uniform(0, L, dim)
            if i == 0 or min_d == 0:
                break
            d = _min_image(centres[:i] - cand, L, True, dim)
            if np.min(np.einsum("ij,ij->i", d, d)) >= min_d ** 2:
                break
        else:
            raise RuntimeError(
                f"Could not place n-mer {i+1}/{P} without overlap - density too high "
                f"(box={L:.0f} A, exclusion distance={min_d:.1f} A). Lower density/radius or set avoid_overlap=False.")
        centres[i] = cand

    coords = np.empty((P * len(offs), 3))
    nmer_id = np.repeat(np.arange(P), len(offs))
    for i in range(P):
        R = _random_rotation(rng, dim)
        coords[i * len(offs):(i + 1) * len(offs)] = centres[i] + offs @ R.T
    if dim == 2:
        coords[:, 2] = 0.0
    coords[:, :dim] %= L
    return coords, nmer_id, L


def load_user_coordinates(path):
    xyz, types = [], []
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            t = line.replace(",", " ").split()
            if len(t) < 4:
                raise ValueError(f"Bad line in coordinate file: '{line}' (need x y z type)")
            xyz.append([float(t[0]), float(t[1]), float(t[2])])
            tp = t[3].upper()
            types.append(DONOR if tp in ("D", "DONOR", "0") else ACCEPTOR if tp in ("A", "ACCEPTOR", "1") else EMPTY)
    return np.array(xyz), np.array(types, int)


# --------------------------------------------------------------------------- #
# Step 2: types
# --------------------------------------------------------------------------- #
def assign_types(p: FretParams, n_total, nmer_id, rng):
    types = np.full(n_total, ACCEPTOR, int)
    n = p.nmer_size
    P = p.particles
    if p.da_separate:
        n_don = int(round(p.p_donor * P))
        don_ids = rng.choice(P, n_don, replace=False)
        types[np.isin(nmer_id, don_ids)] = DONOR
    elif p.da_stoichiometry:
        if p.donors_per_nmer + p.acceptors_per_nmer > n:
            raise ValueError("donors_per_nmer + acceptors_per_nmer cannot exceed n-mer size")
        types[:] = EMPTY
        for k in range(P):
            idx = rng.permutation(np.where(nmer_id == k)[0])
            types[idx[:p.donors_per_nmer]] = DONOR
            types[idx[p.donors_per_nmer:p.donors_per_nmer + p.acceptors_per_nmer]] = ACCEPTOR
    elif p.independent_types:  # each site independently donor with prob p_donor
        types[:] = np.where(rng.random(n_total) < p.p_donor, DONOR, ACCEPTOR)
    else:  # system-wide ratio enforced
        n_don = int(round(p.p_donor * n_total))
        types[rng.choice(n_total, n_don, replace=False)] = DONOR
    # labelling efficiency -> random unlabelled ("empty") sites
    if p.labeling_efficiency < 1.0:
        n_empty = int(round(n_total * (1 - p.labeling_efficiency)))
        types[rng.choice(n_total, n_empty, replace=False)] = EMPTY
    return types


# --------------------------------------------------------------------------- #
# Step 3: transfer rates
# --------------------------------------------------------------------------- #
def transfer_rates(coords, types, p: FretParams, L):
    D = coords[types == DONOR]
    A = coords[types == ACCEPTOR]
    if len(D) == 0:
        return np.zeros((0, len(A)))
    if len(A) == 0:
        return np.zeros((len(D), 0))
    diff = D[:, None, :] - A[None, :, :]
    if L is not None and p.periodic:
        diff = diff - L * np.round(diff / L)          # minimum-image convention (all axes; z is 0 in 2-D)
    r = np.maximum(np.linalg.norm(diff, axis=2), 1e-3)
    return (1.0 / p.donor_lifetime) * (p.R0 / r) ** 6       # ns^-1


# --------------------------------------------------------------------------- #
# Step 4: exciton schedule
# --------------------------------------------------------------------------- #
def absorption_rate_per_donor(p: FretParams):
    """Photon absorption rate of one donor (ns^-1)."""
    E_ph = H * C / (p.wavelength * 1e-9)                    # J
    flux = p.irradiance / E_ph                              # photons s^-1 um^-2
    sigma_cm2 = 3.82e-21 * p.extinction_coefficient         # cm^2 per molecule
    sigma_um2 = sigma_cm2 * 1e8
    return flux * sigma_um2 * 1e-9


# --------------------------------------------------------------------------- #
# Step 5: play excitons
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _play_excitons(kT, kf, tau_acc, times, r_donor, r_out, r_time, r_acc):
    nD, nA = kT.shape
    donor_free = np.zeros(nD)
    acc_free = np.zeros(nA)
    n_fl = 0
    n_tr = 0
    n_lost = 0
    for i in range(times.shape[0]):
        t = times[i]
        d = int(r_donor[i] * nD)
        if d >= nD:
            d = nD - 1
        if donor_free[d] > t:           # target donor still excited -> next exciton
            n_lost += 1
            continue
        ksum = 0.0
        for a in range(nA):             # "find acceptor": only non-excited acceptors
            if acc_free[a] <= t:
                ksum += kT[d, a]
        ktot = kf + ksum
        donor_free[d] = t - math.log(r_time[i] + 1e-300) / ktot
        x = r_out[i] * ktot
        if x < kf:                      # fluorescence
            n_fl += 1
        else:                           # energy transfer
            x -= kf
            cum = 0.0
            chosen = -1
            for a in range(nA):
                if acc_free[a] <= t:
                    cum += kT[d, a]
                    chosen = a
                    if x < cum:
                        break
            acc_free[chosen] = t - tau_acc * math.log(r_acc[i] + 1e-300)   # Ta = -tau_A ln(g)
            n_tr += 1
    return n_fl, n_tr, n_lost


def simulate_configuration(coords, types, p: FretParams, L, rng):
    kT = np.ascontiguousarray(transfer_rates(coords, types, p, L))
    nD = kT.shape[0]
    if nD == 0:
        return (0.0 if p.zero_if_no_donor else np.nan), 0, 0, 0
    k_abs_total = nD * absorption_rate_per_donor(p)
    times = np.cumsum(rng.exponential(1.0 / k_abs_total, p.photons))
    n_fl, n_tr, n_lost = _play_excitons(
        kT, 1.0 / p.donor_lifetime, p.acceptor_lifetime, times,
        rng.random(p.photons), rng.random(p.photons), rng.random(p.photons), rng.random(p.photons))
    tot = n_fl + n_tr
    return (n_tr / tot if tot else np.nan), n_fl, n_tr, n_lost


# --------------------------------------------------------------------------- #
# Step 6: driver
# --------------------------------------------------------------------------- #
def run_simulation(p: FretParams, verbose=False, progress=None, cancel=None):
    """Returns dict with mean FRET efficiency, std, sem and per-configuration values.
    progress(i, n, running_mean) is called after every configuration; cancel is a threading.Event
    (set it to stop early - the partial result is returned with 'cancelled': True)."""
    if p.dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3")
    if not (0 < p.labeling_efficiency <= 1):
        raise ValueError("labeling_efficiency must be in (0, 1]")
    rng = np.random.default_rng(p.seed)

    # user-supplied coordinates -> single configuration
    if p.user_coordinates:
        coords, types = load_user_coordinates(p.user_coordinates)
        E, nf, nt, nl = simulate_configuration(coords, types, p, None, rng)
        return {"efficiency": E, "std": 0.0, "sem": 0.0, "per_configuration": [E],
                "n_donors": int((types == DONOR).sum()), "n_acceptors": int((types == ACCEPTOR).sum()),
                "configurations": 1}

    effs = []
    n_don = n_acc = 0
    cancelled = False
    for c in range(p.configurations):
        if cancel is not None and cancel.is_set():
            cancelled = True
            break
        coords, nmer_id, L = generate_coordinates(p, rng)          # step 1
        types = assign_types(p, len(coords), nmer_id, rng)         # step 2
        E, nf, nt, nl = simulate_configuration(coords, types, p, L, rng)  # steps 3-5
        effs.append(E)
        n_don, n_acc = int((types == DONOR).sum()), int((types == ACCEPTOR).sum())
        if progress is not None:
            progress(c + 1, p.configurations, float(np.nanmean(effs)))
        if verbose and ((c + 1) % max(1, p.configurations // 10) == 0):
            print(f"  config {c+1}/{p.configurations}: running mean E = {np.nanmean(effs):.4f}")
    if not effs:
        raise RuntimeError("Cancelled before the first configuration finished")
    effs = np.array(effs, float)
    return {"efficiency": float(np.nanmean(effs)),
            "std": float(np.nanstd(effs, ddof=1)) if len(effs) > 1 else 0.0,
            "sem": float(np.nanstd(effs, ddof=1) / np.sqrt(np.sum(~np.isnan(effs)))) if len(effs) > 1 else 0.0,
            "per_configuration": effs.tolist(),
            "n_donors": n_don, "n_acceptors": n_acc, "configurations": len(effs), "cancelled": cancelled}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser():
    ap = argparse.ArgumentParser(description="ExiFRET-style Monte Carlo FRET efficiency simulator",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", help="JSON file with any of the parameters below")
    ap.add_argument("--user-coords", dest="user_coordinates", help="file with x y z type per line")
    d = FretParams()
    for f in fields(FretParams):
        if f.name == "user_coordinates":
            continue
        flag = "--" + f.name.replace("_", "-")
        val = getattr(d, f.name)
        if isinstance(val, bool):
            ap.add_argument(flag, dest=f.name, type=lambda s: s.lower() in ("1", "true", "yes", "t"),
                            default=None, metavar="BOOL", help=f"(default {val})")
        elif f.name == "seed":
            ap.add_argument(flag, dest=f.name, type=int, default=None)
        else:
            ap.add_argument(flag, dest=f.name, type=type(val), default=None, help=f"(default {val})")
    ap.add_argument("--json-out", help="write full results to this JSON file")
    return ap


def main():
    ap = _build_parser()
    args = ap.parse_args()
    a = vars(args)
    json_out = a.pop("json_out")
    config = a.pop("config")
    cfg = {}
    if config:
        with open(config) as f:
            cfg.update(json.load(f))
    cfg.update({k: v for k, v in a.items() if v is not None})
    p = FretParams(**cfg)
    print("Parameters:", json.dumps(asdict(p), indent=2))
    res = run_simulation(p, verbose=True)
    print("\n=========== RESULT ===========")
    print(f"FRET efficiency : {res['efficiency']:.4f}  (std {res['std']:.4f}, s.e.m. {res['sem']:.4f})")
    print(f"Configurations  : {res['configurations']}")
    print(f"Donors/Acceptors: {res['n_donors']} / {res['n_acceptors']} (last configuration)")
    if json_out:
        with open(json_out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved -> {json_out}")


if __name__ == "__main__":
    main()
