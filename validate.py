#!/usr/bin/env python3
"""
validate.py - check exifret.py against Corry et al. 2005 (Biophys J 89:3822) and
Deplazes et al. 2012 (J Biomed Opt 17:011005).

Usage:  python validate.py [test ...]      tests: pair pentamer solution plane pairs3d labeling inter irr
        (no argument = all)
Every test prints   simulated | reference | difference.
"""
import itertools
import math
import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import erfc

from exifret import FretParams, run_simulation

NA = 6.02214076e23
LOW_I = 1e-15            # "irradiance ~ 0" (Corry 2005 default)


def mM_to_per_A3(c_mM):
    return c_mM * 1e-3 * NA / 1e27          # mol/L -> molecules per A^3


def row(label, sim, ref, tol=None, err=None):
    d = sim - ref
    flag = ""
    if tol is not None:
        flag = "  OK" if abs(d) <= tol else "  <-- CHECK"
    e = f" +-{err:.3f}" if err is not None else ""
    print(f"  {label:<34s} sim {sim:6.3f}{e:<8s} ref {ref:6.3f}   diff {d:+.3f}{flag}")


# --------------------------------------------------------------------------- #
def test_pair():
    print("\n[1] Single D-A pair, E = 1/(1+(r/R0)^6)   (Corry 2005 Eq. 3)")
    import tempfile, os
    for r in (25, 50, 75):
        f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        f.write(f"0 0 0 D\n{r} 0 0 A\n"); f.close()
        res = run_simulation(FretParams(user_coordinates=f.name, R0=50, photons=60000, irradiance=LOW_I, seed=1))
        os.unlink(f.name)
        row(f"r = {r} A, R0 = 50 A", res["efficiency"], 1 / (1 + (r / 50) ** 6), tol=0.01)


# --------------------------------------------------------------------------- #
def pentamer_analytic(r, R0, Pd):
    """Corry 2005 Appendix A (Eq. 23 averaged over all 2^5 donor/acceptor arrangements)."""
    ang = 2 * np.pi * np.arange(5) / 5
    xy = r * np.c_[np.cos(ang), np.sin(ang)]
    dist = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    E_tot = 0.0
    for conf in itertools.product([0, 1], repeat=5):       # 1 = donor
        conf = np.array(conf)
        nd = conf.sum()
        prob = Pd ** nd * (1 - Pd) ** (5 - nd)
        if nd in (0, 5):
            continue                                       # E = 0 (no donor / no acceptor)
        D, A = np.where(conf == 1)[0], np.where(conf == 0)[0]
        s = np.array([(R0 / dist[d, A]) ** 6 for d in D]).sum(axis=1)
        E_tot += prob * np.mean(s / (1 + s))
    return E_tot


def test_pentamer():
    print("\n[2] Isolated pentamer, R0 = 60 A, ~zero irradiance  (Corry 2005 Fig. 10 / Appendix A)")
    for Pd in (0.5, 0.25, 0.1):
        for r in (10, 40, 60, 80):
            p = FretParams(dimension=2, nmer_size=5, radius=r, particles=1, density=1e-9, avoid_overlap=False,
                           p_donor=Pd, independent_types=True, R0=60, configurations=6000, photons=300,
                           irradiance=LOW_I, seed=7)
            res = run_simulation(p)
            row(f"Pd = {Pd:<4} radius = {r:>2} A", res["efficiency"], pentamer_analytic(r, 60, Pd), tol=0.015, err=res["sem"])
    print("  P(all donors or all acceptors) at Pd=0.5/0.25/0.1 :",
          [round(Pd ** 5 + (1 - Pd) ** 5, 2) for Pd in (0.5, 0.25, 0.1)], " (paper: 0.06, 0.24, 0.59)")


# --------------------------------------------------------------------------- #
def forster_3d(c_total_mM, R0, pd=0.5):
    c0 = 3000 / (2 * math.pi ** 1.5 * NA * (R0 * 1e-8) ** 3) * 1e3      # critical conc. in mM (Eq. 13)
    g = pd * c_total_mM / c0
    return math.sqrt(math.pi) * g * math.exp(g * g) * erfc(g), c0


def test_solution():
    print("\n[3] Fluorophores in 3-D solution vs Foerster's continuum result (Corry 2005 Eq. 13, Figs 3 & 5)")
    for R0 in (40, 60):
        c0 = forster_3d(1, R0)[1]
        print(f"  R0 = {R0} A: critical concentration c0 = {c0:.2f} mM (acceptor conc. at which E = 0.76)")
        for c in (0.5, 1, 2, 4, 8, 14):
            p = FretParams(dimension=3, nmer_size=1, avoid_overlap=False, particles=800,
                           density=mM_to_per_A3(c), R0=R0, p_donor=0.5, configurations=25, photons=4000,
                           irradiance=LOW_I, donor_lifetime=1, acceptor_lifetime=1, seed=3)
            res = run_simulation(p)
            row(f"R0 = {R0}, c_total = {c:>4} mM", res["efficiency"], forster_3d(c, R0)[0], tol=0.02, err=res["sem"])


# --------------------------------------------------------------------------- #
def forster_2d(rho_total, R0, pd=0.5):
    """Exact decay for Poisson-distributed acceptors in a plane (kinetic model):
       S(s) = exp(-s - rho_A*pi*R0^2*Gamma(2/3)*s^(1/3)),  E = 1 - int S ds"""
    rho_A = pd * rho_total
    k = rho_A * math.pi * R0 ** 2 * math.gamma(2 / 3)
    val, _ = quad(lambda s: math.exp(-s - k * s ** (1 / 3)), 0, np.inf, limit=200)
    return 1 - val


def test_plane():
    print("\n[4] Fluorophores randomly in a plane vs exact 2-D result (Corry 2005 Fig. 7; Fung-Stryer/Wolber-Hudson)")
    for R0 in (30, 50):
        for rho in (0.2e-3, 0.5e-3, 1.0e-3):
            p = FretParams(dimension=2, nmer_size=1, avoid_overlap=False, particles=1000, density=rho, R0=R0,
                           configurations=30, photons=4000, irradiance=LOW_I, seed=5)
            res = run_simulation(p)
            row(f"R0 = {R0}, density = {rho*1e3:.1f}e-3 /A^2", res["efficiency"], forster_2d(rho, R0), tol=0.02, err=res["sem"])


# --------------------------------------------------------------------------- #
def test_pairs3d():
    print("\n[5] Linked D-A pairs in 3-D (Corry 2005 Fig. 8B). dilute limit = 1/(1+(r/R0)^6)")
    for sep, R0 in ((50, 60), (50, 50), (70, 60), (70, 50)):
        out = []
        for c in (0.01, 0.5, 1.0, 2.0):
            p = FretParams(dimension=3, nmer_size=2, radius=sep / 2, da_stoichiometry=True, particles=300,
                           density=mM_to_per_A3(c), R0=R0, configurations=30, photons=3000, irradiance=LOW_I, seed=11)
            out.append(run_simulation(p)["efficiency"])
        print(f"  sep {sep} A, R0 {R0} A : dilute theory {1/(1+(sep/R0)**6):.3f} | sim at 0.01/0.5/1/2 mM = "
              + " / ".join(f"{x:.3f}" for x in out))


# --------------------------------------------------------------------------- #
def test_labeling():
    print("\n[6] Tetramers, labelling efficiency, radius -> 0   (Deplazes 2012 Fig. 9)  analytic: E = 1-(1-le/2)^3")
    for le in (1.0, 0.8, 0.6, 0.4, 0.2):
        p = FretParams(dimension=2, nmer_size=4, radius=0.5, particles=400, density=1e-8, avoid_overlap=False,
                       labeling_efficiency=le, R0=30, configurations=60, photons=3000, irradiance=LOW_I, seed=2)
        res = run_simulation(p)
        row(f"labelling efficiency {int(le*100)}%", res["efficiency"], 1 - (1 - le / 2) ** 3, tol=0.02, err=res["sem"])


# --------------------------------------------------------------------------- #
def test_inter():
    print("\n[7] Hosts with one D and one A in 2-D, R0 = 30 A, radius 18 A (separation 36 A)  (Deplazes 2012 Fig. 3b)")
    print("     paper (read from figure): total E = 0.26 at density 0 rising to ~0.63 at 0.4e-3 /A^2")
    print("     radius 18 A = half the separation: single pair E = 1/(1+(36/30)^6) =", round(1 / (1 + 1.2 ** 6), 3))
    for rho in (1e-8, 0.1e-3, 0.2e-3, 0.3e-3, 0.4e-3):
        p = FretParams(dimension=2, nmer_size=2, radius=18, da_stoichiometry=True, particles=300, density=rho, R0=30,
                       configurations=40, photons=3000, irradiance=LOW_I, seed=4)
        res = run_simulation(p)
        print(f"     density {rho*1e3:5.2f}e-3 /A^2  ->  E = {res['efficiency']:.3f} +- {res['sem']:.3f}")
    print("\n  Deplazes 2012 Fig. 4: R0 = 30 A, radius 15 A; case1 mixed, case2 1D+1A (da_stoichiometry), case3 da_separate")
    print("     paper at density 0 : case1 0.25, case2 0.50, case3 0.00 ; all rise to ~0.8-0.86 at 0.8e-3")
    for rho in (1e-8, 0.4e-3, 0.8e-3):
        vals = []
        for kw in (dict(), dict(da_stoichiometry=True), dict(da_separate=True)):
            p = FretParams(dimension=2, nmer_size=2, radius=15, particles=200, density=rho, R0=30,
                           configurations=8, photons=3000, irradiance=LOW_I, seed=9, **kw)
            vals.append(run_simulation(p)["efficiency"])
        print(f"     density {rho*1e3:5.2f}e-3 : case1 {vals[0]:.3f}  case2 {vals[1]:.3f}  case3 {vals[2]:.3f}")


# --------------------------------------------------------------------------- #
def test_irr():
    print("\n[8] Exciton competition at r -> 0 (acceptor still excited, donor busy while fluorescing)")
    print("     renewal-theory reference: x = qk/(1-qk*pf), E = 1/(1+x), qk = k/(k+1/tauA), pf = kf/(1/tauA+kf)")
    import tempfile, os
    from exifret import absorption_rate_per_donor
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    f.write("0 0 0 D\n1 0 0 A\n"); f.close()
    for irr in (1e-3, 1e-2, 3e-2, 1e-1):
        p = FretParams(user_coordinates=f.name, R0=30, irradiance=irr, photons=300000, donor_lifetime=1,
                       acceptor_lifetime=1, extinction_coefficient=7e4, seed=1)
        k = absorption_rate_per_donor(p)
        a, kf = 1.0, 1.0
        qk, pf = k / (k + a), kf / (a + kf)
        ref = 1 / (1 + qk / (1 - qk * pf))
        res = run_simulation(p)
        row(f"irradiance {irr} W/um^2 (k*tauA={k:.2f})", res["efficiency"], ref, tol=0.01)
    os.unlink(f.name)


TESTS = dict(pair=test_pair, pentamer=test_pentamer, solution=test_solution, plane=test_plane,
             pairs3d=test_pairs3d, labeling=test_labeling, inter=test_inter, irr=test_irr)

if __name__ == "__main__":
    names = sys.argv[1:] or list(TESTS)
    for n in names:
        t = time.time()
        TESTS[n]()
        print(f"  ({time.time()-t:.0f} s)")
