#!/usr/bin/env python3
"""
hbonds.py — geometric hydrogen bonds and short D-H...A contacts in periodic VASP structures.

Background
----------
A hydrogen bond D-H...A links a donor D (N, O ...) that carries a hydrogen to an acceptor A
(a halide, O, N ...). In a structure file there are no bonds, only positions, so hydrogen bonds
are counted with geometric criteria (Jeffrey 1997; Steiner 2002; IUPAC 2011):

  * the H...A distance is below a cutoff — several cutoffs are reported, because the right
    value depends on the acceptor (2.5 Å for O/N, 3.0-3.2 Å for I) and is a convention;
  * the D-H...A angle (at H) is at least a minimum value (default 120°), so that a short
    contact "behind" the N-H bond is not counted as a hydrogen bond;
  * optionally, the D...A distance is below a cutoff.

What this script does
---------------------
1. reads a POSCAR/CONTCAR (VASP 5, Direct or Cartesian, selective dynamics, negative scale)
   or the last frame of an XDATCAR;
2. assigns every H to its nearest heavy atom (periodic, every image); an H whose nearest
   heavy atom is a donor element within --dh-max Å is a donor H (C-H hydrogens are ignored);
3. groups the donor H by donor atom (NH3, NH2, OH ...);
4. finds every acceptor within the largest cutoff of every donor H, in every periodic
   image (not only the minimum image), with the D-H...A angle and the D...A distance;
5. counts the contacts under each cutoff, with and without the angle criterion, per donor
   group and per structure, plus the soft count  sum clip((d_max - d)/(d_max - d_min), 0, 1)
   that weights short contacts more;
6. optionally tags each acceptor by the number of metal neighbours (--bridging-metal Pb):
   0 = free, 1 = terminal, >= 2 = bridging, and splits the counts by tag.

Many files at once give one summary row per structure (--summary-csv); --csv writes the
full contact table.

Usage
-----
    python hbonds.py CONTCAR
    python hbonds.py CONTCAR --donors N --acceptors I --bridging-metal Pb --csv contacts.csv
    python hbonds.py "runs/*" --summary-csv hbonds.csv            # one row per structure
    python hbonds.py POSCAR --cutoffs 2.5,3.0 --angle 140 --da-max 3.5 --plot hbonds.png

A folder argument means its CONTCAR (or POSCAR when there is no CONTCAR). Atom indices
are 1-based, as in VESTA. Only the Python standard library and numpy are used
(matplotlib for --plot).

Author: Israel C. Ribeiro — https://github.com/israel-c-ribeiro/VASP_tools (MIT licence)
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import math
import os
import sys
from pathlib import Path

import numpy as np

__version__ = "1.1.0"

DEFAULT_CUTOFFS = (2.8, 3.0, 3.2, 3.5)
TAGS = ("free", "terminal", "bridging")


# =========================================================================== reading
class Structure:
    """Symbols, cell (rows = lattice vectors, Å) and fractional coordinates."""

    def __init__(self, symbols, cell, frac, comment=""):
        self.symbols = list(symbols)
        self.cell = np.asarray(cell, float)
        self.frac = np.asarray(frac, float)
        self.comment = comment

    @property
    def cart(self):
        return self.frac @ self.cell


def _header(L, path):
    """Comment, scale-applied cell, names, counts and the index of the line after the counts."""
    scale = float(L[1].split()[0])
    cell = np.array([[float(t) for t in L[i].split()[:3]] for i in (2, 3, 4)])
    if scale < 0:                                           # negative scale = target volume
        scale = (-scale / abs(np.linalg.det(cell))) ** (1 / 3)
    names = L[5].split()
    if not names or names[0].lstrip("-").isdigit():
        raise ValueError(f"{path}: VASP 4 format (no element line) is not supported; add the element names")
    counts = [int(x) for x in L[6].split()[:len(names)]]
    return cell * scale, scale, names, counts


def read_poscar(path) -> Structure:
    L = Path(path).read_text(errors="replace").splitlines()
    if len(L) < 8:
        raise ValueError(f"{path}: too short to be a POSCAR")
    cell, scale, names, counts = _header(L, path)
    i = 7
    if L[i].strip()[:1] in "Ss":
        i += 1
    direct = L[i].strip()[:1] in "Dd"
    n = sum(counts)
    rows = [L[j].split() for j in range(i + 1, i + 1 + n)]
    if len(rows) < n or any(len(r) < 3 for r in rows):
        raise ValueError(f"{path}: expected {n} coordinate lines")
    xyz = np.array([[float(t) for t in r[:3]] for r in rows])
    frac = xyz if direct else (xyz * scale) @ np.linalg.inv(cell)
    symbols = [e for e, c in zip(names, counts) for _ in range(c)]
    return Structure(symbols, cell, frac, L[0].strip())


def read_xdatcar_last(path) -> Structure:
    """Last frame of an XDATCAR (fixed or variable cell)."""
    L = Path(path).read_text(errors="replace").splitlines()
    cell, scale, names, counts = _header(L, path)
    n = sum(counts)
    starts = [k for k, ln in enumerate(L) if "configuration" in ln.lower()]
    if not starts:
        raise ValueError(f"{path}: no 'Direct configuration' block")
    k = starts[-1]
    if k >= 7 and L[k - 7].strip() == L[0].strip() and k - 7 > 0:   # variable cell: header repeated
        cell, scale, names, counts = _header(L[k - 7:], path)
    rows = [L[j].split() for j in range(k + 1, k + 1 + n)]
    if len(rows) < n:
        raise ValueError(f"{path}: last configuration is incomplete")
    xyz = np.array([[float(t) for t in r[:3]] for r in rows])
    frac = xyz if L[k].strip()[:1] in "Dd" else (xyz * scale) @ np.linalg.inv(cell)
    symbols = [e for e, c in zip(names, counts) for _ in range(c)]
    return Structure(symbols, cell, frac, L[0].strip())


def read_structure(path) -> Structure:
    """POSCAR/CONTCAR, or the last frame of an XDATCAR (detected by name or content)."""
    path = Path(path)
    if "XDATCAR" in path.name.upper():
        return read_xdatcar_last(path)
    head = path.read_text(errors="replace").splitlines()[:10]
    if any("configuration=" in ln.replace(" ", "") for ln in head):
        return read_xdatcar_last(path)
    return read_poscar(path)


# =========================================================================== geometry
def interplanar_spacings(cell) -> np.ndarray:
    """Distance between consecutive lattice planes along each cell vector (Å)."""
    vol = abs(np.linalg.det(cell))
    return np.array([vol / np.linalg.norm(np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])) for k in range(3)])


def pairs_within(src, dst, cell, rcut):
    """Every pair (i in src, j in dst, periodic image) with |r_j + T - r_i| <= rcut.

    Not a minimum-image search: all images within ``rcut`` are returned, so the result is
    correct in small or strongly skewed cells, where ``rcut`` can exceed half an interplanar
    spacing and the same pair can be in contact through two images.
    Returns (i, j, vectors i->j (n, 3), distances) sorted by i then distance."""
    src = np.atleast_2d(np.asarray(src, float))
    dst = np.atleast_2d(np.asarray(dst, float))
    if src.size == 0 or dst.size == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3)), np.zeros(0)
    inv = np.linalg.inv(cell)
    df = (dst[None, :, :] - src[:, None, :]) @ inv
    df -= np.round(df)                                       # every component in [-0.5, 0.5]
    nmax = np.ceil(rcut / interplanar_spacings(cell) + 0.5).astype(int)
    I, J, V = [], [], []
    for s in itertools.product(*(range(-m, m + 1) for m in nmax)):
        v = (df + np.array(s, float)) @ cell
        d2 = np.einsum("ijk,ijk->ij", v, v)
        i, j = np.nonzero(d2 <= rcut * rcut)
        if i.size:
            I.append(i), J.append(j), V.append(v[i, j])
    if not I:
        return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3)), np.zeros(0)
    I, J, V = np.concatenate(I), np.concatenate(J), np.concatenate(V)
    D = np.linalg.norm(V, axis=1)
    order = np.lexsort((D, I))
    return I[order], J[order], V[order], D[order]


def angle_deg(u, v) -> float:
    c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


# =========================================================================== analysis
def ck(c) -> str:
    """Cutoff label: 3.0 -> '3.0', 2.75 -> '2.75' (used in column names such as n_hb_3.0)."""
    t = f"{float(c):g}"
    return t if "." in t or "e" in t else t + ".0"


def analyse(st: Structure, *, donors=("N", "O"), acceptors=("I", "Br", "Cl", "O", "N", "F"),
            dh_max=1.25, cutoffs=DEFAULT_CUTOFFS, angle_min=120.0, da_max=None, da_search=5.0,
            min_h=1, bridging_metal=None, metal_cutoff=3.8, soft=(2.6, 3.5)):
    """Hydrogen-bond analysis of one structure.

    Returns a dict with ``contacts`` (one dict per D-H...A contact within the search radius),
    ``groups`` (one dict per donor group) and ``summary`` (one flat dict for the structure)."""
    cutoffs = tuple(sorted(float(c) for c in cutoffs))
    if not cutoffs:
        raise ValueError("at least one H...A cutoff is needed")
    sym = np.array(st.symbols)
    cart = st.cart
    cell = st.cell
    iH = np.where(sym == "H")[0]
    iHeavy = np.where(sym != "H")[0]
    metals = set(bridging_metal or ())

    # 1. every H to its nearest heavy atom (any image)
    owner, vDH = {}, {}
    n_unbonded = 0
    if iH.size and iHeavy.size:
        i, j, V, D = pairs_within(cart[iH], cart[iHeavy], cell, dh_max)
        for k in range(iH.size):
            m = np.nonzero(i == k)[0]
            if not m.size:
                n_unbonded += 1
                continue
            first = m[0]                                     # sorted by distance
            owner[int(iH[k])] = int(iHeavy[j[first]])
            vDH[int(iH[k])] = -V[first]                      # donor -> H
    else:
        n_unbonded = int(iH.size)

    # 2. donor groups
    gH = {}
    for h, d in owner.items():
        if sym[d] in donors:
            gH.setdefault(d, []).append(h)
    gH = {d: sorted(hs) for d, hs in sorted(gH.items()) if len(hs) >= min_h}
    group_of = {d: g for g, d in enumerate(gH, start=1)}
    donor_h = np.array([h for hs in gH.values() for h in hs], int)
    n_other_h = int(len(owner) - donor_h.size)

    # 3. acceptor tags (number of metal neighbours)
    iA = np.where(np.isin(sym, list(acceptors)) & (sym != "H"))[0]
    n_metal = {}
    if metals:
        iM = np.where(np.isin(sym, list(metals)))[0]
        i, j, V, D = pairs_within(cart[iA], cart[iM], cell, metal_cutoff) if iM.size else (np.zeros(0, int),) * 4
        counts = np.bincount(i, minlength=iA.size) if iA.size else np.zeros(0, int)
        n_metal = {int(a): int(c) for a, c in zip(iA, counts)}

    def tag_of(a):
        if not metals:
            return ""
        return TAGS[min(n_metal.get(a, 0), 2)]

    # 4. contacts H...A
    rsearch = max(max(cutoffs), soft[1] if soft else 0.0)
    contacts = []
    if donor_h.size and iA.size:
        i, j, V, D = pairs_within(cart[donor_h], cart[iA], cell, rsearch)
        for k, jj, vHA, dHA in zip(i, j, V, D):
            h, a = int(donor_h[k]), int(iA[jj])
            d = owner[h]
            vDA = vDH[h] + vHA
            dDA = float(np.linalg.norm(vDA))
            if a == d and dDA < 1e-6:                        # the donor itself (same image)
                continue
            ang = angle_deg(-vDH[h], vHA)
            ok = ang >= angle_min and (da_max is None or dDA <= da_max)
            contacts.append(dict(group=group_of[d], donor=f"{sym[d]}{d + 1}", donor_index=d + 1,
                                 H_index=h + 1, acceptor=f"{sym[a]}{a + 1}", acceptor_index=a + 1,
                                 acceptor_element=str(sym[a]), d_HA=float(dHA), angle=ang, d_DA=dDA,
                                 n_metal=n_metal.get(a, "") if metals else "", tag=tag_of(a),
                                 hbond=bool(ok)))

    # 5. shortest D...A per group (any acceptor, any H)
    dDA_min = {}
    if gH and iA.size:
        dons = np.array(list(gH), int)
        i, j, V, D = pairs_within(cart[dons], cart[iA], cell, da_search)
        for k, jj, dd in zip(i, j, D):
            if dons[k] == iA[jj] and dd < 1e-6:
                continue
            dDA_min.setdefault(int(dons[k]), float(dd))      # sorted: first = shortest

    def soft_term(d):
        lo, hi = soft
        return min(1.0, max(0.0, (hi - d) / (hi - lo)))

    # 6. per group
    groups = []
    for d, hs in gH.items():
        g = group_of[d]
        cg = [c for c in contacts if c["group"] == g]
        hb = [c for c in cg if c["hbond"]]
        row = dict(group=g, donor=f"{sym[d]}{d + 1}", donor_index=d + 1, label=f"{sym[d]}H{len(hs) if len(hs) > 1 else ''}",
                   n_H=len(hs), H_indices=" ".join(str(h + 1) for h in hs))
        for c in cutoffs:
            row[f"n_HA_{ck(c)}"] = sum(x["d_HA"] <= c for x in cg)
            row[f"n_hb_{ck(c)}"] = sum(x["d_HA"] <= c for x in hb)
        best = min(cg, key=lambda x: x["d_HA"]) if cg else None
        bhb = min(hb, key=lambda x: x["d_HA"]) if hb else None
        row.update(d_HA_min=best["d_HA"] if best else math.nan, angle_at_min=best["angle"] if best else math.nan,
                   acceptor_at_min=best["acceptor"] if best else "", tag_at_min=best["tag"] if best else "",
                   d_HA_min_hb=bhb["d_HA"] if bhb else math.nan,
                   d_DA_min=dDA_min.get(d, math.nan),
                   soft=sum(soft_term(x["d_HA"]) for x in hb) if soft else math.nan)
        groups.append(row)

    # 7. per structure
    hb_all = [c for c in contacts if c["hbond"]]
    s = dict(n_atoms=len(sym), n_groups=len(groups), n_donor_H=int(donor_h.size), n_other_H=n_other_h,
             n_unbonded_H=n_unbonded)
    for c in cutoffs:
        s[f"n_HA_{ck(c)}"] = sum(x["d_HA"] <= c for x in contacts)
        s[f"n_hb_{ck(c)}"] = sum(x["d_HA"] <= c for x in hb_all)
        s[f"n_hb_{ck(c)}_per_group"] = s[f"n_hb_{ck(c)}"] / len(groups) if groups else math.nan
        if metals:
            for t in TAGS:
                s[f"n_hb_{ck(c)}_{t}"] = sum(x["d_HA"] <= c and x["tag"] == t for x in hb_all)
    cmax = max(cutoffs)
    s["n_rejected_angle"] = sum(x["d_HA"] <= cmax and not x["hbond"] for x in contacts)
    s["n_groups_without_hb"] = sum(g[f"n_hb_{ck(cmax)}"] == 0 for g in groups)
    dmins = [g["d_HA_min"] for g in groups if np.isfinite(g["d_HA_min"])]
    best = min(contacts, key=lambda x: x["d_HA"]) if contacts else None
    s["d_HA_min"] = best["d_HA"] if best else math.nan
    s["angle_at_min"] = best["angle"] if best else math.nan
    s["d_HA_min_hb"] = min((x["d_HA"] for x in hb_all), default=math.nan)
    s["d_HA_min_group_mean"] = float(np.mean(dmins)) if dmins else math.nan
    das = [g["d_DA_min"] for g in groups if np.isfinite(g["d_DA_min"])]
    s["d_DA_min"] = min(das) if das else math.nan
    s["soft"] = sum(g["soft"] for g in groups) if soft else math.nan
    s["soft_per_group"] = s["soft"] / len(groups) if groups else math.nan
    return dict(contacts=contacts, groups=groups, summary=s, cutoffs=cutoffs)


# =========================================================================== output
def fmt(x, nd=3):
    if isinstance(x, float):
        return "-" if not np.isfinite(x) else f"{x:.{nd}f}"
    return str(x)


def print_report(name, st, res, a):
    s, cut = res["summary"], res["cutoffs"]
    lens = np.linalg.norm(st.cell, axis=1)
    c = st.cell
    angs = [angle_deg(c[1], c[2]), angle_deg(c[0], c[2]), angle_deg(c[0], c[1])]
    print(f"{name}: {s['n_atoms']} atoms, cell {lens[0]:.3f} {lens[1]:.3f} {lens[2]:.3f} Å, "
          f"{angs[0]:.1f} {angs[1]:.1f} {angs[2]:.1f}°")
    extra = f", D...A <= {a.da_max:g} Å" if a.da_max is not None else ""
    print(f"donors {','.join(a.donors)} (H within {a.dh_max:g} Å), acceptors {','.join(a.acceptors)}, "
          f"hydrogen bond = H...A <= cutoff and D-H...A >= {a.angle:g}°{extra}")
    print(f"{s['n_groups']} donor group(s), {s['n_donor_H']} donor H; "
          f"{s['n_other_H']} H on other atoms (e.g. C-H) ignored"
          + (f"; {s['n_unbonded_H']} H with no atom within {a.dh_max:g} Å" if s["n_unbonded_H"] else ""))
    if not res["groups"]:
        print("no donor group found: check --donors and --dh-max")
        return
    print()
    hdr = (f"{'group':>5s} {'donor':>7s} {'label':>6s} " + " ".join(f"{'<=' + ck(x):>5s}" for x in cut)
           + f" {'d_HA_min':>9s} {'angle':>6s} {'acceptor':>9s}" + (f" {'tag':>9s}" if a.bridging_metal else "")
           + f" {'d_DA_min':>9s}")
    print(" " * 21 + "hydrogen bonds with H...A <= cutoff (Å)" if len(cut) > 1 else "")
    print(hdr)
    for g in res["groups"]:
        line = (f"{g['group']:5d} {g['donor']:>7s} {g['label']:>6s} "
                + " ".join(f"{g[f'n_hb_{ck(x)}']:5d}" for x in cut)
                + f" {fmt(g['d_HA_min']):>9s} {fmt(g['angle_at_min'], 1):>6s} {g['acceptor_at_min'] or '-':>9s}"
                + (f" {g['tag_at_min'] or '-':>9s}" if a.bridging_metal else "")
                + f" {fmt(g['d_DA_min']):>9s}")
        print(line)
    print()
    print("hydrogen bonds  " + "  ".join(f"<={ck(x)}: {s[f'n_hb_{ck(x)}']}" for x in cut)
          + f"   (all H...A contacts: " + "  ".join(f"<={ck(x)}: {s[f'n_HA_{ck(x)}']}" for x in cut) + ")")
    if a.bridging_metal:
        for t in TAGS:
            print(f"  {t + ' acceptors':<20s}" + "  ".join(f"<={ck(x)}: {s[f'n_hb_{ck(x)}_{t}']}" for x in cut))
    rej = [x for x in res["contacts"] if x["d_HA"] <= max(cut) and not x["hbond"]]
    if rej:
        r = min(rej, key=lambda x: x["d_HA"])
        print(f"contacts <= {max(cut):g} Å rejected by the angle{' or D...A' if a.da_max is not None else ''}: "
              f"{len(rej)} (shortest {r['d_HA']:.3f} Å, {r['angle']:.1f}°, {r['donor']}-H{r['H_index']}...{r['acceptor']})")
    print(f"shortest H...A {fmt(s['d_HA_min'])} Å ({fmt(s['angle_at_min'], 1)}°), shortest hydrogen bond "
          f"{fmt(s['d_HA_min_hb'])} Å; groups without hydrogen bond <= {max(cut):g} Å: {s['n_groups_without_hb']}")
    if a.soft:
        print(f"soft count sum clip(({a.soft[1]:g} - d)/({a.soft[1]:g} - {a.soft[0]:g}), 0, 1): {s['soft']:.3f} "
              f"({s['soft_per_group']:.3f} per group)")


def print_batch(rows, cut):
    w = max(8, max(len(r["file"]) for r in rows))
    print(f"{'file':<{w}s} {'groups':>6s} " + " ".join(f"{'<=' + ck(x):>5s}" for x in cut)
          + f" {'d_HA_min':>9s} {'d_hb_min':>9s} {'soft':>7s}")
    for r in rows:
        if "error" in r:
            print(f"{r['file']:<{w}s} ERROR: {r['error']}")
            continue
        print(f"{r['file']:<{w}s} {r['n_groups']:6d} " + " ".join(f"{r[f'n_hb_{ck(x)}']:5d}" for x in cut)
              + f" {fmt(r['d_HA_min']):>9s} {fmt(r['d_HA_min_hb']):>9s} {fmt(r['soft']):>7s}")


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def plot(contacts, cutoffs, angle_min, out):
    """H...A distance histogram (left) and D-H...A angle against H...A distance (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.array([c["d_HA"] for c in contacts])
    ang = np.array([c["angle"] for c in contacts])
    ok = np.array([c["hbond"] for c in contacts], bool)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[1, 1.15]))
    hi = max(max(cutoffs), d.max() if d.size else 0) + 0.1
    bins = np.arange(1.45, hi + 0.1, 0.1)
    ax1.hist([d[ok], d[~ok]], bins=bins, stacked=True, color=["#0072B2", "#D55E00"], edgecolor="white", lw=0.5,
             label=[f"angle ≥ {angle_min:g}°", f"angle < {angle_min:g}°"])
    ymax = ax1.get_ylim()[1]
    ax1.set_ylim(0, ymax * 1.45)                             # room for the legend
    ax1.yaxis.get_major_locator().set_params(integer=True)
    for ax in (ax1, ax2):
        for c in cutoffs:
            ax.axvline(c, color="0.4", lw=0.8, ls="--", zorder=0)
            ax.text(c, 1.01, ck(c), transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=6.5, color="0.3")
    ax1.set_xlabel("H···A distance (Å)")
    ax1.set_ylabel("contacts")
    ax1.legend(frameon=False, fontsize=7, loc="upper left")
    ax2.axhspan(0, angle_min, color="#D55E00", alpha=0.08, lw=0)
    ax2.axhline(angle_min, color="#D55E00", lw=0.8)
    ax2.scatter(d[ok], ang[ok], s=22, color="#0072B2", edgecolor="white", lw=0.4, label="hydrogen bond", zorder=3)
    ax2.scatter(d[~ok], ang[~ok], s=22, color="#D55E00", marker="s", edgecolor="white", lw=0.4,
                label="rejected", zorder=3)
    ax2.set_xlim(bins[0], hi)
    ax2.set_ylim(min(80, ang.min() - 5) if ang.size else 80, 182)
    ax2.set_xlabel("H···A distance (Å)")
    ax2.set_ylabel("D–H···A angle (°)")
    ax2.legend(frameon=False, fontsize=7, loc="lower left")
    for ax, lab in ((ax1, "a"), (ax2, "b")):
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.18, 1.06, lab, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


# =========================================================================== CLI
def expand(paths):
    """Expand glob patterns (Windows shells do not) and resolve folders to CONTCAR/POSCAR."""
    out = []
    for p in paths:
        hits = sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p]
        if not hits:
            raise FileNotFoundError(f"no file matches {p}")
        for h in hits:
            if os.path.isdir(h):
                for name in ("CONTCAR", "POSCAR"):
                    f = os.path.join(h, name)
                    if os.path.isfile(f) and os.path.getsize(f) > 0:
                        out.append(f)
                        break
            else:
                out.append(h)
    return out


def _floats(text):
    return tuple(float(x) for x in text.split(",") if x.strip())


def _names(text):
    return tuple(x.strip() for x in text.split(",") if x.strip())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("files", nargs="+", help="POSCAR/CONTCAR/XDATCAR files, run folders or glob patterns")
    ap.add_argument("--donors", type=_names, default=("N", "O"), help="donor elements (default N,O)")
    ap.add_argument("--acceptors", type=_names, default=("I", "Br", "Cl", "O", "N", "F"),
                    help="acceptor elements (default I,Br,Cl,O,N,F)")
    ap.add_argument("--dh-max", type=float, default=1.25, help="max distance H - nearest heavy atom (default 1.25 Å)")
    ap.add_argument("--min-h", type=int, default=1, help="min number of H for a donor group (default 1; 3 = NH3 only)")
    ap.add_argument("--cutoffs", type=_floats, default=DEFAULT_CUTOFFS, help="H...A cutoffs, Å (default 2.8,3.0,3.2,3.5)")
    ap.add_argument("--angle", type=float, default=120.0, help="min D-H...A angle, degrees (default 120)")
    ap.add_argument("--da-max", type=float, default=None, help="max D...A distance, Å (default: no D...A criterion)")
    ap.add_argument("--da-search", type=float, default=5.0, help="radius for the shortest D...A per group (default 5 Å)")
    ap.add_argument("--bridging-metal", type=_names, default=None,
                    help="metal element(s), e.g. Pb: tag acceptors free/terminal/bridging by metal neighbours")
    ap.add_argument("--metal-cutoff", type=float, default=3.8, help="metal-acceptor bond cutoff (default 3.8 Å)")
    ap.add_argument("--soft", type=_floats, default=(2.6, 3.5), help="d_min,d_max of the soft count (default 2.6,3.5)")
    ap.add_argument("--per-group", action="store_true", help="print the per-group table for every file in batch mode")
    ap.add_argument("--csv", help="write every contact (all files) to this CSV")
    ap.add_argument("--groups-csv", help="write the per-group table (all files) to this CSV")
    ap.add_argument("--summary-csv", help="write one row per structure to this CSV")
    ap.add_argument("--plot", help="PNG: H...A histogram and angle-distance scatter (needs matplotlib)")
    a = ap.parse_args(argv)
    if a.soft is not None and len(a.soft) != 2:
        ap.error("--soft needs two numbers: d_min,d_max")
    if a.soft and a.soft[1] <= a.soft[0]:
        ap.error("--soft: d_max must be larger than d_min")
    try:
        files = expand(a.files)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if not files:
        print("ERROR: no structure file found", file=sys.stderr)
        return 1
    opts = dict(donors=a.donors, acceptors=a.acceptors, dh_max=a.dh_max, cutoffs=a.cutoffs, angle_min=a.angle,
                da_max=a.da_max, da_search=a.da_search, min_h=a.min_h, bridging_metal=a.bridging_metal,
                metal_cutoff=a.metal_cutoff, soft=a.soft)
    rows, contacts, groups, failed = [], [], [], 0
    single = len(files) == 1
    for f in files:
        try:
            st = read_structure(f)
            res = analyse(st, **opts)
        except Exception as e:                               # fail loudly, but keep the batch going
            failed += 1
            rows.append(dict(file=f, error=str(e)))
            print(f"ERROR: {f}: {e}", file=sys.stderr)
            continue
        rows.append(dict(file=f, **res["summary"]))
        contacts += [dict(file=f, **c) for c in res["contacts"]]
        groups += [dict(file=f, **g) for g in res["groups"]]
        if single or a.per_group:
            print_report(f, st, res, a)
            if not single:
                print()
    cut = tuple(sorted(a.cutoffs))
    if not single:
        print_batch(rows, cut)
    if a.csv:
        write_csv(a.csv, contacts or [dict(file="")])
        print(f"contacts -> {a.csv}")
    if a.groups_csv:
        write_csv(a.groups_csv, groups or [dict(file="")])
        print(f"groups -> {a.groups_csv}")
    if a.summary_csv:
        write_csv(a.summary_csv, rows)
        print(f"summary -> {a.summary_csv}")
    if a.plot:
        if not contacts:
            print("no contact to plot", file=sys.stderr)
        else:
            plot(contacts, cut, a.angle, a.plot)
            print(f"figure -> {a.plot}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
