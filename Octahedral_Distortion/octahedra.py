#!/usr/bin/env python3
"""
octahedra.py — bond lengths, distortion indices and tilts of the BX6 octahedra of a VASP structure.

Background
----------
In metal-halide and metal-oxide perovskites (3D, and layered Ruddlesden-Popper / Dion-Jacobson
phases) the band edges are built from the B-X bonds of corner-sharing octahedra, so small changes
of the framework move the electronic structure. The standard descriptors are

    bond-length distortion (Baur 1974)     Delta_d  = 1/6 sum_i |d_i - <d>| / <d>
    quadratic elongation (Robinson 1971)   <lambda> = 1/6 sum_i (d_i / l0)^2
                                           l0 = centre-vertex distance of the regular
                                                octahedron with the same volume
    bond-angle variance (Robinson 1971)    sigma^2  = 1/11 sum_(12 cis angles) (theta_i - 90)^2

and, for layered sheets, the B-X-B angle at every bridging ligand (180 deg = no rotation or tilt of
neighbouring octahedra), the tilt of the terminal (axial) bond from the sheet normal and the area
per octahedron in the sheet plane.

What this script does
---------------------
1. reads a POSCAR/CONTCAR (VASP 5, Direct or Cartesian, selective dynamics, negative scale);
2. finds, for every centre atom (--centers, default Pb,Sn,Ge), all ligand atoms (--ligands, default
   I,Br,Cl,O) within the cutoff, over every periodic image (bulk cells, slabs, triclinic cells; a
   ligand can bond to the same centre through two images in a small cell). The cutoff is taken in
   the largest gap of the centre-ligand distance distribution unless --cutoff is given;
3. per octahedron: the six bonds, <d>, Delta_d, <lambda>, sigma^2 and the volume; per ligand: the
   number of centres it bridges. Ligands shared by two centres are bridging (equatorial in an n = 1
   sheet), ligands bonded to one centre are terminal (axial);
4. prints a summary (mean, min, max of every quantity), writes one CSV row per octahedron (--csv),
   handles many structures at once (batch table), and compares two structures, e.g. before and
   after a relaxation (--compare BEFORE AFTER).

Usage
-----
    python octahedra.py CONTCAR
    python octahedra.py "runs/*/CONTCAR" --csv octahedra.csv          # batch (quote the glob)
    python octahedra.py POSCAR --centers Sn --ligands Br --cutoff 3.6
    python octahedra.py --compare POSCAR CONTCAR --plot octahedra.png  # how did the relaxation move it?

A folder argument means its CONTCAR (or POSCAR if there is no CONTCAR). Only numpy is needed
(matplotlib for --plot).

Author: Israel C. Ribeiro — https://github.com/israel-c-ribeiro/VASP_tools (MIT licence)
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np

__version__ = "1.1.0"

DEFAULT_CENTERS = "Pb,Sn,Ge"
DEFAULT_LIGANDS = "I,Br,Cl,O"


# =========================================================================== reader
def read_poscar(path):
    """(symbols, cell (3x3, Å, rows = a, b, c), fractional coordinates) of a VASP 5 POSCAR/CONTCAR."""
    L = Path(path).read_text(errors="replace").splitlines()
    if len(L) < 8:
        raise ValueError(f"{path}: too short to be a POSCAR")
    scale = float(L[1].split()[0])
    cell = np.array([[float(t) for t in L[i].split()[:3]] for i in (2, 3, 4)])
    if scale < 0:                                            # negative scale = target volume
        scale = (-scale / abs(np.linalg.det(cell))) ** (1 / 3)
    cell *= scale
    names = L[5].split()
    if names[0].lstrip("-").isdigit():
        raise ValueError(f"{path}: VASP 4 format (no element line) is not supported")
    names = [n.split("/")[0].split("_")[0] for n in names]  # 'Pb_d/1a2b...' (VASP 6) -> 'Pb'
    counts = [int(x) for x in L[6].split()[:len(names)]]
    i = 7
    if L[i].strip()[:1] in "Ss":
        i += 1
    direct = L[i].strip()[:1] in "Dd"
    n = sum(counts)
    xyz = np.array([[float(t) for t in L[j].split()[:3]] for j in range(i + 1, i + 1 + n)])
    frac = xyz if direct else (xyz * scale) @ np.linalg.inv(cell)
    symbols = [e for e, c in zip(names, counts) for _ in range(c)]
    return symbols, cell, frac


def resolve(paths):
    """Expand glob patterns (Windows shells do not) and folders (-> CONTCAR, else POSCAR)."""
    out = []
    for p in paths:
        hits = sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p]
        if not hits:
            raise FileNotFoundError(f"no file matches {p}")
        for h in hits:
            h = Path(h)
            if h.is_dir():
                f = next((h / n for n in ("CONTCAR", "POSCAR") if (h / n).is_file()), None)
                if f is None:
                    if any(ch in p for ch in "*?["):
                        continue
                    raise FileNotFoundError(f"{h}: no CONTCAR or POSCAR inside")
                h = f
            elif not h.is_file():
                raise FileNotFoundError(f"{h}: not found")
            out.append(h)
    if not out:
        raise FileNotFoundError("no structure found")
    return out


# =========================================================================== geometry
def sheet_normal(cell, normal="c"):
    """Unit normal of the sheet: 'c' = a x b (default), 'a' = b x c, 'b' = c x a, or 'x,y,z' (Cartesian)."""
    a, b, c = cell
    key = str(normal).strip().lower()
    if key in ("a", "b", "c"):
        v = {"c": np.cross(a, b), "a": np.cross(b, c), "b": np.cross(c, a)}[key]
    else:
        v = np.array([float(t) for t in key.replace(" ", ",").split(",") if t], float)
        if v.shape != (3,) or np.linalg.norm(v) == 0:
            raise ValueError(f"--normal must be a, b, c or three numbers, got {normal!r}")
    return v / np.linalg.norm(v)


def pair_vectors(cell, frac, centers, ligands, rmax):
    """Every centre -> ligand vector shorter than ``rmax`` Å over all periodic images.

    Returns (ci, lj, vec, dist): atom indices of the centre and the ligand, Cartesian vectors
    (centre -> that image of the ligand) and their lengths. Exact for any cell shape: the image
    search covers every lattice shift that can bring a ligand within rmax."""
    recip = np.linalg.inv(cell).T                            # rows: reciprocal vectors (no 2 pi)
    nmax = np.ceil(rmax * np.linalg.norm(recip, axis=1) + 0.5).astype(int)
    shifts = np.array(np.meshgrid(*[np.arange(-k, k + 1) for k in nmax], indexing="ij")).reshape(3, -1).T
    ci, lj, vec = [], [], []
    for c in centers:
        d = frac[ligands] - frac[c]
        d -= np.round(d)
        cart = (d[:, None, :] + shifts[None, :, :]) @ cell      # (nlig, nshift, 3)
        r = np.linalg.norm(cart, axis=2)
        k, s = np.nonzero((r < rmax) & (r > 1e-6))
        ci += [c] * len(k)
        lj += list(np.asarray(ligands)[k])
        vec += list(cart[k, s])
    vec = np.array(vec).reshape(-1, 3)
    return np.array(ci, int), np.array(lj, int), vec, np.linalg.norm(vec, axis=1)


def auto_cutoff(dist):
    """Cutoff in the largest gap of the sorted centre-ligand distances between dmin and 1.6 dmin
    (1.6 dmin itself closes the window). Returns (cutoff, (gap_lo, gap_hi), method)."""
    dmin = dist.min()
    top = 1.6 * dmin
    d = np.append(np.sort(dist[dist <= top]), top)
    g = np.diff(d)
    k = int(np.argmax(g))
    if g[k] < 0.1 * dmin:                                    # no clear gap: continuous distribution
        return 1.3 * dmin, (np.nan, np.nan), "1.3 x shortest bond (no clear gap)"
    return 0.5 * (d[k] + d[k + 1]), (d[k], d[k + 1]), "largest gap"


def octahedron_shape(v):
    """Baur index, quadratic elongation, bond-angle variance and volume of one BX6 unit.

    ``v`` = the six centre -> ligand vectors. Trans pairs are the most opposite ligands; the volume
    is that of the closed triangulated surface (one vertex from each trans pair per face), so it is
    exact even when the centre is off the middle of the octahedron."""
    d = np.linalg.norm(v, axis=1)
    baur = float(np.mean(np.abs(d - d.mean()) / d.mean()))
    u = v / d[:, None]
    cos = np.clip(u @ u.T, -1.0, 1.0)
    np.fill_diagonal(cos, 2.0)
    trans = np.argmin(cos, axis=1)
    if not np.array_equal(trans[trans], np.arange(6)):
        return baur, np.nan, np.nan, np.nan                  # no three clean trans pairs
    pairs = sorted({tuple(sorted((i, int(trans[i])))) for i in range(6)})
    ang = np.degrees(np.arccos(cos[np.triu_indices(6, 1)]))
    is_trans = np.array([(i, j) in pairs for i, j in zip(*np.triu_indices(6, 1))])
    sigma2 = float(np.sum((ang[~is_trans] - 90.0) ** 2) / 11.0)
    vol = 0.0
    for sa in (0, 1):
        for sb in (0, 1):
            for sc in (0, 1):
                a, b, c = (v[p[s]] for p, s in zip(pairs, (sa, sb, sc)))
                sign = (-1) ** (sa + sb + sc)
                vol += sign * np.dot(a, np.cross(b, c)) / 6.0
    vol = abs(vol)
    l0 = (3.0 * vol / 4.0) ** (1.0 / 3.0)
    lam = float(np.mean((d / l0) ** 2))
    return baur, lam, sigma2, float(vol)


def _angle(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))


def layer_count(cell, frac, centers, n, tol):
    """Number of sheets of centres per cell along ``n`` and the area of the cell face of the sheet."""
    a, b, c = cell
    faces = {"ab": np.cross(a, b), "bc": np.cross(b, c), "ca": np.cross(c, a)}
    area, period = np.nan, None
    for f in faces.values():
        if abs(np.dot(f, n)) / np.linalg.norm(f) > 0.999:   # this face lies in the sheet plane
            area = float(np.linalg.norm(f))
            period = abs(np.linalg.det(cell)) / area
    h = np.sort((frac[centers] @ cell) @ n)
    if period is not None:
        h = np.sort(h % period)
    groups = 1 + int(np.sum(np.diff(h) > tol))
    if period is not None and groups > 1 and (h[0] + period - h[-1]) <= tol:
        groups -= 1                                          # the first and last sheet are one
    return groups, area


def analyse(path, centers=DEFAULT_CENTERS, ligands=DEFAULT_LIGANDS, cutoff=None, normal="c"):
    """Measure every octahedron of one structure. Returns a dict with 'octahedra' (list of per-centre
    dicts), 'summary' (mean, min, max of every quantity) and the bonding information."""
    if Path(path).is_dir():
        path = resolve([str(path)])[0]
    symbols, cell, frac = read_poscar(path)
    sym = np.array(symbols)
    cen_el = [e for e in centers.split(",") if e in symbols] if isinstance(centers, str) else list(centers)
    lig_el = [e for e in ligands.split(",") if e in symbols] if isinstance(ligands, str) else list(ligands)
    cidx = np.where(np.isin(sym, cen_el))[0]
    lidx = np.where(np.isin(sym, lig_el))[0]
    if not len(cidx):
        raise ValueError(f"{path}: no centre atom ({centers}) in the structure (species: {' '.join(dict.fromkeys(symbols))})")
    if not len(lidx):
        raise ValueError(f"{path}: no ligand atom ({ligands}) in the structure (species: {' '.join(dict.fromkeys(symbols))})")
    if cutoff is None:
        ci, lj, vec, dist = pair_vectors(cell, frac, cidx, lidx, 8.0)
        if not len(dist):
            raise ValueError(f"{path}: no ligand within 8 Å of any centre")
        if 1.6 * dist.min() > 8.0:
            ci, lj, vec, dist = pair_vectors(cell, frac, cidx, lidx, 1.6 * dist.min() + 0.01)
        cut, gap, method = auto_cutoff(dist)
    else:
        cut, gap, method = float(cutoff), (np.nan, np.nan), "user"
        ci, lj, vec, dist = pair_vectors(cell, frac, cidx, lidx, cut)
    keep = dist < cut
    ci, lj, vec, dist = ci[keep], lj[keep], vec[keep], dist[keep]
    n = sheet_normal(cell, normal)

    nbond = {int(k): int((lj == k).sum()) for k in lidx}
    bridging = {k for k, v in nbond.items() if v == 2}
    terminal = {k for k, v in nbond.items() if v == 1}
    overbonded = {k for k, v in nbond.items() if v > 2}
    bxb = {}
    for k in bridging:
        v = vec[lj == k]
        bxb[k] = _angle(-v[0], -v[1])                        # angle at X between X->B1 and X->B2
    is_eq = np.array([k in bridging for k in lj], bool)
    is_ax = np.array([k in terminal for k in lj], bool)
    tilt = np.degrees(np.arccos(np.clip(np.abs(vec @ n) / dist, 0.0, 1.0)))

    octs = []
    for c in cidx:
        m = ci == c
        rec = dict(index=int(c) + 1, element=str(sym[c]), n_ligands=int(m.sum()),
                   d_mean=float(dist[m].mean()) if m.any() else np.nan,
                   d_min=float(dist[m].min()) if m.any() else np.nan,
                   d_max=float(dist[m].max()) if m.any() else np.nan,
                   n_bridging=int((m & is_eq).sum()), n_terminal=int((m & is_ax).sum()),
                   d_eq=float(dist[m & is_eq].mean()) if (m & is_eq).any() else np.nan,
                   d_ax=float(dist[m & is_ax].mean()) if (m & is_ax).any() else np.nan,
                   tilt=float(tilt[m & is_ax].mean()) if (m & is_ax).any() else np.nan,
                   bxb=float(np.mean([bxb[k] for k in lj[m] if k in bxb])) if (m & is_eq).any() else np.nan)
        if m.sum() == 6:
            rec["baur"], rec["lambda"], rec["sigma2"], rec["volume"] = octahedron_shape(vec[m])
        else:
            rec["baur"] = rec["lambda"] = rec["sigma2"] = rec["volume"] = np.nan
        octs.append(rec)

    nlay, face = layer_count(cell, frac, cidx, n, tol=0.5 * float(np.median(dist)) if len(dist) else 1.0)
    area = face * nlay / len(cidx) if np.isfinite(face) else np.nan

    def stat(x):
        x = np.asarray([v for v in x if np.isfinite(v)], float)
        return (float(x.mean()), float(x.min()), float(x.max())) if x.size else (np.nan,) * 3

    summary = {
        "bond": stat(dist), "baur": stat([o["baur"] for o in octs]),
        "lambda": stat([o["lambda"] for o in octs]), "sigma2": stat([o["sigma2"] for o in octs]),
        "volume": stat([o["volume"] for o in octs]),
        "d_eq": stat(dist[is_eq]), "d_ax": stat(dist[is_ax]), "bxb": stat(list(bxb.values())),
        "tilt": stat(tilt[is_ax]), "area": (area, area, area),
    }
    return dict(path=str(path), symbols=symbols, cell=cell, frac=frac, normal=n, cutoff=float(cut),
                cutoff_gap=gap, cutoff_method=method, centers=cen_el, ligands=lig_el,
                bonds=dict(ci=ci, lj=lj, vec=vec, dist=dist), bridging=sorted(bridging),
                terminal=sorted(terminal), overbonded=sorted(overbonded), bxb=bxb,
                n_layers=nlay, octahedra=octs, summary=summary,
                n_complete=sum(o["n_ligands"] == 6 for o in octs))


# =========================================================================== output
ROWS = [  # key, label, format
    ("bond", "B-X bond length (Å)", "{:9.4f}"),
    ("baur", "Baur distortion index Delta_d", "{:9.5f}"),
    ("lambda", "quadratic elongation <lambda>", "{:9.5f}"),
    ("sigma2", "bond-angle variance (deg^2)", "{:9.3f}"),
    ("volume", "octahedron volume (Å^3)", "{:9.3f}"),
    ("d_eq", "equatorial (bridging) B-X (Å)", "{:9.4f}"),
    ("d_ax", "axial (terminal) B-X (Å)", "{:9.4f}"),
    ("bxb", "B-X-B bridging angle (deg)", "{:9.2f}"),
    ("tilt", "axial tilt from normal (deg)", "{:9.2f}"),
    ("area", "area per centre (Å^2)", "{:9.3f}"),
]


def _f(fmt, x):
    return fmt.format(x) if np.isfinite(x) else f"{'-':>{len(fmt.format(0.0))}s}"


def describe(r):
    """Header lines of one analysed structure."""
    cen = {e: sum(1 for o in r["octahedra"] if o["element"] == e) for e in r["centers"]}
    gap = r["cutoff_gap"]
    how = (f"auto, largest gap {gap[0]:.2f}-{gap[1]:.2f} Å" if r["cutoff_method"] == "largest gap"
           else "auto, " + r["cutoff_method"] if r["cutoff_method"] != "user" else "user")
    nb, nt = len(r["bridging"]), len(r["terminal"])
    kind = ("layered sheet (terminal ligands present)" if nt and nb else
            "corner-sharing 3D framework (every ligand bridging)" if nb and not nt else
            "isolated octahedra (no ligand shared)" if nt else "no ligand bonded")
    n = r["normal"]
    return [f"{r['path']}",
            f"  centres {' '.join(f'{e}{k}' for e, k in cen.items())}, ligands {','.join(r['ligands'])}, "
            f"cutoff {r['cutoff']:.2f} Å ({how})",
            f"  octahedra: {r['n_complete']} of {len(r['octahedra'])} with 6 ligands; "
            f"{nb} bridging + {nt} terminal ligands -> {kind}",
            f"  sheets per cell {r['n_layers']}, normal ({n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f})"]


def warnings_for(r):
    out = []
    bad = [o for o in r["octahedra"] if o["n_ligands"] != 6]
    if bad:
        out.append(f"{r['path']}: {len(bad)} centre(s) without exactly 6 ligands within {r['cutoff']:.2f} Å "
                   f"(e.g. atom {bad[0]['index']} {bad[0]['element']}: {bad[0]['n_ligands']}); "
                   "their distortion indices are left empty - check --cutoff")
    if r["overbonded"]:
        out.append(f"{r['path']}: {len(r['overbonded'])} ligand(s) bonded to more than two centres "
                   f"(e.g. atom {r['overbonded'][0] + 1}); not counted as bridging")
    if any(o["n_ligands"] == 6 and not np.isfinite(o["lambda"]) for o in r["octahedra"]):
        out.append(f"{r['path']}: some octahedra are too distorted to pair their trans ligands; "
                   "<lambda> and sigma^2 are left empty for them")
    return out


def print_summary(r):
    for ln in describe(r):
        print(ln)
    print(f"  {'quantity':<32s}{'mean':>9s} {'min':>9s} {'max':>9s}")
    for key, label, fmt in ROWS:
        m, lo, hi = r["summary"][key]
        if key == "area":
            print(f"  {label:<32s}{_f(fmt, m)}")
        else:
            print(f"  {label:<32s}{_f(fmt, m)} {_f(fmt, lo)} {_f(fmt, hi)}")


def print_batch(results):
    head = f"{'structure':<40s} {'oct':>4s} {'<d> Å':>7s} {'Delta_d':>8s} {'<lambda>':>9s} {'sigma2':>8s} {'B-X-B':>7s} {'tilt':>6s}"
    print(head)
    for r in results:
        s = r["summary"]
        name = r["path"] if len(r["path"]) <= 40 else "..." + r["path"][-37:]
        print(f"{name:<40s} {len(r['octahedra']):4d} {_f('{:7.4f}', s['bond'][0])} {_f('{:8.5f}', s['baur'][0])} "
              f"{_f('{:9.5f}', s['lambda'][0])} {_f('{:8.3f}', s['sigma2'][0])} {_f('{:7.2f}', s['bxb'][0])} "
              f"{_f('{:6.2f}', s['tilt'][0])}")


CSV_COLS = ["index", "element", "n_ligands", "d_mean", "d_min", "d_max", "baur", "lambda", "sigma2",
            "volume", "n_bridging", "n_terminal", "d_eq", "d_ax", "bxb", "tilt"]


def _csv_val(x):
    if isinstance(x, float):
        return "" if not np.isfinite(x) else f"{x:.6g}"
    return x


def write_csv(results, out):
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["structure"] + CSV_COLS)
        for r in results:
            for o in r["octahedra"]:
                w.writerow([r["path"]] + [_csv_val(o[k]) for k in CSV_COLS])


def compare(rb, ra, out_csv=None):
    """Print the change of every summary quantity between two structures (and per octahedron to CSV)."""
    print(f"before: {rb['path']}")
    print(f"after : {ra['path']}")
    print(f"  {'quantity (mean)':<32s}{'before':>9s} {'after':>9s} {'change':>9s}")
    for key, label, fmt in ROWS:
        b, a = rb["summary"][key][0], ra["summary"][key][0]
        ch = a - b if np.isfinite(a) and np.isfinite(b) else np.nan
        sfmt = fmt.replace("{:", "{:+")
        print(f"  {label:<32s}{_f(fmt, b)} {_f(fmt, a)} {_f(sfmt, ch)}")
    same = [(o["index"], o["element"]) for o in rb["octahedra"]] == [(o["index"], o["element"]) for o in ra["octahedra"]]
    if not same:
        print("  the two structures do not have the same centre atoms: no per-octahedron comparison")
        if out_csv:
            raise ValueError("--csv with --compare needs the same centre atoms in both structures")
        return
    keys = ["d_mean", "baur", "lambda", "sigma2", "bxb", "tilt"]
    dif = {k: np.array([oa[k] - ob[k] for ob, oa in zip(rb["octahedra"], ra["octahedra"])]) for k in keys}
    print(f"  per octahedron ({len(rb['octahedra'])} matched by atom index), largest |change|:")
    for k, label in zip(keys, ["<d> (Å)", "Delta_d", "<lambda>", "sigma^2 (deg^2)", "B-X-B (deg)", "tilt (deg)"]):
        x = dif[k]
        if np.isfinite(x).any():
            i = int(np.nanargmax(np.abs(x)))
            o = rb["octahedra"][i]
            print(f"    {label:<18s}{x[i]:+10.5f}  (atom {o['index']} {o['element']})")
    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "element"] + [f"{k}_{t}" for k in CSV_COLS[3:] for t in ("before", "after", "change")])
            for ob, oa in zip(rb["octahedra"], ra["octahedra"]):
                row = [ob["index"], ob["element"]]
                for k in CSV_COLS[3:]:
                    row += [_csv_val(float(ob[k])), _csv_val(float(oa[k])), _csv_val(float(oa[k]) - float(ob[k]))]
                w.writerow(row)


# =========================================================================== plot
def plot(results, out, titles=None):
    """Top view of the first sheet of each structure: centres, ligands, bonds and B-X-B angles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(2.9 * len(results) + 0.2, 3.6), squeeze=False)
    for ax, r, title in zip(axes[0], results, titles or [Path(r["path"]).parent.name or r["path"] for r in results]):
        cell, n = r["cell"], r["normal"]
        e1 = cell[0] - np.dot(cell[0], n) * n
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        P = np.column_stack([e1, e2])
        b = r["bonds"]
        cart = r["frac"] @ cell
        cen = sorted({int(c) for c in b["ci"]})
        h = cart[cen] @ n
        first = [c for c, z in zip(cen, h) if abs(z - h.min()) < 1.0] or cen
        corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], float) @ cell
        xy = corners @ P
        ax.plot(xy[:, 0], xy[:, 1], color="0.75", lw=0.8, ls="--", zorder=0)
        for c in first:
            m = b["ci"] == c
            p0 = cart[c] @ P
            for k, v in zip(b["lj"][m], b["vec"][m]):
                q = p0 + v @ P
                if k in r["terminal"]:                      # seen from above: a ring around its foot
                    ax.plot([p0[0], q[0]], [p0[1], q[1]], color="#E69F00", lw=2.0, zorder=1)
                    ax.scatter(*q, s=60, facecolor="none", edgecolor="#E69F00", linewidth=1.3, zorder=5)
                    continue
                ax.plot([p0[0], q[0]], [p0[1], q[1]], color="#CC79A7", lw=1.4, zorder=1, solid_capstyle="round")
                ax.scatter(*q, s=26, color="#CC79A7", zorder=3, edgecolor="white", linewidth=0.4)
            ax.scatter(*p0, s=70, color="#4D4D4D", zorder=4, edgecolor="white", linewidth=0.5)
        shown = set()
        for c in first:                                     # B-X-B angle once per bridging X
            m = b["ci"] == c
            p0 = cart[c] @ P
            for k, v in zip(b["lj"][m], b["vec"][m]):
                if k in r["bxb"]:
                    q = p0 + v @ P
                    if k in shown:
                        continue
                    shown.add(k)
                    off = np.array([-v @ e2, v @ e1])
                    off = 0.45 * off / max(np.linalg.norm(off), 1e-9)
                    ha = "left" if off[0] > 0.15 else "right" if off[0] < -0.15 else "center"
                    va = "bottom" if off[1] > 0.15 else "top" if off[1] < -0.15 else "center"
                    ax.text(q[0] + off[0], q[1] + off[1], f"{r['bxb'][k]:.0f}°", fontsize=6.5,
                            ha=ha, va=va, color="#7A3B69", zorder=5)
        s = r["summary"]
        txt = (f"$\\Delta_d$ = {s['baur'][0]:.4f}\n$\\langle\\lambda\\rangle$ = {s['lambda'][0]:.4f}\n"
               f"$\\sigma^2$ = {s['sigma2'][0]:.1f} deg$^2$\nB–X–B = {s['bxb'][0]:.1f}°\n"
               f"axial tilt = {s['tilt'][0]:.1f}°")
        ax.set_title(title, fontsize=8.5)
        ax.text(0.5, -0.04, txt, transform=ax.transAxes, ha="center", va="top", fontsize=7, linespacing=1.35)
        ax.set_aspect("equal")
        ax.axis("off")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color="#4D4D4D", label="centre B"),
               Line2D([], [], marker="o", ls="-", color="#CC79A7", label="bridging X (B–X–B angle)"),
               Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="#E69F00",
                      markersize=6, label="terminal X (seen from above)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0.0, 1, 0.93))
    fig.subplots_adjust(bottom=0.27)
    fig.savefig(out, dpi=300)
    plt.close(fig)


# =========================================================================== main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("structures", nargs="*", help="POSCAR/CONTCAR files, folders or glob patterns")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="change of every quantity between two structures")
    ap.add_argument("--centers", default=DEFAULT_CENTERS, help=f"centre elements (default {DEFAULT_CENTERS})")
    ap.add_argument("--ligands", default=DEFAULT_LIGANDS, help=f"ligand elements (default {DEFAULT_LIGANDS})")
    ap.add_argument("--cutoff", type=float, help="B-X bond cutoff in Å (default: automatic)")
    ap.add_argument("--normal", default="c", help="sheet normal: c (= a x b, default), a (= b x c), b (= c x a) or x,y,z")
    ap.add_argument("--csv", help="one row per octahedron (with --compare: before/after/change)")
    ap.add_argument("--plot", help="top view of the sheet(s) with the B-X-B angles (PNG/PDF)")
    ap.add_argument("--verbose", action="store_true", help="full summary of every structure in batch mode")
    a = ap.parse_args(argv)
    if bool(a.compare) == bool(a.structures):
        ap.error("give either structures or --compare BEFORE AFTER")
    kw = dict(centers=a.centers, ligands=a.ligands, cutoff=a.cutoff, normal=a.normal)
    try:
        if a.compare:
            paths = [resolve([p])[0] for p in a.compare]
            results = [analyse(p, **kw) for p in paths]
            for r in results:
                for w in warnings_for(r):
                    print("warning: " + w, file=sys.stderr)
            compare(*results, out_csv=a.csv)
            if a.plot:
                plot(results, a.plot, titles=["before", "after"])
        else:
            results = [analyse(p, **kw) for p in resolve(a.structures)]
            for r in results:
                for w in warnings_for(r):
                    print("warning: " + w, file=sys.stderr)
            if len(results) == 1 or a.verbose:
                for i, r in enumerate(results):
                    if i:
                        print()
                    print_summary(r)
            if len(results) > 1:
                if a.verbose:
                    print()
                print_batch(results)
            if a.csv:
                write_csv(results, a.csv)
            if a.plot:
                plot(results[:3], a.plot)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if a.csv:
        print(f"csv -> {a.csv}")
    if a.plot:
        print(f"plot -> {a.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
