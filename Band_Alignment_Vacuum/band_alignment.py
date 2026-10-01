#!/usr/bin/env python3
"""
band_alignment.py — absolute (vacuum-referenced) band edges from VASP slab calculations.

The problem
-----------
Kohn-Sham eigenvalues of two separate periodic calculations do not share a common
zero: VASP sets the average electrostatic potential of each cell to zero, so a
"VBM = -1.23 eV" from one run cannot be compared with another run. For a slab
with vacuum the natural common reference is the VACUUM LEVEL, the plateau of the
planar-averaged electrostatic potential in the vacuum region:

    E_aligned = E_KS - V_vac          IP = V_vac - VBM      EA = V_vac - CBM
    work function  Phi = V_vac - E_F

What this script does, for one or many run folders
--------------------------------------------------
1. checks that the run finished and that its last SCF loop converged
   (a run that hit NELM still writes all files, with meaningless eigenvalues);
2. checks that the cell really has vacuum (largest atom-free gap >= --min-vacuum);
3. reads the vacuum level from LOCPOT (written with LVHAR = .TRUE.) on both sides
   of the slab and reports the difference (a dipole step or a sloped vacuum);
4. finds the true band edges (highest occupied / lowest unoccupied state at any k,
   using the occupations of EIGENVAL);
5. optionally (``--fragment``, needs PROCAR from LORBIT = 10/11) splits the states by
   their projection on a group of elements (fragment A) and the rest (fragment B),
   and reports the edges of each fragment and the band-alignment type between them,
   e.g. inorganic layer vs organic cations in a hybrid perovskite, or the two
   layers of a heterostructure.

Required files per folder: OUTCAR, EIGENVAL, CONTCAR (or POSCAR), LOCPOT.
Optional: PROCAR (fragment analysis). Only numpy is needed (matplotlib for --plot).

Usage
-----
    python band_alignment.py run_dir                         # one run, true edges only
    python band_alignment.py "runs/*" --csv edges.csv        # many runs (quote the glob)
    python band_alignment.py run_dir --fragment Pb,I --labels inorganic,organic
    python band_alignment.py "runs/*" --fragment Pb,I --gamma-only --jobs 4 --plot levels.png

Author: Israel C. Ribeiro — https://github.com/israel-c-ribeiro/VASP_tools (MIT licence)
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

__version__ = "1.0.0"


# =========================================================================== readers
def read_poscar(path):
    """Return (symbols, cell 3x3 Å, cartesian positions (N,3)) of a VASP 5 POSCAR/CONTCAR."""
    with open(path) as f:
        L = f.readlines()
    scale = float(L[1].split()[0])
    cell = np.array([[float(t) for t in L[i].split()[:3]] for i in (2, 3, 4)])
    factor = scale if scale > 0 else (-scale / abs(np.linalg.det(cell))) ** (1 / 3)   # < 0 = volume
    cell = cell * factor
    names = L[5].split()
    if names[0].lstrip("-").isdigit():
        raise ValueError(f"{path}: no species line (VASP 4 format) — add the element names on line 6")
    counts = [int(x) for x in L[6].split()]
    symbols = [n for n, c in zip(names, counts) for _ in range(c)]
    i = 7
    if L[i].strip()[:1] in "Ss":                       # "Selective dynamics"
        i += 1
    direct = L[i].strip()[:1] in "Dd"
    coords = np.array([[float(t) for t in L[j].split()[:3]] for j in range(i + 1, i + 1 + len(symbols))])
    cart = coords @ cell if direct else coords * factor
    return symbols, cell, cart


def read_eigenval(path):
    """Return (eig, occ, kpts) with eig/occ of shape (nspin, nk, nb)."""
    with open(path) as f:
        L = f.readlines()
    if len(L) < 8:
        raise ValueError("EIGENVAL is truncated or empty")
    nspin = int(L[0].split()[3])
    nelect, nk, nb = (int(float(x)) for x in L[5].split()[:3])
    eig = np.zeros((nspin, nk, nb))
    occ = np.full((nspin, nk, nb), np.nan)
    kpts = np.zeros((nk, 3))
    idx = 7
    for ik in range(nk):
        if idx + nb > len(L):
            raise ValueError(f"EIGENVAL ends at k-point {ik + 1}/{nk} — incomplete run")
        kpts[ik] = [float(x) for x in L[idx].split()[:3]]
        idx += 1
        for ib in range(nb):
            v = L[idx].split()
            if nspin == 1:
                eig[0, ik, ib] = float(v[1])
                if len(v) > 2:
                    occ[0, ik, ib] = float(v[2])
            else:
                eig[:, ik, ib] = float(v[1]), float(v[2])
                if len(v) > 4:
                    occ[:, ik, ib] = float(v[3]), float(v[4])
            idx += 1
        idx += 1                                          # blank line between k-points
    return eig, occ, kpts


def read_procar(path, natoms, nspin, nk, nb):
    """Site projections summed over orbitals, shape (nspin, nk, nb, natoms).

    Works for collinear, spin-polarized (two consecutive sets) and non-collinear PROCARs
    (four blocks per band: total, m_x, m_y, m_z — only the total is used)."""
    with open(path) as f:
        L = f.readlines()
    hdr = L[1].split()
    pk, pb, pi = int(hdr[3]), int(hdr[7]), int(hdr[11])
    if (pk, pb, pi) != (nk, nb, natoms):
        raise ValueError(f"PROCAR (k={pk}, bands={pb}, ions={pi}) does not match "
                         f"EIGENVAL/POSCAR (k={nk}, bands={nb}, ions={natoms})")
    band_lines = [i for i, ln in enumerate(L) if ln.startswith("band ")]
    if len(band_lines) < nspin * nk * nb:
        raise ValueError(f"PROCAR has {len(band_lines)} band blocks, expected {nspin * nk * nb} — truncated")
    first = band_lines[0]
    ion_row = next(i for i in range(first + 1, len(L)) if L[i].split() and L[i].split()[0] == "1")
    header = next(L[i] for i in range(first, ion_row) if L[i].strip().startswith("ion"))
    norb = len(header.split()) - 2                        # minus 'ion' and 'tot'
    off = ion_row - first
    rows = list(itertools.chain.from_iterable(L[b + off: b + off + natoms] for b in band_lines[:nspin * nk * nb]))
    vals = np.array(" ".join(rows).split(), dtype=float)
    ncol = norb + 2
    if vals.size != nspin * nk * nb * natoms * ncol:
        raise ValueError("PROCAR ion rows have an unexpected layout (LORBIT = 12 phases are not supported)")
    proj = vals.reshape(nspin, nk, nb, natoms, ncol)[..., 1:1 + norb].sum(axis=-1)
    return proj


def outcar_info(path):
    """E_fermi (last), LSORBIT, finished flag and convergence of the last SCF loop."""
    info = dict(efermi=np.nan, soc=False, finished=False, scf_converged=None)
    last_loop = None
    with open(path, errors="replace") as f:
        for ln in f:
            if "E-fermi" in ln:
                try:
                    info["efermi"] = float(ln.split()[2])
                except (IndexError, ValueError):
                    pass
            elif "LSORBIT" in ln and "=" in ln:
                info["soc"] = ln.split("=")[1].strip()[:1] in "Tt"
            elif "aborting loop" in ln:
                last_loop = ln
            elif "General timing and accounting" in ln:
                info["finished"] = True
    if last_loop is not None:
        info["scf_converged"] = "unconverged" not in last_loop and "not reached" not in last_loop
    return info


def read_locpot(path):
    """Return (grid (nx, ny, nz) in eV, cell). Reads only the first data block, streaming
    (a non-collinear LOCPOT has four blocks; the first one is the potential)."""
    with open(path) as f:
        head = [next(f) for _ in range(6)]
        scale = float(head[1].split()[0])
        cell = np.array([[float(t) for t in head[i].split()[:3]] for i in (2, 3, 4)]) * scale
        ln = head[5]
        if not all(t.isdigit() for t in ln.split()):     # species line present
            ln = next(f)
        natoms = sum(int(x) for x in ln.split())
        ln = next(f)
        if ln.strip()[:1] in "Ss":
            ln = next(f)
        for _ in range(natoms):
            next(f)
        ln = next(f)
        while not ln.strip():
            ln = next(f)
        nx, ny, nz = (int(x) for x in ln.split())
        n = nx * ny * nz
        first = next(f)
        per = len(first.split())
        txt = first + "".join(itertools.islice(f, -(-n // per) - 1))
    vals = np.array(txt.split()[:n], dtype=float)
    if vals.size < n:
        raise ValueError(f"LOCPOT grid truncated ({vals.size} of {n} values)")
    return vals.reshape((nz, ny, nx)).T, cell


# =========================================================================== vacuum level
def layer_height(cell, axis):
    """Distance between consecutive lattice planes normal to ``axis`` (Å)."""
    other = [cell[a] for a in range(3) if a != axis]
    return abs(np.linalg.det(cell)) / np.linalg.norm(np.cross(*other))


def largest_gap(cell, cart, axis=2):
    """Largest atom-free gap along ``axis``: (start_frac, length_frac, width_Å)."""
    frac = np.sort((cart @ np.linalg.inv(cell))[:, axis] % 1.0)
    gaps = np.diff(np.r_[frac, frac[0] + 1.0])
    k = int(np.argmax(gaps))
    return float(frac[k]), float(gaps[k]), float(gaps[k] * layer_height(cell, axis))


def vacuum_level(grid, cell, cart, axis=2, margin=4.0):
    """Vacuum level from the planar-averaged potential.

    The vacuum region is the largest atom-free gap. Slices closer than ``margin`` Å to the
    slab are discarded (the potential is still rising there). The remaining part is split in
    two halves: 'upper' (just above the top surface) and 'lower' (just below the bottom surface).
    Returns a dict with both plateaus, their difference (dipole step), the spread inside each
    window and the profile itself."""
    prof = grid.mean(axis=tuple(a for a in range(3) if a != axis))
    n = len(prof)
    z = np.arange(n) / n
    start, length, width = largest_gap(cell, cart, axis)
    h = layer_height(cell, axis)
    m = min(margin / h, 0.35 * length)                   # never eat more than 70 % of the gap
    s = (z - start) % 1.0                                # distance from the top surface, in fractions
    upper = (s >= m) & (s <= length / 2 - 0.05 * length)
    lower = (s >= length / 2 + 0.05 * length) & (s <= length - m)
    if upper.sum() < 2 or lower.sum() < 2:
        raise RuntimeError(f"vacuum too thin to find a plateau ({width:.1f} Å)")
    vu, vl = float(prof[upper].mean()), float(prof[lower].mean())
    return dict(V_upper=vu, V_lower=vl, dipole_step=vu - vl,
                spread=float(max(np.ptp(prof[upper]), np.ptp(prof[lower]))),
                vac_width=width, profile=prof)


# =========================================================================== band edges
def classify(eig, occ, efermi, proj=None, fragment_mask=None, threshold=0.70, gamma_index=None):
    """True edges and (optionally) fragment-resolved edges.

    A state belongs to fragment A if its A-fraction >= ``threshold``, to B if its
    B-fraction >= ``threshold``, otherwise it is 'mixed' and is not used as a fragment edge."""
    occupied = np.where(np.isfinite(occ), occ > 0.5, eig <= efermi + 1e-3)
    res = {}
    e_occ, e_unocc = np.where(occupied, eig, -np.inf), np.where(~occupied, eig, np.inf)
    iv, ic = np.unravel_index(np.argmax(e_occ), eig.shape), np.unravel_index(np.argmin(e_unocc), eig.shape)
    res["VBM"], res["CBM"] = float(eig[iv]), float(eig[ic])
    res["VBM_k"], res["CBM_k"] = int(iv[1]) + 1, int(ic[1]) + 1
    res["direct_gap"] = bool(iv[1] == ic[1])
    if proj is None:
        return res
    a = proj[..., fragment_mask].sum(axis=-1)
    b = proj[..., ~fragment_mask].sum(axis=-1)
    tot = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        fa = np.where(tot > 1e-12, a / tot, np.nan)
    res["fA_VBM"], res["fA_CBM"] = float(fa[iv]), float(fa[ic])
    is_a, is_b = fa >= threshold, (1.0 - fa) >= threshold
    kmask = np.ones(eig.shape, bool)
    if gamma_index is not None:                          # fragment-A edges restricted to Gamma
        kmask[:] = False
        kmask[:, gamma_index, :] = True
    for name, sel in (("A", is_a & kmask), ("B", is_b)):
        top = np.where(occupied & sel, eig, -np.inf)
        bot = np.where(~occupied & sel, eig, np.inf)
        res[f"VBM_{name}"] = float(top.max()) if np.isfinite(top.max()) else np.nan
        res[f"CBM_{name}"] = float(bot.min()) if np.isfinite(bot.min()) else np.nan
    res["offset_VB"] = res["VBM_B"] - res["VBM_A"]
    res["offset_CB"] = res["CBM_B"] - res["CBM_A"]
    res["alignment"] = alignment_type(res["VBM_A"], res["CBM_A"], res["VBM_B"], res["CBM_B"])
    return res


def alignment_type(va, ca, vb, cb):
    """Type I (straddling), II (staggered) or III (broken gap) from the edges of A and B."""
    if not all(np.isfinite([va, ca, vb, cb])):
        return "undetermined"
    if ca < vb or cb < va:
        return "III (broken gap)"
    if va >= vb and ca <= cb:
        return "I (A inside B)"
    if vb >= va and cb <= ca:
        return "I (B inside A)"
    return "II (staggered)"


# =========================================================================== one folder
def analyze(folder, fragment=None, threshold=0.70, gamma_only=False, axis=2, min_vacuum=8.0,
            margin=4.0, side="mean", allow_unconverged=False):
    """Analyse one run folder and return a flat record (dict)."""
    folder = Path(folder)
    oc = outcar_info(folder / "OUTCAR")
    if not oc["finished"]:
        raise RuntimeError("OUTCAR has no final timing block — run not finished")
    if oc["scf_converged"] is False and not allow_unconverged:
        raise RuntimeError("last SCF loop did not reach EDIFF — eigenvalues are not meaningful")
    struct = next((folder / n for n in ("CONTCAR", "POSCAR")
                   if (folder / n).is_file() and (folder / n).stat().st_size > 0), None)
    if struct is None:
        raise FileNotFoundError("no CONTCAR/POSCAR")
    symbols, cell, cart = read_poscar(struct)
    _, _, width = largest_gap(cell, cart, axis)
    if width < min_vacuum:
        raise RuntimeError(f"no vacuum: largest atom-free gap is {width:.1f} Å < {min_vacuum} Å (bulk cell?)")
    eig, occ, kpts = read_eigenval(folder / "EIGENVAL")
    proj = mask = gamma = None
    if fragment:
        mask = np.isin(symbols, fragment)
        if not mask.any() or mask.all():
            raise ValueError(f"fragment {fragment} selects {mask.sum()} of {len(symbols)} atoms")
        proj = read_procar(folder / "PROCAR", len(symbols), *eig.shape)
        if gamma_only:
            g = np.where(np.all(np.abs(kpts) < 1e-6, axis=1))[0]
            if not len(g):
                raise RuntimeError("--gamma-only requested but the k-point list has no Gamma point")
            gamma = int(g[0])
    rec = dict(folder=str(folder), n_atoms=len(symbols), soc=oc["soc"], E_fermi=oc["efermi"],
               scf_converged=oc["scf_converged"], vac_width=width)
    rec.update(classify(eig, occ, oc["efermi"], proj, mask, threshold, gamma))
    vac = dict(V_upper=np.nan, V_lower=np.nan, dipole_step=np.nan, spread=np.nan)
    if (folder / "LOCPOT").is_file():
        grid, gcell = read_locpot(folder / "LOCPOT")
        vac = vacuum_level(grid, gcell, cart, axis=axis, margin=margin)
    v = {"mean": 0.5 * (vac["V_upper"] + vac["V_lower"]), "upper": vac["V_upper"], "lower": vac["V_lower"]}[side]
    rec.update(V_vac=v, V_upper=vac["V_upper"], V_lower=vac["V_lower"],
               dipole_step=vac["dipole_step"], vac_spread=vac["spread"])
    for key in ("VBM", "CBM", "VBM_A", "CBM_A", "VBM_B", "CBM_B"):
        if key in rec:
            rec[f"{key}_al"] = rec[key] - v
    rec["gap"] = rec["CBM"] - rec["VBM"]
    rec["IP"], rec["EA"] = v - rec["VBM"], v - rec["CBM"]
    rec["work_function"] = v - rec["E_fermi"]
    return rec


def _job(args):
    folder, kw = args
    try:
        return analyze(folder, **kw), None
    except Exception as exc:                              # report and continue with the batch
        tb = traceback.extract_tb(exc.__traceback__)
        where = f" [{tb[-1].name}(), line {tb[-1].lineno}]" if tb else ""
        return None, (str(folder), f"{type(exc).__name__}: {exc}{where}")


# =========================================================================== output
COLUMNS = ["folder", "VBM", "CBM", "gap", "VBM_k", "CBM_k", "direct_gap", "E_fermi", "V_vac",
           "VBM_al", "CBM_al", "IP", "EA", "work_function",
           "VBM_A", "CBM_A", "VBM_B", "CBM_B", "VBM_A_al", "CBM_A_al", "VBM_B_al", "CBM_B_al",
           "offset_VB", "offset_CB", "alignment", "fA_VBM", "fA_CBM",
           "V_upper", "V_lower", "dipole_step", "vac_spread", "vac_width", "soc", "n_atoms", "scf_converged"]


def write_csv(path, records):
    cols = [c for c in COLUMNS if any(c in r for r in records)]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})


def print_table(records, labels):
    a, b = labels
    print(f"\n{'run':<28s} {'VBM_al':>8s} {'CBM_al':>8s} {'gap':>6s} {'IP':>6s} {'EA':>6s} {'Phi':>6s} {'dV_dip':>7s}")
    for r in records:
        print(f"{Path(r['folder']).name[:28]:<28s} {r['VBM_al']:8.3f} {r['CBM_al']:8.3f} {r['gap']:6.3f} "
              f"{r['IP']:6.3f} {r['EA']:6.3f} {r['work_function']:6.3f} {r['dipole_step']:7.3f}")
    if any("VBM_A" in r for r in records):
        print(f"\n{'run':<28s} {'VBM_' + a:>14s} {'VBM_' + b:>14s} {'CBM_' + a:>14s} {'CBM_' + b:>14s}  alignment")
        for r in records:
            print(f"{Path(r['folder']).name[:28]:<28s} {r['VBM_A_al']:14.3f} {r['VBM_B_al']:14.3f} "
                  f"{r['CBM_A_al']:14.3f} {r['CBM_B_al']:14.3f}  {r['alignment']}")
    print("\n(all energies in eV; *_al and the table values are referenced to the vacuum level)")


def plot_levels(path, records, labels):
    """Energy-level diagram: one column per run, vacuum at 0 eV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frag = any("VBM_A" in r for r in records)
    boxes = ((("A", -0.21, labels[0], "#0072B2"), ("B", 0.21, labels[1], "#E69F00")) if frag
             else (("", 0.0, "gap", "#0072B2"),))
    fig, ax = plt.subplots(figsize=(max(3.2, 1.2 * len(records) + 1.2), 3.4))
    lows = []
    for i, r in enumerate(records):
        for key, dx, lab, col in boxes:
            sfx = f"_{key}_al" if key else "_al"
            v, c = r["VBM" + sfx], r["CBM" + sfx]
            w = 0.38 if frag else 0.5
            ax.bar(i + dx, c - v, bottom=v, width=w, color=col, alpha=0.30, edgecolor=col,
                   label=lab if (i == 0 and frag) else None)
            ax.text(i + dx, v - 0.08, f"{v:.2f}", ha="center", va="top", fontsize=6.5)
            ax.text(i + dx, c + 0.08, f"{c:.2f}", ha="center", va="bottom", fontsize=6.5)
            lows.append(v)
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.text(len(records) - 0.45, 0.06, "vacuum level", ha="right", va="bottom", fontsize=7, color="0.4")
    ax.set_ylim(min(lows) - 0.8, 0.5)
    ax.set_xlim(-0.6, len(records) - 0.4)
    ax.set_xticks(range(len(records)), [Path(r["folder"]).name for r in records], rotation=20, ha="right")
    ax.set_ylabel("Energy vs vacuum (eV)")
    ax.spines[["top", "right"]].set_visible(False)
    if frag:
        ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    print(f"figure -> {path}")


# =========================================================================== CLI
def expand(patterns):
    out = []
    for p in patterns:
        hits = sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p]
        out += [h for h in hits if os.path.isdir(h)]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*", default=["."], help="run folders or glob patterns (default: .)")
    ap.add_argument("--fragment", help="comma-separated elements of fragment A, e.g. Pb,I (needs PROCAR)")
    ap.add_argument("--labels", default="A,B", help="names of fragments A and B for the output, e.g. inorganic,organic")
    ap.add_argument("--threshold", type=float, default=0.70, help="minimum fragment weight of a state (default 0.70)")
    ap.add_argument("--gamma-only", action="store_true", help="take the fragment-A edges at Gamma only")
    ap.add_argument("--axis", type=int, default=2, choices=[0, 1, 2], help="surface normal: 0=a, 1=b, 2=c (default)")
    ap.add_argument("--min-vacuum", type=float, default=8.0, help="minimum atom-free gap (Å) to accept a slab")
    ap.add_argument("--margin", type=float, default=4.0, help="distance (Å) from the slab ignored in the plateau")
    ap.add_argument("--side", choices=["mean", "upper", "lower"], default="mean",
                    help="which vacuum plateau to use as reference (relevant with a dipole step)")
    ap.add_argument("--allow-unconverged", action="store_true", help="do not reject runs whose last SCF did not converge")
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes for many folders")
    ap.add_argument("--csv", help="write all records to this CSV file")
    ap.add_argument("--plot", help="write an energy-level diagram (PNG/PDF/SVG)")
    ap.add_argument("--version", action="version", version=__version__)
    a = ap.parse_args(argv)

    folders = expand(a.folders)
    if not folders:
        print("no run folder found", file=sys.stderr)
        return 1
    labels = a.labels.split(",")
    kw = dict(fragment=a.fragment.split(",") if a.fragment else None, threshold=a.threshold,
              gamma_only=a.gamma_only, axis=a.axis, min_vacuum=a.min_vacuum, margin=a.margin,
              side=a.side, allow_unconverged=a.allow_unconverged)
    jobs = [(f, kw) for f in folders]
    if a.jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            results = list(ex.map(_job, jobs))
    else:
        results = [_job(j) for j in jobs]
    records = [r for r, _ in results if r is not None]
    fails = [e for _, e in results if e is not None]
    for folder, msg in fails:
        print(f"[FAIL] {folder}: {msg}", file=sys.stderr)
    if not records:
        return 1
    print_table(records, labels)
    for r in records:
        if abs(r["dipole_step"]) > 0.05:
            print(f"[warn] {Path(r['folder']).name}: the two vacuum plateaus differ by {r['dipole_step']:+.3f} eV "
                  f"(dipole); choose --side upper/lower for the surface you care about")
        if r["vac_spread"] > 0.05:
            print(f"[warn] {Path(r['folder']).name}: vacuum potential not flat (spread {r['vac_spread']:.3f} eV) — "
                  f"more vacuum or a dipole correction may be needed")
    if a.csv:
        write_csv(a.csv, records)
        print(f"{len(records)} records -> {a.csv}")
    if a.plot:
        plot_levels(a.plot, records, labels)
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
