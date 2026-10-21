#!/usr/bin/env python3
"""
vacuum_check.py — is the electrostatic potential in the vacuum of a slab clean enough to be a reference?

Background
----------
There is no charge between two slab images, so Poisson's equation, d2V/dz2 = -rho/eps0 = 0,
allows only a straight line in the vacuum: a constant (the vacuum level), plus a uniform field
(a ramp) if the slab carries a dipole and no correction is applied, plus a step where the
dipole correction (IDIPOL/LDIPOL) places its compensating dipole layer. Anything else in the
vacuum is an error of the calculation. The usual one is a long-wavelength (G -> 0) component
of the Hartree potential that has not converged: in the charge-density mixing the longest
wavelengths are the slowest modes, so a run stopped on a loose EDIFF, or with mixing settings
that damp them poorly, can carry a smooth bump across the whole cell. The total energy hardly
notices, but the vacuum level — and every IP, EA and work function referenced to it — moves.

The planar average V(z) of the LOCPOT is fitted inside the vacuum gap, more than --margin Å from
every atom, to

    V(z) = c0 + c1 (z - z_mid) [+ D H(z - z_step)] + A cos(2 pi (z - z_mid) / L)

L = cell length along the normal, z_mid = middle of the vacuum gap, H = step function (used only
when a step is found in the vacuum). A clean vacuum gives A = 0 to within the numerical noise.
Reported: A (meV, the headline number) and a free-phase cosine amplitude for comparison, the ramp
c1 (meV/Å), the step D, the two-side difference of the vacuum potential, the RMS residual, the
mixing / EDIFF / dipole settings of the INCAR next to the LOCPOT, and a verdict (OK / WARN / BAD).

Write the LOCPOT with LVHAR = .TRUE. (ionic + Hartree potential). The LOCPOT is read streaming,
first data block only (non-collinear runs write four), so large files need little memory.

Usage
-----
    python vacuum_check.py LOCPOT                             # one file
    python vacuum_check.py run_dir                            # the LOCPOT inside a run folder
    python vacuum_check.py "runs/*" --csv vacuum.csv          # many runs (quote the glob)
    python vacuum_check.py run_dir --plot vacuum.png --margin 4 --warn 2 --bad 10

Author: Israel C. Ribeiro — https://github.com/israel-c-ribeiro/VASP_tools (MIT licence)
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import itertools
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

__version__ = "1.1.0"

INCAR_KEYS = ["ALGO", "EDIFF", "NELM", "IMIX", "AMIX", "BMIX", "AMIN", "IDIPOL", "LDIPOL", "LVHAR", "LVTOT"]


# =========================================================================== reader
def _open(path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def read_locpot_profile(path, axis=2, chunk_lines=200_000):
    """Planar average of the first data block of a LOCPOT, read streaming.

    Returns (profile (n_axis,), cell 3x3 Å, symbols, cartesian positions Å). The grid is never
    held in memory: values are summed plane by plane chunk after chunk (x runs fastest, then y,
    then z, as VASP writes them). Gzipped files (LOCPOT.gz) are read directly."""
    with _open(path) as f:
        head = [next(f) for _ in range(6)]
        scale = float(head[1].split()[0])
        cell = np.array([[float(t) for t in head[i].split()[:3]] for i in (2, 3, 4)])
        if scale < 0:                                    # negative scale = target volume
            scale = (abs(scale) / abs(np.linalg.det(cell))) ** (1 / 3)
        cell = cell * scale
        names = head[5].split()
        ln = head[5]
        if not all(t.isdigit() for t in names):          # VASP 5 species line present
            ln = next(f)
        else:
            names = [f"X{i}" for i in range(len(names))]
        counts = [int(x) for x in ln.split()]
        symbols = [n for n, c in zip(names, counts) for _ in range(c)]
        ln = next(f)
        if ln.strip()[:1] in "Ss":                       # selective dynamics
            ln = next(f)
        direct = ln.strip()[:1] in "Dd"
        coords = np.array([[float(t) for t in next(f).split()[:3]] for _ in symbols]).reshape(-1, 3)
        ln = next(f)
        while not ln.strip():
            ln = next(f)
        nx, ny, nz = (int(x) for x in ln.split()[:3])
        n = nx * ny * nz
        nplanes = (nx, ny, nz)[axis]
        sums = np.zeros(nplanes)
        done = 0
        while done < n:
            lines = list(itertools.islice(f, chunk_lines))
            if not lines:
                break
            try:
                vals = np.array("".join(lines).split()[: n - done], dtype=float)
            except ValueError as exc:
                raise ValueError(f"LOCPOT grid unreadable after {done} of {n} values ({exc}) — "
                                 "truncated or corrupted file?") from None
            idx = done + np.arange(vals.size)
            plane = idx // (nx * ny) if axis == 2 else (idx % nx if axis == 0 else (idx // nx) % ny)
            sums += np.bincount(plane, weights=vals, minlength=nplanes)
            done += vals.size
    if done < n:
        raise ValueError(f"LOCPOT grid truncated ({done} of {n} values) — was the file copied completely?")
    cart = coords @ cell if direct else coords * scale
    return sums / (n // nplanes), cell, symbols, cart


def layer_height(cell, axis=2):
    """Distance between lattice planes normal to ``axis`` (Å): the length of the profile."""
    other = [cell[a] for a in range(3) if a != axis]
    return abs(np.linalg.det(cell)) / np.linalg.norm(np.cross(*other))


def read_incar(folder):
    """INCAR tags next to the LOCPOT as {TAG: value string}; {} when there is no INCAR."""
    p = Path(folder) / "INCAR"
    if not p.is_file():
        return {}
    tags = {}
    for line in p.read_text(errors="replace").splitlines():
        line = re.split(r"[!#]", line, maxsplit=1)[0]
        for stmt in line.split(";"):
            if "=" in stmt:
                k, v = stmt.split("=", 1)
                if k.strip() and v.strip():
                    tags[k.strip().upper()] = v.split()[0]
    return tags


def is_true(value):
    return str(value).strip(".").upper().startswith("T")


# =========================================================================== analysis
def vacuum_window(n, cell, cart, axis=2, margin=3.0):
    """Coordinates of the vacuum gap. Returns dict(u, W, L, start): u = distance (Å) of each grid
    plane above the outermost atom of the top surface (0 <= u < L), W = width of the largest
    atom-free gap, L = profile length."""
    L = layer_height(cell, axis)
    frac = np.sort((cart @ np.linalg.inv(cell))[:, axis] % 1.0)
    gaps = np.diff(np.r_[frac, frac[0] + 1.0])
    k = int(np.argmax(gaps))
    u = ((np.arange(n) / n - frac[k]) % 1.0) * L
    return dict(u=u, W=float(gaps[k] * L), L=float(L), start=float(frac[k] * L))


def fit_vacuum(u, v, W, L, margin=3.0, step="auto", step_min=0.02, step_exclude=0.5, side_width=1.0):
    """Fit the planar average ``v`` at heights ``u`` (Å above the top surface) inside the vacuum.

    Returns a dict with the fit coefficients (eV, eV/Å), the cosine amplitudes, residuals and the
    arrays needed for plotting. ``step``: 'auto' (fit a step if the largest jump between two
    neighbouring planes exceeds ``step_min`` eV), 'on' or 'off'."""
    sel = (u > margin) & (u < W - margin)
    order = np.argsort(u[sel])
    uu, vv = u[sel][order], v[sel][order]
    if uu.size < 8 or uu[-1] - uu[0] < 2.0:
        raise RuntimeError(f"vacuum gap of {W:.2f} Å leaves no fit window with --margin {margin} Å "
                           f"({uu.size} grid planes): add vacuum or lower --margin")
    width = float(uu[-1] - uu[0])
    umid = W / 2.0
    x = uu - umid
    z_step = None
    if step != "off":
        dv = np.diff(vv)
        jump = np.abs(dv - np.median(dv))
        j = int(np.argmax(jump))
        if step == "on" or jump[j] > step_min:
            z_step = float(0.5 * (uu[j] + uu[j + 1]))
    keep = np.ones(uu.size, bool) if z_step is None else np.abs(uu - z_step) > step_exclude
    base = [np.ones_like(x), x] + ([(uu > z_step).astype(float)] if z_step is not None else [])
    B = np.column_stack(base)
    cosv = np.cos(2 * np.pi * x / L)
    sinv = np.sin(2 * np.pi * x / L)
    E = np.column_stack([B, cosv])
    F = np.column_stack([B, cosv, sinv])
    ce = np.linalg.lstsq(E[keep], vv[keep], rcond=None)[0]
    cf = np.linalg.lstsq(F[keep], vv[keep], rcond=None)[0]
    cl = np.linalg.lstsq(B[keep], vv[keep], rcond=None)[0]
    A = float(ce[-1])
    lin_part = B @ ce[:-1]                               # constant + ramp (+ step) of the full fit
    resid = vv - E @ ce
    first = keep & (uu - uu[0] <= side_width)
    last = keep & (uu[-1] - uu <= side_width)
    return dict(
        cos_amp=A, cos_amp_free=float(np.hypot(cf[-2], cf[-1])),
        bow=float(abs(A) * (1 - np.cos(np.pi * width / L))),
        ramp=float(ce[1]), step=float(-ce[2]) if z_step is not None else 0.0, z_step=z_step,
        V_mid=float(ce[0] + A), V_mean=float(vv[keep].mean()),
        V_upper=float(vv[first].mean()), V_lower=float(vv[last].mean()),
        two_side=float(vv[first].mean() - vv[last].mean()),
        rms=float(np.sqrt(np.mean(resid[keep] ** 2))),
        rms_linear=float(np.sqrt(np.mean((vv[keep] - B[keep] @ cl) ** 2))),
        width=width, n_points=int(keep.sum()),
        _u=uu, _v=vv, _keep=keep, _lin=lin_part, _cos=A * cosv)


def judge(r, warn=5.0, bad=20.0, ramp_warn=2.0, rms_warn=5.0, min_window=5.0):
    """Verdict (OK / WARN / BAD) and the reasons, from a record of :func:`check` (thresholds in meV)."""
    reasons = []
    a = abs(r["cos_amp_meV"])
    if a >= warn:
        level = "BAD" if a >= bad else "WARN"
        reasons.append((level, f"long-wavelength residual in the vacuum (cosine amplitude {r['cos_amp_meV']:+.1f} meV): "
                               "the electrostatic potential is not converged; tighten EDIFF, restart from the "
                               "converged density, check the mixing (AMIX, BMIX, AMIN) and compare two runs"))
    ldipol = is_true(r.get("LDIPOL", "F"))
    if abs(r["ramp_meV_per_A"]) >= ramp_warn:
        msg = (f"ramp of {r['ramp_meV_per_A']:+.2f} meV/Å despite LDIPOL: check IDIPOL (the normal) and DIPOL (the slab centre)"
               if ldipol else
               f"uniform field in the vacuum ({r['ramp_meV_per_A']:+.2f} meV/Å): an uncorrected slab dipole, so the vacuum "
               "level is not defined; use IDIPOL = 3 with LDIPOL = .TRUE. (or a symmetric slab)")
        reasons.append(("WARN", msg))
    if r["step_eV"] and r.get("incar") and not ldipol:
        reasons.append(("WARN", f"a step of {r['step_eV']:+.3f} eV inside the vacuum, but LDIPOL is not set: "
                                "a step is only expected with a dipole correction"))
    if r["rms_meV"] >= rms_warn:
        reasons.append(("WARN", f"the vacuum is not a line + cosine (RMS residual {r['rms_meV']:.1f} meV): vacuum too "
                                "thin, a charged cell, or a potential that contains exchange-correlation"))
    if is_true(r.get("LVTOT", "F")) and not is_true(r.get("LVHAR", "F")):
        reasons.append(("WARN", "LOCPOT written with LVTOT: it contains the exchange-correlation potential, which decays "
                                "slowly into the vacuum; use LVHAR = .TRUE. for vacuum levels"))
    if r["window_A"] < min_window:
        reasons.append(("WARN", f"fit window only {r['window_A']:.1f} Å: the cosine is poorly determined; "
                                "more vacuum or a smaller --margin"))
    verdict = "BAD" if any(lv == "BAD" for lv, _ in reasons) else ("WARN" if reasons else "OK")
    return verdict, [m for _, m in reasons]


def check(path, axis=2, margin=3.0, step="auto", step_min=0.02, step_exclude=0.5, side_width=1.0,
          thresholds=None, keep_arrays=False):
    """Analyse one LOCPOT (or the LOCPOT inside a folder). Returns a flat record (dict)."""
    p = Path(path)
    if p.is_dir():
        hits = [p / n for n in ("LOCPOT", "LOCPOT.gz") if (p / n).is_file()]
        if not hits:
            raise FileNotFoundError(f"no LOCPOT in {p}")
        p = hits[0]
    prof, cell, symbols, cart = read_locpot_profile(p, axis)
    win = vacuum_window(len(prof), cell, cart, axis, margin)
    fit = fit_vacuum(win["u"], prof, win["W"], win["L"], margin, step, step_min, step_exclude, side_width)
    tags = read_incar(p.parent)
    r = dict(run=p.parent.name if p.name.startswith("LOCPOT") else p.name, file=str(p),
             L_A=win["L"], vac_gap_A=win["W"], window_A=fit["width"], n_points=fit["n_points"],
             cos_amp_meV=1000 * fit["cos_amp"], cos_amp_free_meV=1000 * fit["cos_amp_free"],
             bow_meV=1000 * fit["bow"], ramp_meV_per_A=1000 * fit["ramp"], step_eV=fit["step"],
             two_side_diff_eV=fit["two_side"], V_upper_eV=fit["V_upper"], V_lower_eV=fit["V_lower"],
             V_vac_mean_eV=fit["V_mean"], rms_meV=1000 * fit["rms"], rms_linear_meV=1000 * fit["rms_linear"],
             incar=bool(tags))
    r.update({k: tags.get(k, "-") for k in INCAR_KEYS})
    r["verdict"], r["reasons"] = judge(r, **(thresholds or {}))
    if keep_arrays:
        r["_plot"] = dict(u=win["u"], prof=prof, W=win["W"], L=win["L"], margin=margin,
                          **{k: fit[k] for k in ("_u", "_v", "_keep", "_lin", "_cos", "z_step")})
    return r


def _job(args):
    path, kw = args
    try:
        return check(path, **kw), None
    except Exception as exc:                              # report and continue with the batch
        tb = traceback.extract_tb(exc.__traceback__)
        where = f" [{tb[-1].name}(), line {tb[-1].lineno}]" if tb else ""
        return None, (str(path), f"{type(exc).__name__}: {exc}{where}")


# =========================================================================== output
COLUMNS = ["run", "verdict", "cos_amp_meV", "cos_amp_free_meV", "bow_meV", "ramp_meV_per_A", "step_eV",
           "two_side_diff_eV", "V_upper_eV", "V_lower_eV", "V_vac_mean_eV", "rms_meV", "rms_linear_meV",
           "L_A", "vac_gap_A", "window_A", "n_points"] + INCAR_KEYS + ["reasons", "file"]


def write_csv(path, records):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = {k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()}
            row["reasons"] = " | ".join(r["reasons"])
            w.writerow(row)


def _nz(x, digits):
    """``x``, with a value that rounds to zero returned as 0.0 (printed as 0, not -0)."""
    return 0.0 if round(x, digits) == 0 else x


def print_table(records):
    print(f"{'run':<22s} {'gap':>6s} {'window':>6s} {'A_cos':>7s} {'A_free':>7s} {'ramp':>7s} {'step':>7s} "
          f"{'rms':>5s}  {'EDIFF':>6s} {'AMIX':>5s} {'BMIX':>6s} {'LDIPOL':>6s}  verdict")
    print(f"{'':<22s} {'Å':>6s} {'Å':>6s} {'meV':>7s} {'meV':>7s} {'meV/Å':>7s} {'eV':>7s} {'meV':>5s}")
    for r in records:
        print(f"{r['run'][:22]:<22s} {r['vac_gap_A']:6.2f} {r['window_A']:6.2f} {_nz(r['cos_amp_meV'], 1):7.1f} "
              f"{r['cos_amp_free_meV']:7.1f} {_nz(r['ramp_meV_per_A'], 2):7.2f} {_nz(r['step_eV'], 3):7.3f} {r['rms_meV']:5.1f}  "
              f"{r['EDIFF'][:6]:>6s} {r['AMIX'][:5]:>5s} {r['BMIX'][:6]:>6s} {r['LDIPOL'][:6]:>6s}  {r['verdict']}")


COLORS = {"OK": "#009E73", "WARN": "#E69F00", "BAD": "#D55E00"}


def plot(path, records):
    """One column per run: the whole profile with the fit window (top) and the vacuum residual
    after removing the straight line, with the fitted cosine (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = [r for r in records if "_plot" in r][:6]
    n = len(recs)
    fig, axes = plt.subplots(2, n, figsize=(2.9 * n + 0.4, 4.6), squeeze=False)
    for j, r in enumerate(recs):
        d = r["_plot"]
        o = np.argsort(d["u"])
        top, bot = axes[0, j], axes[1, j]
        top.plot(d["u"][o], d["prof"][o], color="#0072B2", lw=0.9, label="planar average")
        top.axvspan(d["_u"][0], d["_u"][-1], color="#56B4E9", alpha=0.18, lw=0, label="fit window")
        top.axvspan(d["W"], d["L"], color="0.85", alpha=0.6, lw=0, label="slab")
        top.set_xlim(0, d["L"])
        top.set_title(f"{r['run']}  ·  {r['verdict']}", fontsize=8.5, color=COLORS[r["verdict"]], fontweight="bold")
        top.set_xlabel("Height above the top surface (Å)", fontsize=8)
        k = d["_keep"]
        res = 1000 * (d["_v"] - d["_lin"])
        bot.axhline(0, color="0.5", lw=0.7)
        bot.plot(d["_u"][k], res[k], "o", ms=2.2, color="#0072B2", label="data − line")
        bot.plot(d["_u"], 1000 * d["_cos"], color="#D55E00", lw=1.5, label="fitted cosine")
        if d["z_step"] is not None:
            bot.axvline(d["z_step"], color="0.4", ls=":", lw=1)
        lim = max(5.0, 1.35 * float(np.max(np.abs(res[k]))))
        bot.set_ylim(-lim, lim)
        bot.set_xlim(d["_u"][0] - 0.3, d["_u"][-1] + 0.3)
        bot.text(0.03, 0.04, f"A = {_nz(r['cos_amp_meV'], 1):+.1f} meV\nramp = {_nz(r['ramp_meV_per_A'], 2):+.2f} meV/Å"
                 + (f"\nstep = {r['step_eV']:+.3f} eV" if r["step_eV"] else ""),
                 transform=bot.transAxes, fontsize=7, va="bottom", ha="left")
        bot.set_xlabel("Height above the top surface (Å)", fontsize=8)
        for ax in (top, bot):
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
        if j == 0:
            top.set_ylabel("Potential (eV)", fontsize=8)
            bot.set_ylabel("Vacuum residual (meV)", fontsize=8)
    axes[0, -1].legend(frameon=False, fontsize=6.5, loc="lower left", bbox_to_anchor=(0.08, 0.02))
    axes[1, -1].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"figure -> {path}")


# =========================================================================== CLI
def expand(patterns):
    """LOCPOT files and run folders from paths or glob patterns (expanded here, for Windows shells)."""
    out = []
    for p in patterns:
        hits = sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p]
        for h in hits:
            if os.path.isdir(h) and not any(os.path.isfile(os.path.join(h, n)) for n in ("LOCPOT", "LOCPOT.gz")):
                continue                                  # a folder without LOCPOT matched by a glob
            out.append(h)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=["."], help="LOCPOT files, run folders or glob patterns (default: .)")
    ap.add_argument("--axis", type=int, default=2, choices=[0, 1, 2], help="surface normal: 0=a, 1=b, 2=c (default)")
    ap.add_argument("--margin", type=float, default=3.0, help="distance (Å) from every atom excluded from the fit (default 3)")
    ap.add_argument("--step", choices=["auto", "on", "off"], default="auto",
                    help="fit a potential step inside the vacuum (dipole correction): auto-detect (default), always, never")
    ap.add_argument("--step-min", type=float, default=0.02, help="smallest jump (eV) between two planes taken as a step (auto)")
    ap.add_argument("--step-exclude", type=float, default=0.5, help="half width (Å) left out of the fit around a step")
    ap.add_argument("--side-width", type=float, default=1.0, help="width (Å) of each vacuum side for the two-side difference")
    ap.add_argument("--warn", type=float, default=5.0, help="cosine amplitude (meV) for WARN (default 5)")
    ap.add_argument("--bad", type=float, default=20.0, help="cosine amplitude (meV) for BAD (default 20)")
    ap.add_argument("--ramp-warn", type=float, default=2.0, help="|ramp| (meV/Å) for WARN (default 2)")
    ap.add_argument("--rms-warn", type=float, default=5.0, help="RMS residual (meV) for WARN (default 5)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel processes for many files")
    ap.add_argument("--csv", help="write one row per run to this CSV file")
    ap.add_argument("--plot", help="figure of the vacuum fit (PNG/PDF/SVG; first 6 runs)")
    ap.add_argument("--quiet", action="store_true", help="do not print the reasons under the table")
    ap.add_argument("--version", action="version", version=__version__)
    a = ap.parse_args(argv)

    if a.bad < a.warn:
        ap.error("--bad must be >= --warn")
    paths = expand(a.paths)
    if not paths:
        print("no LOCPOT or run folder found", file=sys.stderr)
        return 1
    thr = dict(warn=a.warn, bad=a.bad, ramp_warn=a.ramp_warn, rms_warn=a.rms_warn)
    kw = dict(axis=a.axis, margin=a.margin, step=a.step, step_min=a.step_min, step_exclude=a.step_exclude,
              side_width=a.side_width, thresholds=thr, keep_arrays=bool(a.plot))
    jobs = [(p, kw) for p in paths]
    if a.jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            results = list(ex.map(_job, jobs))
    else:
        results = [_job(j) for j in jobs]
    records = [r for r, _ in results if r is not None]
    fails = [e for _, e in results if e is not None]
    for path, msg in fails:
        print(f"[FAIL] {path}: {msg}", file=sys.stderr)
    if not records:
        return 1
    print_table(records)
    if not a.quiet:
        for r in records:
            for msg in r["reasons"]:
                print(f"[{r['verdict']}] {r['run']}: {msg}")
    counts = {v: sum(r["verdict"] == v for r in records) for v in ("OK", "WARN", "BAD")}
    print(f"{len(records)} runs: {counts['OK']} OK, {counts['WARN']} WARN, {counts['BAD']} BAD "
          f"(thresholds: cosine {a.warn:g}/{a.bad:g} meV, ramp {a.ramp_warn:g} meV/Å, rms {a.rms_warn:g} meV)")
    if a.csv:
        write_csv(a.csv, records)
        print(f"{len(records)} records -> {a.csv}")
    if a.plot:
        plot(a.plot, records)
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
