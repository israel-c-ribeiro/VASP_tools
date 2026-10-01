#!/usr/bin/env python3
"""
vasp_diagnose.py — did my VASP run really work? A per-run health check.

"The job finished" is not the same as "the result is usable". This script reads the
files of one or many run folders and checks, in order:

  1. status      finished normally / stopped (wall time, crash) / probably still running
  2. electronic  did EVERY ionic step reach EDIFF? (after NELM iterations VASP moves on
                 silently, and a single point that hit NELM still writes all its files)
  3. ionic       relaxation converged? largest residual force on the FREE atoms vs EDIFFG
  4. constraints selective dynamics: how many atoms were fixed, the forces left on them and
                 whether they moved — fixed atoms inherited from an earlier POSCAR are a
                 classic way to get a "converged" relaxation that is not relaxed
  5. cell        volume change of a variable-cell run (> ~3 % -> restart, Pulay stress)
  6. logs        known VASP/Slurm error messages in *.out / *.err / stdout files,
                 each with its usual cause and what to try

Only the Python standard library and numpy are used.

Usage
-----
    python vasp_diagnose.py run_dir                    # detailed report
    python vasp_diagnose.py "relax/*" --summary        # one line per run (quote the glob)
    python vasp_diagnose.py "relax/*" --summary --csv health.csv --only-problems

Exit code: 0 if every run is OK (warnings allowed), 1 otherwise (handy in scripts).

Author: Israel C. Ribeiro — https://github.com/israel-c-ribeiro/VASP_tools (MIT licence)
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

__version__ = "1.0.0"

# Known error messages: (regex, short name, usual cause and what to try)
LOG_PATTERNS = [
    (r"DUE TO TIME LIMIT", "wall time reached",
     "the job was killed by Slurm. Copy CONTCAR to POSCAR and resubmit, or use an auto-restart job script."),
    (r"oom-kill|Out Of Memory|out-of-memory|OOM Killed", "out of memory",
     "use more nodes or fewer MPI tasks per node; check NCORE/KPAR; LREAL = Auto for large cells."),
    (r"ZBRENT: fatal error", "ZBRENT (line minimisation)",
     "the CG optimiser lost its bracket, usually close to convergence. Restart from CONTCAR, "
     "or use IBRION = 1 / a smaller POTIM, and a tighter EDIFF."),
    (r"Error EDDDAV|EDDDAV: Call to ZHEGV failed", "EDDDAV (Davidson)",
     "subspace diagonalisation failed. Try ALGO = Normal or All, check for atoms that are too close."),
    (r"BRMIX: very serious problems", "BRMIX (charge mixing)",
     "the density mixing broke down, often after a large change of geometry or cell. "
     "Start without WAVECAR/CHGCAR or change AMIX/BMIX."),
    (r"ZPOTRF", "ZPOTRF (orthonormalisation)",
     "the orbitals became linearly dependent: a diverging SCF or atoms too close. For hybrid "
     "functionals see ../SCF_Rescue_Hybrid."),
    (r"Sub-Space-Matrix is not hermitian", "non-hermitian subspace matrix",
     "Davidson instability. Try ALGO = Normal or All, or PREC = Accurate."),
    (r"EDWAV: internal error", "EDWAV internal error",
     "precision problem in the wavefunction update; try ALGO = Normal and PREC = Accurate."),
    (r"VERY BAD NEWS! internal error in subroutine (SGRCON|IBZKPT|PRICEL)", "symmetry error",
     "the symmetry analysis failed. Symmetrise the structure, change SYMPREC, or use ISYM = 0."),
    (r"FEXCF|FEXCP", "exchange-correlation table",
     "a density outside the tabulated range, nearly always atoms far too close: check the geometry."),
    (r"segmentation fault|SIGSEGV|forrtl: severe \(174\)", "segmentation fault",
     "often the stack size (add 'ulimit -s unlimited' to the job script), memory, or a binary/MPI mismatch."),
    (r"NODE FAILURE", "node failure", "hardware problem on the cluster side: resubmit."),
    (r"CANCELLED AT(?!.*DUE TO TIME LIMIT)", "cancelled", "the job was cancelled (by you, a dependency or the administrators)."),
]
LOG_GLOBS = ["*.out", "*.err", "slurm-*", "stdout*", "vasp.out", "out", "log*"]


# =========================================================================== parsing
def read_poscar(path):
    """(symbols, cell, fractional coords, selective-dynamics flags or None). Flags: True = free."""
    with open(path) as f:
        L = f.read().splitlines()
    scale = float(L[1].split()[0])
    cell = np.array([[float(t) for t in L[i].split()[:3]] for i in (2, 3, 4)]) * scale
    names, counts = L[5].split(), [int(x) for x in L[6].split()]
    symbols = [n for n, c in zip(names, counts) for _ in range(c)]
    i = 7
    sd = L[i].strip()[:1] in "Ss"
    if sd:
        i += 1
    direct = L[i].strip()[:1] in "Dd"
    rows = [L[j].split() for j in range(i + 1, i + 1 + len(symbols))]
    xyz = np.array([[float(t) for t in r[:3]] for r in rows])
    frac = xyz if direct else (xyz * scale) @ np.linalg.inv(cell)
    flags = np.array([[t.upper().startswith("T") for t in r[3:6]] for r in rows]) if sd else None
    return symbols, cell, frac, flags


def parse_outcar(path):
    """One pass over OUTCAR; returns a dict with everything the checks need."""
    d = dict(nions=None, nsw=None, ibrion=None, isif=None, ediffg=None, energies=[], scf_ok=[],
             forces=None, volumes=[], pressure=None, finished=False, reached=False, elapsed=None)
    with open(path, errors="replace") as f:
        lines = iter(f)
        for ln in lines:
            if "NIONS" in ln and d["nions"] is None:
                d["nions"] = int(ln.split("NIONS =")[1].split()[0])
            elif re.match(r"\s+NSW\s*=", ln):
                d["nsw"] = int(ln.split("=")[1].split()[0])
            elif re.match(r"\s+IBRION\s*=", ln):
                d["ibrion"] = int(ln.split("=")[1].split()[0])
            elif re.match(r"\s+ISIF\s*=", ln):
                d["isif"] = int(ln.split("=")[1].split()[0])
            elif re.match(r"\s+EDIFFG\s*=", ln):
                d["ediffg"] = float(ln.split("=")[1].split()[0])
            elif "aborting loop" in ln:
                d["scf_ok"].append("unconverged" not in ln and "not reached" not in ln)
            elif "free  energy   TOTEN" in ln:
                d["energies"].append(float(ln.split("=")[1].split()[0]))
            elif "volume of cell" in ln:
                d["volumes"].append(float(ln.split(":")[1].split()[0]))
            elif "external pressure" in ln:
                d["pressure"] = float(ln.split("=")[1].split()[0])
            elif "TOTAL-FORCE" in ln and d["nions"]:
                next(lines)
                rows = [next(lines).split() for _ in range(d["nions"])]
                d["forces"] = np.array([[float(x) for x in r[3:6]] for r in rows])
            elif "reached required accuracy" in ln:
                d["reached"] = True
            elif "General timing and accounting" in ln:
                d["finished"] = True
            elif "Elapsed time (sec)" in ln:
                d["elapsed"] = float(ln.split(":")[1])
    return d


def scan_logs(folder, max_bytes=2_000_000):
    """Known error messages in the log files of a folder: list of (file, name, hint)."""
    found, seen = [], set()
    files = {p for g in LOG_GLOBS for p in Path(folder).glob(g) if p.is_file()}
    for p in sorted(files):
        with open(p, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - max_bytes))
            text = f.read().decode(errors="replace")
        for pat, name, hint in LOG_PATTERNS:
            if name not in seen and re.search(pat, text):
                found.append((p.name, name, hint))
                seen.add(name)
    return found


# =========================================================================== checks
def diagnose(folder, running_minutes=20.0):
    """Run every check on one folder; returns (record, list of (level, message))."""
    folder = Path(folder)
    rec = dict(folder=str(folder), state="not started", n_ionic=0, scf_failed_steps=0,
               E_final=np.nan, fmax_free=np.nan, ediffg=np.nan, n_fixed=0, fmax_fixed=np.nan,
               fixed_moved_A=np.nan, dV_percent=np.nan, errors="")
    notes = []
    oc = folder / "OUTCAR"
    if not oc.is_file() or oc.stat().st_size == 0:
        logs = scan_logs(folder)
        rec["errors"] = "; ".join(n for _, n, _ in logs)
        notes += [("FAIL", f"{n} (in {f}): {h}") for f, n, h in logs]
        return rec, notes or [("INFO", "no OUTCAR: the run has not started (or ran elsewhere)")]
    d = parse_outcar(oc)
    rec["n_ionic"] = len(d["energies"])
    rec["E_final"] = d["energies"][-1] if d["energies"] else np.nan

    # 1. status
    age_min = (time.time() - oc.stat().st_mtime) / 60.0
    logs = scan_logs(folder)
    rec["errors"] = "; ".join(n for _, n, _ in logs)
    if d["finished"]:
        rec["state"] = "finished"
        notes.append(("OK", "finished normally" + (f" in {d['elapsed'] / 3600:.2f} h" if d["elapsed"] else "")))
    elif age_min < running_minutes and not logs:
        rec["state"] = "running?"
        notes.append(("INFO", f"no final timing block, OUTCAR written {age_min:.0f} min ago: probably still running"))
    else:
        rec["state"] = "stopped"
        notes.append(("FAIL", f"stopped before the end (last write {age_min / 60:.1f} h ago)"))
    for f, name, hint in logs:
        notes.append(("FAIL", f"{name} (in {f}): {hint}"))

    # 2. electronic convergence of every ionic step
    bad = [i + 1 for i, ok in enumerate(d["scf_ok"]) if not ok]
    rec["scf_failed_steps"] = len(bad)
    if d["scf_ok"] and not d["scf_ok"][-1]:
        notes.append(("FAIL", "the LAST electronic loop did not reach EDIFF: energies, forces and eigenvalues "
                              "of the final structure are not converged (raise NELM, see ../SCF_Rescue_Hybrid)"))
    elif bad:
        steps = ", ".join(map(str, bad[:10])) + (" ..." if len(bad) > 10 else "")
        notes.append(("WARN", f"{len(bad)} intermediate ionic step(s) hit NELM without reaching EDIFF (steps {steps})"))
    elif d["scf_ok"]:
        notes.append(("OK", f"all {len(d['scf_ok'])} electronic loops reached EDIFF"))

    # 3-4. ionic convergence and constraints
    relax = (d["nsw"] or 0) > 0 and d["ibrion"] in (1, 2, 3)
    rec["ediffg"] = d["ediffg"] if d["ediffg"] is not None else np.nan
    flags = None
    start = next((folder / n for n in ("POSCAR",) if (folder / n).is_file()), None)
    final = folder / "CONTCAR" if (folder / "CONTCAR").is_file() and (folder / "CONTCAR").stat().st_size > 0 else None
    if start is not None:
        try:
            sym, cell0, frac0, flags = read_poscar(start)
        except (ValueError, IndexError):
            flags = None
    if d["forces"] is not None:
        F = d["forces"]
        free = flags if flags is not None and len(flags) == len(F) else np.ones(F.shape, bool)
        fnorm_free = np.linalg.norm(np.where(free, F, 0.0), axis=1)
        rec["fmax_free"] = float(fnorm_free.max())
        fixed_atoms = ~free.any(axis=1)
        rec["n_fixed"] = int(fixed_atoms.sum())
        if fixed_atoms.any():
            rec["fmax_fixed"] = float(np.linalg.norm(F[fixed_atoms], axis=1).max())
    if relax:
        tol = abs(d["ediffg"]) if d["ediffg"] is not None and d["ediffg"] < 0 else None
        if d["reached"]:
            notes.append(("OK", f"relaxation converged in {rec['n_ionic']} ionic steps"
                          + (f", max force on free atoms {rec['fmax_free']:.3f} eV/Å (EDIFFG = {d['ediffg']})"
                             if np.isfinite(rec["fmax_free"]) else "")))
        else:
            msg = f"relaxation NOT converged after {rec['n_ionic']} ionic steps"
            if tol and np.isfinite(rec["fmax_free"]):
                msg += f" (max force {rec['fmax_free']:.3f} eV/Å > {tol} eV/Å)"
            notes.append(("FAIL" if d["finished"] else "WARN", msg + ": restart from CONTCAR"))
        if len(d["energies"]) >= 2:
            dE = np.diff(d["energies"])[-3:]
            notes.append(("INFO", "energy change of the last ionic steps: " + ", ".join(f"{x * 1000:+.2f} meV" for x in dE)))
    elif rec["n_ionic"]:
        notes.append(("INFO", "single point (NSW = 0 or IBRION = -1): no ionic convergence to check"))
    if flags is not None:
        counts = {}
        fixed_any = ~flags.all(axis=1)
        for s, fx in zip(sym, fixed_any):
            if fx:
                counts[s] = counts.get(s, 0) + 1
        if counts:
            desc = ", ".join(f"{n} {s}" for s, n in counts.items())
            level = "WARN" if relax else "INFO"
            notes.append((level, f"selective dynamics: {fixed_any.sum()} atom(s) with fixed coordinates ({desc})"
                          + (f"; max force on fully fixed atoms {rec['fmax_fixed']:.3f} eV/Å"
                             if np.isfinite(rec["fmax_fixed"]) else "")
                          + ". Intended? (flags are copied along when you restart from a CONTCAR)"))
            if final is not None:
                _, cell1, frac1, _ = read_poscar(final)
                dfrac = (frac1 - frac0 + 0.5) % 1.0 - 0.5
                moved = np.abs(np.where(~flags, dfrac, 0.0)) @ np.diag(np.linalg.norm(cell1, axis=1))
                rec["fixed_moved_A"] = float(moved.max())
                if not np.isclose(abs(np.linalg.det(cell1)), abs(np.linalg.det(cell0)), rtol=1e-4):
                    notes.append(("INFO", "the cell changed while fractional coordinates of fixed atoms stayed put: "
                                          "their only displacement is the image of the cell strain"))
        else:
            notes.append(("OK", "selective dynamics present, but every coordinate is free"))

    # 5. cell
    if len(d["volumes"]) >= 2 and d["volumes"][0] > 0:
        dv = 100.0 * (d["volumes"][-1] - d["volumes"][0]) / d["volumes"][0]
        rec["dV_percent"] = dv
        if d["isif"] is not None and d["isif"] >= 3:
            level = "WARN" if abs(dv) > 3.0 else "OK"
            notes.append((level, f"volume changed by {dv:+.2f} %"
                          + (": restart from CONTCAR so the plane-wave basis matches the new cell (Pulay stress)"
                             if level == "WARN" else "")))
    if d["pressure"] is not None and d["isif"] is not None and d["isif"] >= 3:
        notes.append(("INFO", f"final external pressure {d['pressure']:+.2f} kB"))

    if rec["state"] == "finished":
        if (d["scf_ok"] and not d["scf_ok"][-1]) or (relax and not d["reached"]):
            rec["state"] = "finished, NOT converged"
        else:
            n_warn = sum(level == "WARN" for level, _ in notes)
            rec["state"] = f"OK ({n_warn} warning{'s' if n_warn > 1 else ''})" if n_warn else "OK"
    return rec, notes


# =========================================================================== output
COLOR = {"OK": "\033[32m", "WARN": "\033[33m", "FAIL": "\033[31m", "INFO": "\033[36m"}


def colored(level, use):
    return f"{COLOR[level]}{level:<4s}\033[0m" if use else f"{level:<4s}"


def expand(patterns):
    out = []
    for p in patterns:
        hits = sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p]
        out += [h for h in hits if os.path.isdir(h)]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*", default=["."], help="run folders or glob patterns (default: .)")
    ap.add_argument("--summary", action="store_true", help="one line per run instead of the full report")
    ap.add_argument("--only-problems", action="store_true", help="hide runs whose state is OK")
    ap.add_argument("--running-minutes", type=float, default=20.0,
                    help="an unfinished OUTCAR written less than this many minutes ago counts as running")
    ap.add_argument("--csv", help="write one row per run to this CSV file")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--version", action="version", version=__version__)
    a = ap.parse_args(argv)

    use_color = sys.stdout.isatty() and not a.no_color and os.environ.get("NO_COLOR") is None
    folders = expand(a.folders)
    if not folders:
        print("no folder found", file=sys.stderr)
        return 1
    results = [diagnose(f, a.running_minutes) for f in folders]
    shown = [(r, n) for r, n in results if not (a.only_problems and r["state"] == "OK")]
    if a.summary:
        print(f"{'run':<32s} {'state':<24s} {'ionic':>5s} {'SCFfail':>7s} {'Fmax':>7s} {'fixed':>5s} {'dV%':>6s}  errors")
        for r, _ in shown:
            print(f"{Path(r['folder']).name[:32]:<32s} {r['state']:<24s} {r['n_ionic']:5d} {r['scf_failed_steps']:7d} "
                  f"{r['fmax_free']:7.3f} {r['n_fixed']:5d} {r['dV_percent']:6.2f}  {r['errors']}")
    else:
        for r, notes in shown:
            print(f"\n=== {r['folder']}  ->  {r['state']}")
            for level, msg in notes:
                print(f"  [{colored(level, use_color)}] {msg}")
    states = [r["state"] for r, _ in results]
    print(f"\n{len(results)} run(s): " + ", ".join(f"{states.count(s)} {s}" for s in dict.fromkeys(states)))
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0][0]))
            w.writeheader()
            for r, _ in results:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
        print(f"table -> {a.csv}")
    return 0 if all(s.startswith("OK") for s in states) else 1


if __name__ == "__main__":
    raise SystemExit(main())
