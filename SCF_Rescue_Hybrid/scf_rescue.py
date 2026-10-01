#!/usr/bin/env python3
"""
scf_rescue.py — diagnose a self-consistent field (SCF) loop that will not converge, and set up
the two-step rescue that usually fixes hybrid-functional (HSE06, PBE0) runs.

Two commands
------------
diagnose   reads OSZICAR and tells, for every ionic step, how the SCF behaved:
           converged / slowly converging (raise NELM) / oscillating (charge sloshing) /
           diverging. With --plot it draws log10|dE| against the iteration number,
           which is the single most useful picture of an SCF problem.

prepare    turns the production INCAR of a failing run into a two-step job:
             step 1  the same system at the semilocal level (LHFCALC = .FALSE.), with Kerker
                     preconditioning on (BMIX = 1.0) and a modest AMIX, writing a converged WAVECAR;
             step 2  the production INCAR, restarted from that WAVECAR (ISTART = 1, NELMDL = 0)
                     with ALGO = Damped, which damps the orbital update instead of mixing densities.
           NBANDS is fixed explicitly so both steps use the same band count, and the job script
           runs both steps one after the other.

Why it works
------------
Charge sloshing — long-wavelength oscillations of the density from one SCF iteration to the
next — is common in long cells with vacuum (slabs) and in cells with a small gap. The Kerker
preconditioner damps exactly those long-wavelength components; a very small BMIX (e.g.
BMIX = 0.0001, i.e. plain linear mixing) switches it off. Hybrid functionals make every
iteration expensive and their SCF more fragile, so starting them from random orbitals is
risky. Converging the cheap semilocal problem first and then letting a damped algorithm walk
from those orbitals to the hybrid ground state changes only the path of the SCF, not the
state it converges to. The final eigenvalues are those of the production settings.

Usage
-----
    python scf_rescue.py diagnose OSZICAR --plot scf.png
    python scf_rescue.py prepare INCAR --outcar OUTCAR -o rescue/            # NBANDS from the failed run
    python scf_rescue.py prepare INCAR --nbands 480 --exe vasp_ncl -o rescue/ --job-template job.sh

Author: Israel C. Ribeiro — https://github.com/israel-c-ribeiro/VASP_tools (MIT licence)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

__version__ = "1.0.0"


# =========================================================================== diagnose
def read_oszicar(path):
    """List of ionic steps; each is a dict with the algorithm tags, energies E and changes dE."""
    steps, cur = [], dict(algo=[], E=[], dE=[])
    pat = re.compile(r"^\s*([A-Za-z]{2,4})\s*:\s*(\d+)\s+(\S+)\s+(\S+)")
    with open(path, errors="replace") as f:
        for ln in f:
            m = pat.match(ln)
            if m:
                try:
                    cur["E"].append(float(m.group(3)))
                    cur["dE"].append(float(m.group(4)))
                    cur["algo"].append(m.group(1))
                except ValueError:                        # '***' overflow of a diverging run
                    cur["E"].append(np.nan)
                    cur["dE"].append(np.nan)
                    cur["algo"].append(m.group(1))
            elif re.match(r"^\s*\d+\s+F=", ln):
                steps.append(cur)
                cur = dict(algo=[], E=[], dE=[])
    if cur["E"]:
        steps.append(cur)                                 # unfinished ionic step (job killed)
    return steps


def classify_scf(dE, ediff=1e-5, nelm=None, window=20):
    """Verdict for one ionic step from the sequence of energy changes."""
    dE = np.asarray(dE, float)
    n = len(dE)
    if n == 0:
        return "empty"
    if not np.all(np.isfinite(dE)):
        return "DIVERGED (overflow in OSZICAR)"
    a = np.abs(dE) + 1e-14
    if a[-1] < ediff and (nelm is None or n < nelm):
        return "converged"
    tail = dE[-min(window, n):]
    la = np.log10(np.abs(tail) + 1e-14)
    slope = np.polyfit(np.arange(len(la)), la, 1)[0] if len(la) >= 4 else 0.0
    sign_changes = np.mean(np.sign(tail[1:]) != np.sign(tail[:-1])) if len(tail) > 1 else 0.0
    best = a[: max(1, n - len(tail))].min()
    if a[-len(tail):].max() > 1e3 * best and a[-1] > 1.0:
        return "DIVERGING (|dE| grew by > 1000x after its minimum)"
    if sign_changes > 0.6 and slope > -0.02:
        return "OSCILLATING (charge sloshing: dE keeps changing sign without decreasing)"
    if slope < -0.02:
        return "SLOW (still decreasing: raise NELM or improve mixing)"
    return "STALLED (|dE| no longer decreases)"


def cmd_diagnose(a):
    steps = read_oszicar(a.oszicar)
    if not steps:
        print("no SCF iterations found")
        return 1
    print(f"{'ionic':>5s} {'iters':>5s} {'algo':>5s} {'final |dE|':>11s}  verdict")
    bad = 0
    for i, s in enumerate(steps, 1):
        v = classify_scf(s["dE"], a.ediff, a.nelm)
        bad += v != "converged"
        algo = "/".join(dict.fromkeys(s["algo"]))
        last = abs(s["dE"][-1]) if s["dE"] else np.nan
        print(f"{i:5d} {len(s['dE']):5d} {algo:>5s} {last:11.3E}  {v}")
    print(f"\n{bad} of {len(steps)} ionic step(s) not converged (EDIFF = {a.ediff:g}"
          + (f", NELM = {a.nelm}" if a.nelm else "") + ")")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        n = len(steps)
        for i in (range(1, n + 1) if n <= 6 else [1, 2, 3, n - 2, n - 1, n]):
            s = steps[i - 1]
            ax.semilogy(np.arange(1, len(s["dE"]) + 1), np.abs(s["dE"]) + 1e-14, marker=".", ms=3, lw=1,
                        label=f"ionic step {i}")
        ax.axhline(a.ediff, color="0.3", ls="--", lw=0.8)
        ax.text(1, a.ediff, " EDIFF", va="bottom", fontsize=7, color="0.3")
        ax.set_xlabel("SCF iteration")
        ax.set_ylabel("|dE| (eV)")
        ax.spines[["top", "right"]].set_visible(False)
        if n > 1:
            ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=300)
        print(f"figure -> {a.plot}")
    return 0 if bad == 0 else 1


# =========================================================================== prepare
def set_tag(incar: str, tag: str, value: str, comment: str = "") -> str:
    """Set ``tag = value`` (replacing an existing line, else appending) and keep everything else."""
    line = f"{tag:<8s}= {value}" + (f"   ! {comment}" if comment else "")
    pat = re.compile(rf"^[ \t]*{tag}[ \t]*=.*$", re.M | re.I)
    return pat.sub(line, incar, count=1) if pat.search(incar) else incar.rstrip("\n") + "\n" + line + "\n"


def drop_tag(incar: str, tag: str) -> str:
    return re.sub(rf"^[ \t]*{tag}[ \t]*=.*\n?", "", incar, flags=re.M | re.I)


def get_tag(incar: str, tag: str):
    m = re.search(rf"^[ \t]*{tag}[ \t]*=[ \t]*([^!#\n]+)", incar, re.M | re.I)
    return m.group(1).strip() if m else None


def nbands_from_outcar(path):
    with open(path, errors="replace") as f:
        for ln in f:
            if "NBANDS=" in ln:
                return int(ln.split("NBANDS=")[1].split()[0])
    raise ValueError(f"no NBANDS in {path}")


def step_incars(prod: str, nbands: int):
    """(step 1, step 2) INCAR texts from the production INCAR."""
    s1 = prod
    for tag, val, com in [("LHFCALC", ".FALSE.", "step 1: semilocal pre-convergence"),
                          ("ALGO", "Normal", "blocked Davidson"),
                          ("AMIX", "0.2", "modest mixing for a cell with vacuum"),
                          ("BMIX", "1.0", "Kerker preconditioning ON (small BMIX switches it off)"),
                          ("ISTART", "0", ""), ("ICHARG", "2", "start from atomic charges"),
                          ("NELM", "200", ""), ("NELMDL", "-12", "non-self-consistent warm-up"),
                          ("NBANDS", str(nbands), "fixed: identical in both steps"),
                          ("LWAVE", ".TRUE.", "the WAVECAR is the only product of step 1"),
                          ("LCHARG", ".FALSE.", ""), ("LVHAR", ".FALSE.", ""), ("LVTOT", ".FALSE.", "")]:
        s1 = set_tag(s1, tag, val, com)
    s1 = drop_tag(s1, "LORBIT")
    s2 = prod
    for tag, val, com in [("ISTART", "1", "step 2: read the WAVECAR of step 1"),
                          ("ICHARG", "0", "density from the orbitals"),
                          ("NELMDL", "0", "orbitals are already good: no warm-up"),
                          ("ALGO", "Damped", "damped orbital dynamics instead of density mixing"),
                          ("TIME", "0.4", "damping time step; 0.2 if it still oscillates"),
                          ("NELM", "200", ""),
                          ("NBANDS", str(nbands), "same as step 1")]:
        s2 = set_tag(s2, tag, val, com)
    hdr = "# {} - generated by scf_rescue.py (VASP_tools)\n"
    return (hdr.format("STEP 1: semilocal pre-convergence, writes WAVECAR") + s1,
            hdr.format("STEP 2: production hybrid run from the step-1 WAVECAR") + s2)


SLURM_HEADER = """#!/usr/bin/env bash
#SBATCH --job-name=scf_rescue
#SBATCH --account=<your_account>          # >>> EDIT
#SBATCH --partition=<partition>           # >>> EDIT
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=128                      # >>> EDIT: cores per node
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
module load vasp                          # >>> EDIT: the VASP module of your cluster
export OMP_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
"""


def job_script(exe: str, template: str | None = None) -> str:
    if template:
        head = re.split(r"^\s*(mpirun|srun|mpiexec)\b", template, maxsplit=1, flags=re.M)[0]
    else:
        head = SLURM_HEADER
    return head.rstrip("\n") + f"""

set -e
# ---- step 1: semilocal pre-convergence (only the WAVECAR is kept)
cp INCAR.step1 INCAR
mpirun -np "$SLURM_NTASKS" {exe}
mkdir -p step1 && cp OUTCAR OSZICAR INCAR step1/
grep "aborting loop" OUTCAR | tail -n 1 | grep -q "EDIFF is reached" \\
  || {{ echo "step 1 did not converge: inspect step1/OSZICAR"; exit 1; }}
# ---- step 2: production hybrid run from the step-1 orbitals (ALGO = Damped)
cp INCAR.step2 INCAR
mpirun -np "$SLURM_NTASKS" {exe}
grep "aborting loop" OUTCAR | tail -n 1 | grep -q "EDIFF is reached" \\
  && echo "step 2 converged" || echo "step 2 NOT converged: try TIME = 0.2 or ALGO = All"
echo "finished: $(date)"
"""


def cmd_prepare(a):
    prod = Path(a.incar).read_text(errors="replace")
    if (get_tag(prod, "LHFCALC") or "").upper().strip(".") not in ("TRUE", "T"):
        print("note: the production INCAR has no LHFCALC = .TRUE.; the recipe is meant for hybrid functionals")
    if a.nbands:
        nb = a.nbands
    elif a.outcar:
        nb = nbands_from_outcar(a.outcar)
    elif get_tag(prod, "NBANDS"):
        nb = int(get_tag(prod, "NBANDS"))
    else:
        print("NBANDS unknown: give --nbands or --outcar (of the failed run)", file=sys.stderr)
        return 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s1, s2 = step_incars(prod, nb)
    for name, text in (("INCAR.step1", s1), ("INCAR.step2", s2)):     # VASP reads plain ASCII
        (out / name).write_text(text, encoding="ascii", errors="replace", newline="\n")
    template = Path(a.job_template).read_text(errors="replace") if a.job_template else None
    (out / "job_two_step.sh").write_text(job_script(a.exe, template), encoding="utf-8", newline="\n")
    print(f"NBANDS = {nb}\nwrote {out / 'INCAR.step1'}, {out / 'INCAR.step2'}, {out / 'job_two_step.sh'}")
    print("copy POSCAR, KPOINTS and POTCAR of the failed run next to them, then: sbatch job_two_step.sh")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diagnose", help="classify the SCF behaviour of every ionic step")
    d.add_argument("oszicar", nargs="?", default="OSZICAR")
    d.add_argument("--ediff", type=float, default=1e-5, help="EDIFF of the run (default 1e-5)")
    d.add_argument("--nelm", type=int, help="NELM of the run (an SCF that used all NELM steps is not converged)")
    d.add_argument("--plot", help="save log10|dE| vs iteration (PNG/PDF/SVG)")
    p = sub.add_parser("prepare", help="write the two-step INCARs and job script")
    p.add_argument("incar", help="production INCAR of the failing run")
    p.add_argument("-o", "--out", default="rescue", help="output folder (default ./rescue)")
    p.add_argument("--nbands", type=int, help="NBANDS for both steps")
    p.add_argument("--outcar", help="take NBANDS from this OUTCAR (of the failed run)")
    p.add_argument("--exe", default="vasp_std", help="VASP binary: vasp_std, vasp_gam or vasp_ncl (SOC)")
    p.add_argument("--job-template", help="existing Slurm script whose header (everything before mpirun/srun) is reused")
    a = ap.parse_args(argv)
    return cmd_diagnose(a) if a.cmd == "diagnose" else cmd_prepare(a)


if __name__ == "__main__":
    raise SystemExit(main())
