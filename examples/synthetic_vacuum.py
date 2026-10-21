"""
synthetic_vacuum.py — LOCPOTs of a toy slab whose vacuum potential has a KNOWN defect.

Why this exists
---------------
In a charge-free vacuum the electrostatic potential along the surface normal is a straight
line: a constant, plus a ramp if the slab carries an uncorrected dipole, plus a step where a
dipole correction sits. ``Vacuum_Potential_Check/vacuum_check.py`` fits that line and looks for
what is left, in particular a long-wavelength cosine across the whole cell (the signature of an
unconverged G -> 0 component of the Hartree potential). The examples written here contain each
of those features with a known size, so the tool can be checked against the truth.

The slab is the toy slab of ``synthetic_vasp.py`` (1 Pb, 4 I, 2 small cations, c = 30 Å,
~19 Å of vacuum). Its potential decays to the vacuum value within ~1.5 Å of the outermost atom,
so beyond the default 3 Å margin only the injected features remain (plus 0.5 meV of noise).
The numbers are illustrative only; they are NOT the result of a calculation.

    vacuum_clean          flat vacuum                                     cos amplitude 0
    vacuum_residual       + 50 meV cosine of wavelength c (4-block LOCPOT) cos amplitude 50 meV
    vacuum_ramp           + uncorrected slab dipole                       ramp +10 meV/Å
    vacuum_dipole_step    + dipole correction (IDIPOL = 3, LDIPOL)        step 0.30 eV, flat

Usage
-----
    python synthetic_vacuum.py                 # writes the four folders next to this file
    python synthetic_vacuum.py --out my_dir    # somewhere else

    from synthetic_vacuum import make_vacuum_example
    truth = make_vacuum_example("run_dir", cos_amp=0.05)   # returns the ground-truth dictionary
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from synthetic_vasp import toy_slab, write_lines, write_locpot

EXAMPLES = {
    "vacuum_clean": dict(),
    "vacuum_residual": dict(cos_amp=0.050, nblocks=4,
                            incar=["SYSTEM  = toy slab, SCF stopped early (synthetic)",
                                   "ENCUT   = 400", "ISMEAR  = 0 ; SIGMA = 0.02",
                                   "LSORBIT = .TRUE.",
                                   "ALGO    = Normal", "EDIFF   = 1E-3     ! loose", "NELM    = 12",
                                   "LVHAR   = .TRUE."]),
    "vacuum_ramp": dict(ramp=0.010),
    "vacuum_dipole_step": dict(step=0.30,
                               incar=["SYSTEM  = toy slab, dipole-corrected (synthetic)",
                                      "ENCUT   = 400", "ISMEAR  = 0 ; SIGMA = 0.02",
                                      "ALGO    = Normal", "EDIFF   = 1E-6", "NELM    = 100",
                                      "IDIPOL  = 3", "LDIPOL  = .TRUE.", "LVHAR   = .TRUE."]),
}

DEFAULT_INCAR = ["SYSTEM  = toy slab (synthetic)", "ENCUT   = 400", "ISMEAR  = 0 ; SIGMA = 0.02",
                 "ALGO    = Normal", "EDIFF   = 1E-6", "NELM    = 100", "LVHAR   = .TRUE."]


def vacuum_potential(cell, frac, shape=(4, 4, 200), v_vac=1.5, depth=8.0, cos_amp=0.0, ramp=0.0,
                     step=0.0, noise=0.0005, seed=42):
    """Potential grid (eV) of the toy slab with controlled vacuum features.

    u = distance (Å) along c from the outermost atom of the top surface, W = vacuum gap,
    L = cell length. In the vacuum (0 < u < W) the planar average is exactly

        v_vac + ramp (u - W/2) + (step/2) sign(W/2 - u) + cos_amp cos(2 pi (u - W/2) / L)

    The ramp and the step are closed through the slab by a linear segment, so the profile is
    periodic. ``step`` > 0 makes the side above the top surface the higher one."""
    nx, ny, nz = shape
    L = float(np.linalg.norm(cell[2]))
    zf = np.arange(nz) / nz
    fz = np.sort(frac[:, 2] % 1.0)
    gaps = np.diff(np.r_[fz, fz[0] + 1.0])
    k = int(np.argmax(gaps))
    start, W = fz[k], gaps[k] * L
    u = ((zf - start) % 1.0) * L
    half = (L - W) / 2                                        # half thickness of the slab (atom to atom)
    v = (u - (W + L) / 2 + L / 2) % L - L / 2                 # distance from the slab centre, wraps mid-vacuum
    erf = np.vectorize(math.erf)
    box = 0.5 * (1.0 - erf((np.abs(v) - half) / 0.6))         # 1 inside the slab, 0 in the vacuum
    prof = v_vac - depth * box
    for f in frac:
        dz = ((zf - f[2] + 0.5) % 1.0 - 0.5) * L
        prof -= 1.5 * np.exp(-dz ** 2 / 0.3)
    in_vac = u < W
    lin = np.where(in_vac, ramp * (u - W / 2), ramp * W / 2 - ramp * W / (L - W) * (u - W))
    stp = np.where(in_vac, 0.5 * step * np.sign(W / 2 - u), -0.5 * step + step * (u - W) / (L - W))
    prof += lin + stp + cos_amp * np.cos(2 * np.pi * (u - W / 2) / L)
    x, y = np.meshgrid(np.arange(nx) / nx, np.arange(ny) / ny, indexing="ij")
    grid = prof[None, None, :] + 0.8 * (np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y))[:, :, None] * box
    grid += np.random.default_rng(seed).normal(0.0, noise, grid.shape)
    return grid, dict(start=float(start), vac_gap=float(W), L=L)


def make_vacuum_example(folder, *, cos_amp=0.0, ramp=0.0, step=0.0, v_vac=1.5, nblocks=1,
                        incar=None, shape=(4, 4, 200)):
    """Write LOCPOT and INCAR of the toy slab into ``folder``; return the ground truth.

    ``cos_amp`` (eV), ``ramp`` (eV/Å, dV/dz in the vacuum), ``step`` (eV, upper minus lower side)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    symbols, cell, frac = toy_slab()
    grid, geo = vacuum_potential(cell, frac, shape=shape, v_vac=v_vac, cos_amp=cos_amp, ramp=ramp, step=step)
    write_locpot(folder / "LOCPOT", symbols, cell, frac, grid, nblocks=nblocks)
    write_lines(folder / "INCAR", incar or DEFAULT_INCAR)
    return dict(cos_amp_eV=cos_amp, ramp_eV_per_A=ramp, step_eV=step, v_vac=v_vac,
                vac_gap_A=geo["vac_gap"], L=geo["L"], nblocks=nblocks, natoms=len(symbols))


def make_vacuum_examples(out):
    """Write the four example folders under ``out``; return {folder name: truth}."""
    return {name: make_vacuum_example(Path(out) / name, **kw) for name, kw in EXAMPLES.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent), help="parent folder (default: next to this file)")
    a = ap.parse_args()
    for name, t in make_vacuum_examples(a.out).items():
        print(f"{name}: cos amplitude {1000 * t['cos_amp_eV']:.0f} meV, ramp {1000 * t['ramp_eV_per_A']:.0f} meV/Å, "
              f"step {t['step_eV']:.2f} eV, vacuum gap {t['vac_gap_A']:.2f} Å")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
