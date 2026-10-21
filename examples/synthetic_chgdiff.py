"""
synthetic_chgdiff.py — three small CHGCARs (AB, A, B) whose charge-density difference is KNOWN.

The densities are sums of Gaussian charge blobs on a coarse grid (8 x 8 x 80, cell 5 x 5 x 20 Å):

    fragment A : "Pb" (4 e) at z = 5.0 Å and "I" (7 e) at z = 8.0 Å
    fragment B : "N"  (5 e) at z = 11.0 Å and "H" (1 e) at z = 12.0 Å
    rho_AB     = rho_A + rho_B + drho_known

drho_known moves q = 0.25 electrons from a lobe centred at z = 10.4 Å (on the B side) to a lobe
centred at z = 8.6 Å (on the A side). Both lobes have a compact profile along z (cos^2, half-width
0.8 Å), so no density leaks across the A|B mid-plane at z = 9.5 Å (midway between I and N):

    integral of drho          = 0
    dQ at z = 9.5 Å           = +0.25 e  (electrons crossed towards -z, from B to A)
    dipole of drho            = q * (10.4 - 8.6) = +0.45 e·Å per cell

The files use the real CHGCAR layout (values = rho * V_cell, five per line). The A file is written
like a spin-polarized CHGCAR (augmentation occupancies + a magnetization block), so the reader's
skipping logic is exercised (``write_chgcar(..., magnetization_blocks=3)`` gives the non-collinear
layout used in the tests).
A fourth file, B_other_grid/CHGCAR, holds fragment B on a different FFT grid: it must be rejected.
The numbers are illustrative only; they are NOT the result of a calculation.

Usage
-----
    python synthetic_chgdiff.py                 # writes examples/charge_difference/
    python synthetic_chgdiff.py --out my_dir

    from synthetic_chgdiff import make_chgdiff_example
    truth = make_chgdiff_example("charge_difference")
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_vasp import write_poscar  # noqa: E402

CELL = np.diag([5.0, 5.0, 20.0])
SHAPE = (8, 8, 80)
ATOMS_A = [("Pb", (0.0, 0.0, 5.0), 4.0, 1.0), ("I", (2.5, 2.5, 8.0), 7.0, 0.9)]     # symbol, Å, electrons, sigma
ATOMS_B = [("N", (0.0, 0.0, 11.0), 5.0, 0.7), ("H", (0.0, 0.0, 12.0), 1.0, 0.5)]
Q_MOVED, Z_GAIN, Z_LOSS, HALF_WIDTH = 0.25, 8.6, 10.4, 0.8


def _grid_points(cell, shape):
    f = np.stack(np.meshgrid(*[np.arange(n) / n for n in shape], indexing="ij"), axis=-1)
    return f                                                     # (nx, ny, nz, 3) fractional


def _mic(frac_pts, center, cell):
    d = frac_pts - (np.asarray(center) @ np.linalg.inv(cell))
    d -= np.round(d)
    return d @ cell                                              # cartesian displacement, minimum image


def gaussian_blob(center, charge, sigma, cell=CELL, shape=SHAPE):
    """rho * V_cell on the grid for a 3D Gaussian of ``charge`` electrons (discretely normalized)."""
    r = _mic(_grid_points(cell, shape), center, cell)
    g = np.exp(-0.5 * (r ** 2).sum(-1) / sigma ** 2)
    return charge * g / g.mean()                                 # mean(rho * V) = integral of rho


def lobe(center, charge, sigma_xy=0.9, half_width=HALF_WIDTH, cell=CELL, shape=SHAPE):
    """Gaussian in plane x cos^2 along z with compact support |z - zc| < half_width."""
    r = _mic(_grid_points(cell, shape), center, cell)
    gz = np.where(np.abs(r[..., 2]) < half_width, np.cos(0.5 * np.pi * r[..., 2] / half_width) ** 2, 0.0)
    g = np.exp(-0.5 * (r[..., 0] ** 2 + r[..., 1] ** 2) / sigma_xy ** 2) * gz
    return charge * g / g.mean()


def write_chgcar(path, symbols, cell, frac, grid, *, augmentation=True, magnetization_blocks=0,
                 comment="synthetic CHGCAR"):
    """CHGCAR layout: POSCAR header, blank line, grid size, rho*V values (x fastest, 5 per line),
    then (optionally) augmentation occupancies and zero magnetization blocks, as VASP writes them."""
    names = list(dict.fromkeys(symbols))
    counts = [symbols.count(n) for n in names]
    head = [comment, "   1.00000000000000"]
    head += ["  " + "  ".join(f"{x:12.6f}" for x in v) for v in cell]
    head += ["   " + "   ".join(names), "   " + "   ".join(str(n) for n in counts), "Direct"]
    head += ["  " + "  ".join(f"{x:.6f}" for x in f) for f in frac]
    nx, ny, nz = grid.shape

    def block(vals):
        rows = [" ".join(f"{v:18.11E}" for v in vals[i:i + 5]) for i in range(0, vals.size, 5)]
        out = [f"  {nx:4d}  {ny:4d}  {nz:4d}"] + rows
        if augmentation:
            for i in range(len(symbols)):
                occ = np.linspace(0.1, 0.5, 8) / (i + 1)
                out += [f"augmentation occupancies {i + 1:3d}  {occ.size:3d}",
                        " ".join(f"{v:.7E}" for v in occ[:5]), " ".join(f"{v:.7E}" for v in occ[5:])]
        return out

    body = [""] + block(grid.T.ravel())
    for _ in range(magnetization_blocks):
        body += [" ".join(f"{0.0:.7E}" for _ in symbols)] + block(np.zeros(grid.size))
    Path(path).write_text("\n".join(head + body) + "\n", newline="\n")


def _structure(atoms):
    symbols = [a[0] for a in atoms]
    frac = (np.array([a[1] for a in atoms]) @ np.linalg.inv(CELL)) % 1.0
    return symbols, frac


def make_chgdiff_example(folder) -> dict:
    """Write AB/, A/, B/ and B_other_grid/ (CHGCAR + POSCAR each) and return the known answer."""
    folder = Path(folder)
    rho_a = sum(gaussian_blob(p, q, s) for _, p, q, s in ATOMS_A)
    rho_b = sum(gaussian_blob(p, q, s) for _, p, q, s in ATOMS_B)
    drho = lobe((2.5, 2.5, Z_GAIN), Q_MOVED) - lobe((0.0, 0.0, Z_LOSS), Q_MOVED)
    rho_ab = rho_a + rho_b + drho
    assert rho_ab.min() > 0, "the toy density must stay positive"
    sym_a, fr_a = _structure(ATOMS_A)
    sym_b, fr_b = _structure(ATOMS_B)
    sym_ab, fr_ab = _structure(ATOMS_A + ATOMS_B)
    rho_b_other = sum(gaussian_blob(p, q, s, shape=(6, 6, 60)) for _, p, q, s in ATOMS_B)
    for sub, sym, fr, grid, kw in (
            ("AB", sym_ab, fr_ab, rho_ab, {}),
            ("A", sym_a, fr_a, rho_a, {"magnetization_blocks": 1}),          # spin-polarized layout
            ("B", sym_b, fr_b, rho_b, {}),                                    # plain (non-spin) layout
            ("B_other_grid", sym_b, fr_b, rho_b_other, {})):
        (folder / sub).mkdir(parents=True, exist_ok=True)
        write_chgcar(folder / sub / "CHGCAR", sym, CELL, fr, grid, comment=f"synthetic CHGCAR, fragment {sub}", **kw)
        write_poscar(folder / sub / "POSCAR", sym, CELL, fr, comment=f"synthetic structure, fragment {sub}")
    return dict(q_moved=Q_MOVED, plane_z=9.5, integral=0.0, dipole_eA=Q_MOVED * (Z_LOSS - Z_GAIN),
                electrons_AB=17.0, electrons_A=11.0, electrons_B=6.0, shape=SHAPE, length=20.0,
                drho=drho / abs(np.linalg.det(CELL)))                 # e/Å^3 on the grid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "charge_difference"),
                    help="output folder (default: examples/charge_difference)")
    a = ap.parse_args()
    t = make_chgdiff_example(a.out)
    print(f"{a.out}: q moved {t['q_moved']:.3f} e across z = {t['plane_z']:.2f} Å, dipole {t['dipole_eA']:.3f} e·Å")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
