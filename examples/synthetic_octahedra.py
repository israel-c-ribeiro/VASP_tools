"""
synthetic_octahedra.py — corner-sharing octahedral sheets with a KNOWN distortion.

Why this exists
---------------
The tutorial of Octahedral_Distortion/octahedra.py and its tests need structures whose octahedral
descriptors are known exactly, before any program has measured them. This file builds a toy
single-layer (n = 1) sheet of corner-sharing BX6 octahedra (B = Pb, X = I), 2 x 2 octahedra per
cell, with small CH3NH3-like cations in the hollow sites above and below the sheet (they are not
ligands and must be ignored by the tool). The numbers are illustrative only.

Three distortions can be switched on, each with a closed-form answer:

* in-plane rotation phi: the octahedra rotate about the sheet normal by +phi / -phi in a
  checkerboard pattern (the only pattern that keeps the corners shared). Every octahedron stays
  rigid, the bridging X moves off the B-B line by (L/2) tan(phi), and the bridging angle is
  B-X-B = 180 - 2 phi;
* axial elongation: four equatorial bonds of length d_eq and two axial bonds of length d_ax give
  the Baur index  Delta = (4/9) |d_ax - d_eq| / <d>,  with <d> = (4 d_eq + 2 d_ax) / 6;
* axial tilt theta: the axial X-B-X axis (kept straight) leans by theta from the normal, towards
  one equatorial ligand. Of the 12 cis angles, two become 90 - theta, two 90 + theta, the rest
  stay 90, so the bond-angle variance is  sigma^2 = 4 theta^2 / 11.

The octahedron is then a (tilted) square bipyramid of volume V = (4/3) d_eq^2 d_ax cos(theta), and
the quadratic elongation is  <lambda> = (4 d_eq^2 + 2 d_ax^2) / (6 l0^2),  l0 = (3V / 4)^(1/3).

Usage
-----
    python synthetic_octahedra.py                 # writes the example folders next to this file
    python synthetic_octahedra.py --out my_dir    # somewhere else

    from synthetic_octahedra import make_octahedra_example
    truth = make_octahedra_example("folder", rotation=12, tilt=8, d_eq=3.20, d_ax=3.30)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_vasp import cation, write_poscar  # noqa: E402

# the four example folders written by main(): name -> keyword arguments of make_octahedra_example
EXAMPLES = {
    "octahedra_ideal": dict(d_eq=3.15, d_ax=3.15, rotation=0.0, tilt=0.0),
    "octahedra_distorted": dict(d_eq=3.20, d_ax=3.30, rotation=12.0, tilt=8.0),
    "octahedra_skewed": dict(d_eq=3.20, d_ax=3.30, rotation=12.0, tilt=8.0, skew=True),
    "octahedra_slab": dict(d_eq=3.20, d_ax=3.30, rotation=12.0, tilt=8.0, c=30.0, layer_z=1.0),
}


def octahedral_sheet(d_eq=3.15, d_ax=3.15, rotation=0.0, tilt=0.0, c=15.0, layer_z=0.5,
                     skew=False, cations=True):
    """Symbols, cell (3x3, Å) and fractional coordinates of a 2 x 2 corner-sharing sheet.

    ``c`` = repeat along the normal (bulk: 15 Å, no vacuum worth the name; slab: 30 Å, about 19 Å of
    vacuum); ``layer_z`` = height of the B plane (close to 0 so that atoms wrap across the cell
    boundary); ``skew`` = describe the same crystal with a triclinic cell (b' = b + 2a, and c leaning
    by 0.3 a + 0.2 b), which only changes the stacking of the sheets, not their geometry."""
    phi, th = np.radians(rotation), np.radians(tilt)
    L = 2.0 * d_eq * np.cos(phi)                         # B-B distance along the sheet
    delta = 0.5 * L * np.tan(phi)                        # off-line shift of the bridging X
    cell = np.diag([2 * L, 2 * L, c])
    atoms = []
    for i in range(2):
        for j in range(2):
            s = 1.0 if (i + j) % 2 == 0 else -1.0        # checkerboard of +phi / -phi rotations
            B = np.array([i * L, j * L, layer_z])
            atoms.append(("Pb", B))
            atoms.append(("I", B + [0.5 * L, s * delta, 0.0]))          # bridging X towards +a
            atoms.append(("I", B + [-s * delta, 0.5 * L, 0.0]))         # bridging X towards +b
            u = np.array([np.sin(th) * np.cos(s * phi), np.sin(th) * np.sin(s * phi), np.cos(th)])
            atoms.append(("I", B + d_ax * u))                            # terminal (axial) X, up
            atoms.append(("I", B - d_ax * u))                            # terminal (axial) X, down
            if cations:                                   # one cation above and one below each hollow
                hollow = B + [0.5 * L, 0.5 * L, 0.0]
                atoms += [(e, np.array(p)) for e, p in cation(hollow + [0, 0, 3.6], up=True)]
                atoms += [(e, np.array(p)) for e, p in cation(hollow - [0, 0, 3.6], up=False)]
    order = {"Pb": 0, "I": 1, "C": 2, "N": 3, "H": 4}
    atoms.sort(key=lambda t: order[t[0]])
    symbols = [e for e, _ in atoms]
    pos = np.array([p for _, p in atoms])
    if skew:
        a, b = cell[0], cell[1]
        cell = np.array([a, b + 2 * a, cell[2] + 0.3 * a + 0.2 * b])
    return symbols, cell, (pos @ np.linalg.inv(cell)) % 1.0


def known_answer(d_eq, d_ax, rotation, tilt):
    """Closed-form descriptors of the sheet (see the module docstring)."""
    th = np.radians(tilt)
    mean = (4 * d_eq + 2 * d_ax) / 6
    volume = 4.0 / 3.0 * d_eq ** 2 * d_ax * np.cos(th)
    l0 = (3 * volume / 4) ** (1 / 3)
    L = 2.0 * d_eq * np.cos(np.radians(rotation))
    return dict(n_centres=4, n_bridging=8, n_terminal=8, n_layers=1,
                d_eq=d_eq, d_ax=d_ax, bond_mean=mean,
                baur=4.0 / 9.0 * abs(d_ax - d_eq) / mean,
                volume=volume, lambda_=(4 * d_eq ** 2 + 2 * d_ax ** 2) / (6 * l0 ** 2),
                sigma2=4.0 * tilt ** 2 / 11.0, bxb=180.0 - 2.0 * rotation, tilt=float(tilt),
                area_per_centre=L ** 2)


def make_octahedra_example(folder, *, d_eq=3.15, d_ax=3.15, rotation=0.0, tilt=0.0, c=15.0,
                           layer_z=0.5, skew=False, name="POSCAR"):
    """Write ``folder/POSCAR`` with the sheet and return the ground-truth dictionary."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sym, cell, frac = octahedral_sheet(d_eq, d_ax, rotation, tilt, c, layer_z, skew)
    write_poscar(folder / name, sym, cell, frac,
                 comment=f"toy octahedral sheet (synthetic): d_eq {d_eq} d_ax {d_ax} "
                         f"rotation {rotation} tilt {tilt}")
    return known_answer(d_eq, d_ax, rotation, tilt)


def make_octahedra_examples(out):
    """Write every example folder of EXAMPLES under ``out``; return {folder name: truth}."""
    return {name: make_octahedra_example(Path(out) / name, **kw) for name, kw in EXAMPLES.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent), help="parent folder (default: next to this file)")
    a = ap.parse_args()
    for name, t in make_octahedra_examples(a.out).items():
        print(f"{name}: Baur {t['baur']:.5f}, lambda {t['lambda_']:.5f}, sigma2 {t['sigma2']:.2f} deg2, "
              f"B-X-B {t['bxb']:.1f} deg, tilt {t['tilt']:.1f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
