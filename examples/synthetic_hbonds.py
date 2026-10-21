"""
synthetic_hbonds.py — toy structures with KNOWN hydrogen bonds, for Hydrogen_Bonds/hbonds.py.

The toy cell holds isolated fragments (it is not a crystal): two methylammonium-like
cations (NH3 donors), one water molecule and one methanol-like molecule (O-H donors), iodide, bromide and chloride
acceptors, and a few Pb atoms that make some iodides "terminal" (one Pb neighbour) and
one "bridging" (two Pb neighbours). Every acceptor is placed from its hydrogen at a
chosen H...A distance and D-H...A angle, so the answer is known by construction:

    group  donor  contact            H...A (Å)  angle (°)  acceptor tag
    1      NH3    H...I              2.70       170        terminal
    1      NH3    H...I              2.95       160        bridging
    1      NH3    H...I              3.40       150        terminal
    2      NH3    H...I              3.10       165        terminal  (exists only across the a boundary)
    2      NH3    H...Br             3.30       125        free
    3      OH2    H...Cl             2.30       175        free
    4      OH     H...I              2.90        95        terminal  (short, but fails the 120° angle)
    -      C-H    H...I              2.60       180        free      (C-H: not a donor, never counted)

The triclinic variant puts the same fragments (same Cartesian coordinates) in a skewed cell
that keeps the a vector, so every distance and angle is unchanged while the atoms wrap
differently; it has the same answer. The generator checks its own design by brute force
(every image of a 5x5x5 supercell) and refuses to write a structure with an unplanned contact.

Usage
-----
    python synthetic_hbonds.py                     # writes examples/hbond_toy/
    from synthetic_hbonds import make_hbond_example
    truth = make_hbond_example("hbond_toy")        # returns the ground-truth dictionary

The numbers are illustrative only; they are NOT the result of a calculation.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_vasp import write_poscar  # noqa: E402

NH, CN, CH, OH, PBX = 1.03, 1.47, 1.09, 0.97, 3.20       # bond lengths (Å)
TETRA = np.radians(180.0 - 109.47)                        # angle between the axis and an X-H bond
CUTOFFS = (2.8, 3.0, 3.2, 3.5)
ANGLE_MIN = 120.0
SOFT = (2.6, 3.5)
L = 16.0                                                  # orthorhombic cell edge (Å)
CELL_ORTHO = np.diag([L, L, L])
CELL_TRICLINIC = np.array([[L, 0.0, 0.0], [7.0, 15.0, 0.0], [-5.0, 4.0, 16.0]])

# (group, H number, acceptor element, H...A, angle, number of Pb neighbours, rotation of the side direction)
DESIGN = [
    (1, 0, "I", 2.70, 170.0, 1),
    (1, 1, "I", 2.95, 160.0, 2),
    (1, 2, "I", 3.40, 150.0, 1),
    (2, 0, "I", 3.10, 165.0, 1),
    (2, 2, "Br", 3.30, 125.0, 0),
    (3, 0, "Cl", 2.30, 175.0, 0),
    (4, 0, "I", 2.90, 95.0, 1),
]
RMAX_CHECK = 3.6                                          # no unplanned contact up to this H...A distance
TRAP = dict(element="I", d=2.60, h=6)                          # in front of a C-H of group 1


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def ammonium(n_pos, axis, ref):
    """CH3NH3+-like cation: N at ``n_pos``, the three N-H around ``axis``, the C opposite.

    ``ref`` fixes the azimuth of the first N-H. Returns (atoms, N-H unit vectors)."""
    n_pos = np.asarray(n_pos, float)
    z = _unit(axis)
    x = _unit(np.asarray(ref, float) - np.dot(ref, z) * z)
    y = np.cross(z, x)
    atoms, us = [("N", n_pos)], []
    for k in range(3):
        phi = 2 * np.pi * k / 3
        u = np.cos(TETRA) * z + np.sin(TETRA) * (np.cos(phi) * x + np.sin(phi) * y)
        us.append(u)
        atoms.append(("H", n_pos + NH * u))
    c = n_pos - CN * z
    atoms.append(("C", c))
    for k in range(3):                                    # staggered methyl
        phi = 2 * np.pi * k / 3 + np.pi / 3
        u = -np.cos(TETRA) * z + np.sin(TETRA) * (np.cos(phi) * x + np.sin(phi) * y)
        atoms.append(("H", c + CH * u))
    return atoms, us


def water(o_pos, bisector, ref):
    o_pos = np.asarray(o_pos, float)
    z = _unit(bisector)
    x = _unit(np.asarray(ref, float) - np.dot(ref, z) * z)
    half = np.radians(104.5 / 2)
    us = [np.cos(half) * z + np.sin(half) * x, np.cos(half) * z - np.sin(half) * x]
    return [("O", o_pos)] + [("H", o_pos + OH * u) for u in us], us


def methanol(o_pos, axis, ref):
    """CH3OH-like molecule: O at ``o_pos``, C along -axis, the O-H tilted towards ``ref``."""
    o_pos = np.asarray(o_pos, float)
    z = _unit(axis)
    x = _unit(np.asarray(ref, float) - np.dot(ref, z) * z)
    y = np.cross(z, x)
    u = np.cos(TETRA) * z + np.sin(TETRA) * x
    c = o_pos - 1.43 * z
    atoms = [("O", o_pos), ("H", o_pos + OH * u), ("C", c)]
    for k in range(3):
        phi = 2 * np.pi * k / 3 + np.pi / 3
        atoms.append(("H", c + CH * (-np.cos(TETRA) * z + np.sin(TETRA) * (np.cos(phi) * x + np.sin(phi) * y))))
    return atoms, [u]


def place_acceptor(h, u, d, angle, avoid):
    """Acceptor at distance ``d`` from H with D-H...A = ``angle``; the side direction is chosen
    (on a 5° grid around the N-H axis) to keep the acceptor as far as possible from ``avoid``."""
    p0 = _unit(np.cross(u, [0.3, 0.5, 0.8]))
    q0 = np.cross(u, p0)
    best, best_gap = None, -1.0
    for psi in np.radians(np.arange(0, 360, 5)):
        p = np.cos(psi) * p0 + np.sin(psi) * q0
        a = h + d * (-np.cos(np.radians(angle)) * u + np.sin(np.radians(angle)) * p)
        gap = min(np.linalg.norm(a - x) for x in avoid) if avoid else 9.0
        if gap > best_gap + 1e-9:
            best, best_gap = a, gap
    return best


def build(cell=CELL_ORTHO):
    """Symbols, cell, fractional coordinates and the designed contacts (Cartesian design)."""
    frags = {
        1: ammonium([5.0, 5.0, 9.0], [0, 0, 1], [1, 0, 0]),
        2: ammonium([1.4, 12.5, 13.0], [0, 0, 1], [-1, 0, 0]),  # first N-H points to -x: across the boundary
        3: water([11.0, 3.5, 4.0], [0, 1, 0], [1, 0, 0]),
        4: methanol([10.0, 9.0, 12.0], [0, 0, 1], [1, 1, 0]),
    }
    atoms = [a for g in (1, 2, 3, 4) for a in frags[g][0]]
    contacts = []
    for g, k, el, d, ang, npb in DESIGN:
        frag, us = frags[g]
        donor = frag[0][1]
        hs = [p for s, p in frag if s == "H"]
        h = hs[k]
        a = place_acceptor(h, us[k], d, ang, [p for _, p in atoms])
        atoms.append((el, a))
        out = _unit(a - h)                                 # metals continue away from the H
        if npb == 1:
            atoms.append(("Pb", a + PBX * out))
        elif npb == 2:
            side = _unit(np.cross(out, [0.1, 0.2, 1.0]))
            atoms += [("Pb", a + PBX * side), ("Pb", a - PBX * side)]
        dh = NH if frag[0][0] == "N" else OH
        dDA = float(np.sqrt(d * d + dh * dh - 2 * d * dh * np.cos(np.radians(ang))))
        contacts.append(dict(group=g, donor=frag[0][0], acceptor=el, d_HA=d, angle=ang, d_DA=dDA,
                             tag=("free", "terminal", "bridging")[npb]))
    # trap: an iodide 2.60 Å in front of a methyl C-H of group 1 (C-H is not a donor)
    frag = frags[1][0]
    c = frag[4][1]
    hme = frag[TRAP["h"]][1]
    atoms.append((TRAP["element"], hme + TRAP["d"] * _unit(hme - c)))
    order = {"Pb": 0, "I": 1, "Br": 2, "Cl": 3, "C": 4, "N": 5, "O": 6, "H": 7}
    atoms.sort(key=lambda t: order[t[0]])
    symbols = [s for s, _ in atoms]
    pos = np.array([p for _, p in atoms])
    frac = (pos @ np.linalg.inv(cell)) % 1.0
    return symbols, np.asarray(cell, float), frac, contacts


def brute_force_contacts(symbols, cell, frac, rmax=RMAX_CHECK, donors=("N", "O"),
                         acceptors=("I", "Br", "Cl", "O", "N", "F"), dh_max=1.25):
    """Independent check: every donor-H...acceptor distance <= rmax in a 5x5x5 block of images."""
    sym = np.array(symbols)
    cart = frac @ cell
    shifts = np.array(list(itertools.product(range(-2, 3), repeat=3))) @ cell
    out = []
    heavy = np.where(sym != "H")[0]
    for h in np.where(sym == "H")[0]:
        dh = [(np.min(np.linalg.norm(cart[j] + shifts - cart[h], axis=1)), j) for j in heavy]
        dmin, own = min(dh)
        if dmin > dh_max or sym[own] not in donors:
            continue
        for a in np.where(np.isin(sym, acceptors))[0]:
            for d in np.linalg.norm(cart[a] + shifts - cart[h], axis=1):
                if 1e-6 < d <= rmax and not (a == own and d < dh_max):
                    out.append((int(h), int(a), float(d)))
    return out


def metal_neighbours(symbols, cell, frac, metal="Pb", rcut=3.8):
    sym = np.array(symbols)
    cart = frac @ cell
    shifts = np.array(list(itertools.product(range(-2, 3), repeat=3))) @ cell
    return {int(a): int(sum((np.linalg.norm(cart[m] + shifts - cart[a], axis=1) <= rcut).sum()
                            for m in np.where(sym == metal)[0]))
            for a in np.where(np.isin(sym, ["I", "Br", "Cl"]))[0]}


def check(symbols, cell, frac, contacts):
    """Refuse a structure whose contacts or metal neighbours differ from the design."""
    found = sorted(round(d, 4) for _, _, d in brute_force_contacts(symbols, cell, frac))
    planned = sorted(round(c["d_HA"], 4) for c in contacts)
    if found != planned:
        raise RuntimeError(f"unplanned contacts: found {found}, planned {planned}")
    tags = sorted(metal_neighbours(symbols, cell, frac).values())
    want = sorted([{"free": 0, "terminal": 1, "bridging": 2}[c["tag"]] for c in contacts] + [0])
    if tags != want:
        raise RuntimeError(f"metal neighbours {tags}, planned {want}")
    cart = frac @ cell
    shifts = np.array(list(itertools.product(range(-1, 2), repeat=3))) @ cell
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            d = np.min(np.linalg.norm(cart[j] + shifts - cart[i], axis=1))
            if d < 0.9:
                raise RuntimeError(f"atoms {i + 1} and {j + 1} overlap ({d:.2f} Å)")


def truth_from(contacts):
    t = dict(contacts=contacts, cutoffs=CUTOFFS, angle_min=ANGLE_MIN, n_groups=4, n_donor_H=9, n_other_H=9)
    hb = [c for c in contacts if c["angle"] >= ANGLE_MIN]
    t["n_HA"] = {c: sum(x["d_HA"] <= c for x in contacts) for c in CUTOFFS}
    t["n_hb"] = {c: sum(x["d_HA"] <= c for x in hb) for c in CUTOFFS}
    t["n_hb_tag"] = {tag: {c: sum(x["d_HA"] <= c and x["tag"] == tag for x in hb) for c in CUTOFFS}
                     for tag in ("free", "terminal", "bridging")}
    t["n_hb_group"] = {g: sum(x["group"] == g for x in hb) for g in (1, 2, 3, 4)}       # at 3.5 Å
    t["d_HA_min_group"] = {g: min(x["d_HA"] for x in contacts if x["group"] == g) for g in (1, 2, 3, 4)}
    t["angle_at_min_group"] = {g: min((x for x in contacts if x["group"] == g), key=lambda x: x["d_HA"])["angle"]
                               for g in (1, 2, 3, 4)}
    t["d_HA_min_hb_group"] = {g: min((x["d_HA"] for x in hb if x["group"] == g), default=None) for g in (1, 2, 3, 4)}
    lo, hi = SOFT
    t["soft"] = sum(min(1.0, max(0.0, (hi - x["d_HA"]) / (hi - lo))) for x in hb)
    t["n_rejected_angle"] = sum(1 for x in contacts if x["angle"] < ANGLE_MIN)
    t["boundary_contact"] = dict(group=2, d_HA=3.10, angle=165.0)
    t["rejected_contact"] = dict(group=4, d_HA=2.90, angle=95.0)
    t["trap_CH_distance"] = TRAP["d"]
    return t


def make_hbond_example(folder) -> dict:
    """Write POSCAR (orthorhombic) and POSCAR_triclinic into ``folder``; return the known answer."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    truth = None
    for name, cell in (("POSCAR", CELL_ORTHO), ("POSCAR_triclinic", CELL_TRICLINIC)):
        symbols, cell, frac, contacts = build(cell)
        check(symbols, cell, frac, contacts)
        write_poscar(folder / name, symbols, cell, frac,
                     comment=f"synthetic hydrogen-bond toy cell ({'orthorhombic' if name == 'POSCAR' else 'triclinic'})")
        truth = truth or truth_from(contacts)
    truth["files"] = ["POSCAR", "POSCAR_triclinic"]
    truth["n_atoms"] = len(symbols)
    return truth


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "hbond_toy"))
    a = ap.parse_args(argv)
    t = make_hbond_example(a.out)
    print(f"{a.out}: {t['n_atoms']} atoms; hydrogen bonds " +
          "  ".join(f"<={c:g}: {n}" for c, n in t["n_hb"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
