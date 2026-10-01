#!/usr/bin/env python3
"""
sd_tool.py — inspect, remove and set selective-dynamics flags in VASP POSCAR/CONTCAR files.

Why
---
Selective dynamics ("T T T" / "F F F" after each coordinate) decides which atoms VASP
may move. The flags travel silently: VASP copies them into CONTCAR, and every restart
or new calculation built from that CONTCAR inherits them. A relaxation can therefore
report "reached required accuracy" while part of the structure never moved — the forces
on fixed atoms are simply not part of the convergence criterion. Always look at the
flags before you start a new stage of a workflow.

Commands
--------
    report  show how many atoms are fixed, per element and per direction
            (with --outcar: residual forces on fixed and free atoms of the last ionic step)
    free    write a copy with every atom free (no "Selective dynamics" line)
    fix     write a copy with chosen atoms fixed: by height along an axis,
            by element, by index, or a combination (all selections are intersected)

Usage
-----
    python sd_tool.py report CONTCAR --outcar OUTCAR
    python sd_tool.py free CONTCAR -o POSCAR_free
    python sd_tool.py fix POSCAR -o POSCAR_fixed --below 0.30            # fractional c < 0.30
    python sd_tool.py fix POSCAR -o POSCAR_fixed --below-A 8.0 --elements Pb,I
    python sd_tool.py fix POSCAR -o POSCAR_fixed --indices 1-12,20 --directions z

Atom indices are 1-based, as in VESTA and in the OUTCAR force table.
Only the Python standard library and numpy are used.

Author: Israel C. Ribeiro — https://github.com/israel-c-ribeiro/VASP_tools (MIT licence)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

__version__ = "1.0.0"


class Poscar:
    """Minimal VASP 5 POSCAR container: comment, cell, species, fractional coords, flags."""

    def __init__(self, comment, cell, names, counts, frac, flags=None):
        self.comment, self.cell, self.names, self.counts = comment, cell, names, counts
        self.frac = frac
        self.flags = flags                                  # (N, 3) bool, True = free, or None

    @property
    def symbols(self):
        return [n for n, c in zip(self.names, self.counts) for _ in range(c)]

    @classmethod
    def read(cls, path):
        L = Path(path).read_text(errors="replace").splitlines()
        scale = float(L[1].split()[0])
        cell = np.array([[float(t) for t in L[i].split()[:3]] for i in (2, 3, 4)])
        if scale < 0:                                       # negative scale = target volume
            scale = (-scale / abs(np.linalg.det(cell))) ** (1 / 3)
        cell *= scale
        names = L[5].split()
        if names[0].isdigit():
            raise ValueError(f"{path}: VASP 4 format (no element line) is not supported")
        counts = [int(x) for x in L[6].split()]
        i, flags = 7, None
        sd = L[i].strip()[:1] in "Ss"
        if sd:
            i += 1
        direct = L[i].strip()[:1] in "Dd"
        rows = [L[j].split() for j in range(i + 1, i + 1 + sum(counts))]
        xyz = np.array([[float(t) for t in r[:3]] for r in rows])
        frac = xyz if direct else (xyz * scale) @ np.linalg.inv(cell)
        if sd:
            flags = np.array([[t.upper().startswith("T") for t in r[3:6]] for r in rows])
        return cls(L[0], cell, names, counts, frac, flags)

    def write(self, path, comment=None):
        out = [comment or self.comment, "   1.00000000000000"]
        out += ["  " + "  ".join(f"{x:21.16f}" for x in v) for v in self.cell]
        out += ["  " + "  ".join(f"{n:>4s}" for n in self.names), "  " + "  ".join(f"{n:>4d}" for n in self.counts)]
        if self.flags is not None:
            out.append("Selective dynamics")
        out.append("Direct")
        for i, f in enumerate(self.frac):
            row = "  " + "  ".join(f"{x:19.16f}" for x in f)
            if self.flags is not None:
                row += "   " + "   ".join("T" if m else "F" for m in self.flags[i])
            out.append(row)
        Path(path).write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")


def last_forces(outcar, natoms):
    """Forces (natoms, 3) of the last TOTAL-FORCE block of an OUTCAR, or None."""
    F = None
    with open(outcar, errors="replace") as f:
        lines = iter(f)
        for ln in lines:
            if "TOTAL-FORCE" in ln:
                next(lines)
                F = np.array([[float(x) for x in next(lines).split()[3:6]] for _ in range(natoms)])
    return F


def compress(indices):
    """[1,2,3,7,9,10] -> '1-3,7,9-10'."""
    out, idx = [], sorted(indices)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        out.append(f"{idx[i]}" if i == j else f"{idx[i]}-{idx[j]}")
        i = j + 1
    return ",".join(out)


def parse_indices(text, n):
    """'1-3,7' -> boolean mask of length n (1-based input)."""
    m = np.zeros(n, bool)
    for part in text.split(","):
        a, _, b = part.partition("-")
        m[int(a) - 1:int(b or a)] = True
    return m


# =========================================================================== commands
def cmd_report(p, outcar=None):
    sym = np.array(p.symbols)
    print(f"{len(sym)} atoms, species {' '.join(f'{n}{c}' for n, c in zip(p.names, p.counts))}")
    if p.flags is None:
        print("no 'Selective dynamics' line: every atom is free")
        return 0
    fixed_any = ~p.flags.all(axis=1)
    fixed_all = ~p.flags.any(axis=1)
    print(f"selective dynamics: {fixed_any.sum()} atom(s) with at least one fixed coordinate, "
          f"{fixed_all.sum()} completely fixed\n")
    print(f"{'element':<8s} {'atoms':>6s} {'fixed x':>8s} {'fixed y':>8s} {'fixed z':>8s}   indices (1-based)")
    for e in p.names:
        m = sym == e
        fx = (~p.flags[m]).sum(axis=0)
        idx = np.where(m & fixed_any)[0] + 1
        print(f"{e:<8s} {m.sum():6d} {fx[0]:8d} {fx[1]:8d} {fx[2]:8d}   {compress(idx.tolist()) or '-'}")
    if outcar:
        F = last_forces(outcar, len(sym))
        if F is None:
            print("\nno TOTAL-FORCE block in the OUTCAR")
        else:
            fn = np.linalg.norm(F, axis=1)
            ffree = np.linalg.norm(np.where(p.flags, F, 0.0), axis=1)
            print(f"\nlast ionic step: max |F| on free coordinates {ffree.max():.3f} eV/Å "
                  f"(the quantity compared with EDIFFG)")
            if fixed_all.any():
                k = np.argmax(np.where(fixed_all, fn, -1))
                print(f"                 max |F| on fixed atoms     {fn[fixed_all].max():.3f} eV/Å "
                      f"(atom {k + 1}, {sym[k]}) — not part of the convergence test")
    return 0


def cmd_free(p, out):
    n = 0 if p.flags is None else int((~p.flags).sum())
    p.flags = None
    p.write(out)
    print(f"{n} fixed coordinate(s) released -> {out}")
    return 0


def cmd_fix(p, out, a):
    sym = np.array(p.symbols)
    n = len(sym)
    sel = np.ones(n, bool)
    ax = "abc".index(a.axis)
    height = abs(np.linalg.det(p.cell)) / np.linalg.norm(np.cross(*[p.cell[i] for i in range(3) if i != ax]))
    z = p.frac[:, ax] % 1.0
    if a.below is not None:
        sel &= z < a.below
    if a.above is not None:
        sel &= z > a.above
    if a.below_A is not None:
        sel &= z * height < a.below_A
    if a.above_A is not None:
        sel &= z * height > a.above_A
    if a.elements:
        sel &= np.isin(sym, a.elements.split(","))
    if a.indices:
        sel &= parse_indices(a.indices, n)
    if not any(v is not None for v in (a.below, a.above, a.below_A, a.above_A, a.elements, a.indices)):
        print("no selection given (use --below/--above/--below-A/--above-A/--elements/--indices)", file=sys.stderr)
        return 1
    dirs = np.array([c in a.directions for c in "xyz"])
    flags = p.flags.copy() if (p.flags is not None and a.keep_existing) else np.ones((n, 3), bool)
    flags[sel] &= ~dirs
    p.flags = flags
    p.write(out)
    print(f"fixed {sel.sum()} atom(s) along {a.directions}: {compress((np.where(sel)[0] + 1).tolist()) or '-'} -> {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="summary of the flags")
    r.add_argument("poscar")
    r.add_argument("--outcar", help="OUTCAR of the run, for the residual forces")
    f = sub.add_parser("free", help="remove every constraint")
    f.add_argument("poscar")
    f.add_argument("-o", "--out", required=True)
    x = sub.add_parser("fix", help="fix a selection of atoms (selections are intersected)")
    x.add_argument("poscar")
    x.add_argument("-o", "--out", required=True)
    x.add_argument("--axis", default="c", choices=list("abc"), help="axis for the height selections (default c)")
    x.add_argument("--below", type=float, help="fractional coordinate below this value")
    x.add_argument("--above", type=float, help="fractional coordinate above this value")
    x.add_argument("--below-A", dest="below_A", type=float, help="height (Å) below this value")
    x.add_argument("--above-A", dest="above_A", type=float, help="height (Å) above this value")
    x.add_argument("--elements", help="comma-separated elements, e.g. Pb,I")
    x.add_argument("--indices", help="1-based indices and ranges, e.g. 1-12,20")
    x.add_argument("--directions", default="xyz", help="coordinates to fix, e.g. z or xyz (default)")
    x.add_argument("--keep-existing", action="store_true", help="keep the flags already in the file")
    a = ap.parse_args(argv)
    p = Poscar.read(a.poscar)
    if a.cmd == "report":
        return cmd_report(p, a.outcar)
    if a.cmd == "free":
        return cmd_free(p, a.out)
    return cmd_fix(p, a.out, a)


if __name__ == "__main__":
    raise SystemExit(main())
