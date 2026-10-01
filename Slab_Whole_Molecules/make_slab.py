#!/usr/bin/env python3
"""
make_slab.py — cut a slab with vacuum from a layered hybrid (organic-inorganic) bulk cell,
keeping every molecule whole.

The problem
-----------
The usual recipe — wrap the atoms into the cell and call ase ``atoms.center(vacuum, axis=2)``
— works atom by atom. In a layered hybrid material (2D perovskites, intercalated
layered compounds, MOF layers ...) the organic molecules often cross the cell boundary
along the stacking axis. Wrapped atom by atom, a molecule whose C-H or C-N bond crosses
that boundary ends up with one half above the slab and the other half below it, on the
far side of the new vacuum: the slab contains broken molecules, and every property you
compute on it is wrong. The C-H bond lengths are the quickest symptom.

What this script does
---------------------
1. finds the inorganic layer (``--inorganic`` elements) and its mean plane, using the
   atoms of ``--center`` (e.g. Pb) as reference;
2. brings every inorganic atom to the periodic image closest to that plane;
3. finds the molecules (connected components of the remaining atoms, covalent radii x 1.2),
   rebuilds each one through its bonds (minimum image), and places it in the periodic image
   where its anchor atom (``--anchor``, e.g. the N of an ammonium group) is closest to the
   layer it binds to;
4. opens ``--vacuum`` Å of vacuum along the stacking axis (half on each side);
5. sorts the species (``--order``) so that the POTCAR can be concatenated in the same order;
6. checks the result: number of molecules before/after and every X-H bond length.

Atoms only move by lattice vectors, so the crystal is unchanged: same bonds, same molecules.

Requirements: numpy, scipy, ase.

Usage
-----
    python make_slab.py CONTCAR -o POSCAR_slab --inorganic Pb,I --center Pb --anchor N --vacuum 15
    python make_slab.py bulk.vasp -o slab.vasp --inorganic Sn,Br --center Sn --order C,H,Br,N,Sn
    python make_slab.py CONTCAR -o POSCAR_naive --naive          # the atom-by-atom recipe, for comparison

Author: Israel C. Ribeiro — https://github.com/israel-c-ribeiro/VASP_tools (MIT licence)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

__version__ = "1.0.0"


def molecules(atoms, inorganic, mult=1.2):
    """Indices of the non-inorganic atoms and their connected-component labels (periodic bonds)."""
    from ase.neighborlist import NeighborList, natural_cutoffs
    from scipy.sparse.csgraph import connected_components
    sym = np.array(atoms.get_chemical_symbols())
    org = np.where(~np.isin(sym, inorganic))[0]
    sub = atoms[org]
    nl = NeighborList(natural_cutoffs(sub, mult=mult), self_interaction=False, bothways=True)
    nl.update(sub)
    n, comp = connected_components(nl.get_connectivity_matrix(sparse=True), directed=False)
    return org, comp, nl, n


def inorganic_layers(atoms, inorganic, mult=1.2):
    """Number of disconnected inorganic layers in the (fully periodic) cell."""
    from ase.neighborlist import NeighborList, natural_cutoffs
    from scipy.sparse.csgraph import connected_components
    idx = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s in inorganic]
    if not idx:
        raise ValueError(f"no atom of {inorganic} in the structure")
    sub = atoms[idx]
    nl = NeighborList(natural_cutoffs(sub, mult=mult), self_interaction=False, bothways=True)
    nl.update(sub)
    return connected_components(nl.get_connectivity_matrix(sparse=True), directed=False)[0]


def count_fragments(atoms, mult=1.2):
    """Number of connected pieces WITHOUT periodicity along the stacking axis (a split molecule counts twice)."""
    from ase.neighborlist import NeighborList, natural_cutoffs
    from scipy.sparse.csgraph import connected_components
    a = atoms.copy()
    a.pbc = [True, True, False]
    nl = NeighborList(natural_cutoffs(a, mult=mult), self_interaction=False, bothways=True)
    nl.update(a)
    return connected_components(nl.get_connectivity_matrix(sparse=True), directed=False)[0]


def xh_bonds(atoms, cutoff=1.25, periodic=True):
    """Shortest distance of every H to a heavy atom (Å). With ``periodic=False`` the stacking axis
    is treated as open, which is what matters for a slab: a split bond shows up as a long distance."""
    a = atoms.copy()
    if not periodic:
        a.pbc = [True, True, False]
    sym = np.array(a.get_chemical_symbols())
    H, X = np.where(sym == "H")[0], np.where(sym != "H")[0]
    if not len(H) or not len(X):
        return np.array([])
    d = a.get_all_distances(mic=True)[np.ix_(H, X)].min(axis=1)
    return d


def whole_molecule_slab(atoms, inorganic, center, anchor="N", vacuum=15.0, axis=2, mult=1.2):
    """Return a new Atoms object: slab with ``vacuum`` Å along ``axis`` and whole molecules."""
    at = atoms.copy()
    at.set_constraint()
    sym = np.array(at.get_chemical_symbols())
    cell = at.cell.array
    frac = at.get_scaled_positions(wrap=True)
    ref = np.where(sym == center)[0]
    if not len(ref):
        raise ValueError(f"no {center} atom to define the layer plane")
    z0 = frac[ref[0], axis]
    z_ref = z0 + np.mean((frac[ref, axis] - z0) - np.round(frac[ref, axis] - z0))   # periodic mean
    pos = frac @ cell
    inorg = np.isin(sym, inorganic)
    n_layers = inorganic_layers(at, inorganic, mult)
    if n_layers != 1:
        raise ValueError(f"the cell contains {n_layers} separate inorganic layers; this recipe needs exactly one "
                         f"per cell along the stacking axis (cut or reduce the cell first)")
    for i in np.where(inorg)[0]:
        pos[i] -= np.round(frac[i, axis] - z_ref) * cell[axis]
    org, comp, nl, n = molecules(at, inorganic, mult)
    sub_pos = at.positions[org]
    sub_sym = sym[org]
    inv = np.linalg.inv(cell)
    for c in range(n):
        members = np.where(comp == c)[0]
        new = {members[0]: sub_pos[members[0]]}
        todo = [members[0]]
        while todo:                                       # breadth-first walk through the bonds
            i = todo.pop()
            idx, offs = nl.get_neighbors(i)
            for j, off in zip(idx, offs):
                if j not in new:
                    new[j] = new[i] + (sub_pos[j] + off @ cell - sub_pos[i])
                    todo.append(j)
        mpos = np.array([new[j] for j in members])
        anchors = [k for k, j in enumerate(members) if sub_sym[j] == anchor]
        point = mpos[anchors[0]] if anchors else mpos.mean(axis=0)
        mpos -= np.round((point @ inv)[axis] - z_ref) * cell[axis]
        pos[org[members]] = mpos
    at.set_positions(pos)
    at.center(vacuum=vacuum / 2.0, axis=axis)
    return at


def naive_slab(atoms, vacuum=15.0, axis=2):
    """The atom-by-atom recipe (wrap, then center): kept to demonstrate the problem."""
    at = atoms.copy()
    at.set_constraint()
    at.wrap()
    at.center(vacuum=vacuum / 2.0, axis=axis)
    return at


def sort_species(atoms, order):
    rank = {e: i for i, e in enumerate(order)}
    idx = sorted(range(len(atoms)), key=lambda i: (rank.get(atoms[i].symbol, len(rank)), i))
    return atoms[idx]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structure", help="bulk cell (POSCAR/CONTCAR or any format ase can read)")
    ap.add_argument("-o", "--out", required=True, help="output POSCAR")
    ap.add_argument("--inorganic", default="Pb,I", help="elements of the inorganic layer (default Pb,I)")
    ap.add_argument("--center", default="Pb", help="element that defines the layer plane (default Pb)")
    ap.add_argument("--anchor", default="N", help="atom of each molecule kept next to the layer (default N)")
    ap.add_argument("--vacuum", type=float, default=15.0, help="total vacuum thickness in Å (default 15)")
    ap.add_argument("--axis", type=int, default=2, choices=[0, 1, 2], help="stacking axis (default 2 = c)")
    ap.add_argument("--order", help="species order of the output, e.g. C,H,I,N,Pb (default: order of first appearance)")
    ap.add_argument("--naive", action="store_true", help="use the atom-by-atom recipe instead (to see the problem)")
    ap.add_argument("--version", action="version", version=__version__)
    a = ap.parse_args(argv)

    from ase.io import read, write
    bulk = read(a.structure)
    inorganic = a.inorganic.split(",")
    if a.naive:
        slab = naive_slab(bulk, a.vacuum, a.axis)
    else:
        slab = whole_molecule_slab(bulk, inorganic, a.center, a.anchor, a.vacuum, a.axis)
    if a.order:
        slab = sort_species(slab, a.order.split(","))
    write(a.out, slab, format="vasp", direct=True, sort=False)

    _, _, _, n_mol = molecules(bulk, inorganic)
    pieces_bulk = n_mol + 1                               # molecules + one inorganic layer
    pieces_slab = count_fragments(slab)
    xh = xh_bonds(slab, periodic=False)
    gap = slab.cell.lengths()[a.axis] - np.ptp(slab.positions[:, a.axis])
    print(f"bulk : {len(bulk)} atoms, {n_mol} molecules, c = {bulk.cell.lengths()[a.axis]:.3f} Å")
    print(f"slab : c = {slab.cell.lengths()[a.axis]:.3f} Å, atom-free gap {gap:.2f} Å -> {a.out}")
    print(f"connected pieces in the slab: {pieces_slab} (expected {pieces_bulk}: {n_mol} molecules + 1 layer)")
    if len(xh):
        print(f"X-H bonds: {len(xh)}, {xh.min():.3f}-{xh.max():.3f} Å")
    ok = pieces_slab <= pieces_bulk and (not len(xh) or xh.max() < 1.25)
    print("check: OK, every molecule is whole" if ok else
          f"check: FAILED - {pieces_slab - pieces_bulk} extra piece(s): molecules are split across the vacuum")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
