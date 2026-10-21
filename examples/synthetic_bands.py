"""
synthetic_bands.py — synthetic band-structure and DOS runs with a KNOWN answer (Band_Structure_DOS).

The structure is the toy lead-iodide layer of synthetic_vasp.py (1 Pb, 4 I, 2 CH3NH3-like cations,
square in-plane cell a = 6.3 Å). The bands are analytic tight-binding-like cosines, not the result of
a calculation: the top valence band and the bottom conduction band both have their extremum at
S = (1/2, 1/2, 0), so the gap is direct at S and its value is chosen in advance.

    bands_nosoc/   PBE+D3, line-mode path G-X | S-Y-G (a jump between X and S): VBM -1.20, CBM 0.60 eV, gap 1.80 eV
    bands_soc/     PBE+D3+SOC, same path, Kramers-doubled bands, the lowest conduction band pushed down
                   by spin-orbit coupling: VBM -1.15, CBM -0.10 eV, gap 1.05 eV
    bands_hybrid/  HSE06+SOC, explicit KPOINTS: a weighted 4x4 mesh followed by the zero-weight path
                   (labels in the 5th column, as written for hybrid band runs): gap 1.70 eV
    dos_lorbit11/  PBE+D3 DOS run: DOSCAR (LORBIT = 11) made of Gaussian peaks of known area per element
                   and orbital, EIGENVAL on a 4x4 mesh (VBM -1.20, CBM 0.60 eV), POSCAR, INCAR, OUTCAR

Usage
-----
    python synthetic_bands.py                 # writes the four folders next to this file
    python synthetic_bands.py --out my_dir

    from synthetic_bands import make_bands_example
    truth = make_bands_example("examples")    # ground truth of every folder
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from synthetic_vasp import toy_slab, write_eigenval, write_outcar, write_poscar

# path of the line-mode examples: three segments, a jump between X and S
PATH = [("G", (0.0, 0.0, 0.0), "X", (0.5, 0.0, 0.0)),
        ("S", (0.5, 0.5, 0.0), "Y", (0.0, 0.5, 0.0)),
        ("Y", (0.0, 0.5, 0.0), "G", (0.0, 0.0, 0.0))]
NELECT = 8                       # four doubly occupied valence bands (eight singly occupied with SOC)
LEVELS = {                        # (VBM, CBM) in eV of each level of theory
    "nosoc": (-1.20, 0.60),
    "soc": (-1.15, -0.10),
    "hybrid": (-1.45, 0.25),
}

# DOS peaks: element -> list of (orbital l, centre, sigma, area per atom). Centres of the valence peaks
# are relative to the VBM, of the conduction peaks ("c") relative to the CBM. Every peak lies 5 sigma
# inside its window, so the integrals below are exact to < 1e-4 states.
DOS_PEAKS = {
    "Pb": [(0, -0.50, 0.10, 0.60), (1, ("c", 0.50), 0.10, 1.50), (1, ("c", 2.00), 0.20, 0.50)],
    "I": [(1, -0.50, 0.10, 1.00), (1, -2.50, 0.25, 0.60), (0, -4.80, 0.30, 0.50), (1, ("c", 0.50), 0.10, 0.10)],
    "C": [(1, -3.60, 0.25, 0.50), (1, ("c", 2.80), 0.25, 0.30)],
    "N": [(1, -4.20, 0.25, 0.60), (1, ("c", 3.20), 0.25, 0.30)],
    "H": [(0, -3.90, 0.30, 0.10), (0, ("c", 3.00), 0.30, 0.05)],
}
LM_COLUMNS = {0: [0], 1: [1, 2, 3], 2: [4, 5, 6, 7, 8]}      # LORBIT = 11 column of each l (s, py pz px, d)


# --------------------------------------------------------------------------- k-points and bands
def line_path(n: int = 40) -> np.ndarray:
    """k-points of PATH as VASP generates them in line mode: n points per segment, end points included."""
    return np.vstack([np.linspace(a, b, n) for _, a, _, b in PATH])


def _f(k: np.ndarray) -> np.ndarray:
    """0 at S = (1/2, 1/2), 2 at Γ: the shape of every toy band."""
    return np.cos(np.pi * k[:, 0]) ** 2 + np.cos(np.pi * k[:, 1]) ** 2


def toy_bands(kpts: np.ndarray, level: str = "nosoc", ispin: int = 1, spin_split: float = 0.0):
    """Eigenvalues and occupations (nspin, nk, nb), sorted per k-point as in EIGENVAL.

    Without SOC: 4 valence + 4 conduction bands. With SOC ('soc', 'hybrid') every band becomes a
    Kramers pair and the lowest conduction pair (the j = 1/2 split-off band) moves down.
    With ``ispin = 2`` the whole spin-down channel is shifted down by ``spin_split`` (eV): the VBM
    stays in the spin-up channel and the CBM moves to the spin-down channel."""
    vbm, cbm = LEVELS[level]
    f = _f(kpts)
    scale = 1.1 if level == "hybrid" else 1.0
    vb = [vbm - scale * 0.9 * f, vbm - 0.35 - 0.6 * f, vbm - 0.8 - 0.3 * f, vbm - 1.6 - 0.2 * f]
    if level == "nosoc":
        cb = [cbm + scale * 1.0 * f, cbm + 0.10 + 0.8 * f, cbm + 0.10 + 1.2 * f, cbm + 1.5 + 0.3 * f]
        bands = vb + cb
    else:                                              # j = 1/2 pair low, j = 3/2 states 0.8 eV higher
        cb = [cbm + scale * 1.0 * f, cbm + 0.80 + 0.8 * f, cbm + 0.80 + 1.2 * f, cbm + 2.0 + 0.3 * f]
        bands = [b for b in vb + cb for _ in range(2)]
    eig = np.sort(np.array(bands).T, axis=1)            # (nk, nb)
    nocc = NELECT if level != "nosoc" else NELECT // 2
    occ = np.zeros_like(eig)
    occ[:, :nocc] = 1.0
    if ispin == 1:
        return eig[None], occ[None]
    down = eig.copy()
    down[:, :nocc] -= spin_split
    down[:, nocc:] -= spin_split
    return np.stack([eig, down]), np.stack([occ, occ])


def mesh(n: int = 4) -> np.ndarray:
    """Γ-centred n x n x 1 mesh (fractional); contains S = (1/2, 1/2, 0) for even n."""
    g = np.arange(n) / n
    g = np.where(g > 0.5, g - 1.0, g)
    return np.array([(x, y, 0.0) for x in g for y in g])


# --------------------------------------------------------------------------- writers
def write_text(path, lines) -> None:
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_incar(path, tags: dict) -> None:
    write_text(path, [f"{k} = {v}" for k, v in tags.items()])


def write_kpoints_line(path, n: int = 40, gamma: str = "G") -> None:
    """Line-mode KPOINTS of PATH; ``gamma`` is how Γ is spelled (G, GAMMA or \\Gamma)."""
    lines = ["k-path G-X|S-Y-G (synthetic)", str(n), "Line-mode", "Reciprocal"]
    for la, a, lb, b in PATH:
        la, lb = (gamma if x == "G" else x for x in (la, lb))
        lines += [f"  {a[0]:.6f}  {a[1]:.6f}  {a[2]:.6f}  ! {la}", f"  {b[0]:.6f}  {b[1]:.6f}  {b[2]:.6f}  ! {lb}", ""]
    write_text(path, lines)


def write_kpoints_explicit(path, kpts, weights, labels) -> None:
    """Explicit KPOINTS (reciprocal); labels go in the 5th column, as written for hybrid band runs."""
    lines = ["Explicit k-points: weighted SCF mesh + zero-weight band path (synthetic)", str(len(kpts)), "Reciprocal"]
    for k, w, lab in zip(kpts, weights, labels):
        lines.append(f"  {k[0]:.8f}  {k[1]:.8f}  {k[2]:.8f}  {w:g}" + (f"  {lab}" if lab else ""))
    write_text(path, lines)


def write_eigenval_weighted(path, eig, occ, kpts, weights, nelect, natoms) -> None:
    """EIGENVAL with explicit k-point weights (zero for the path points of a hybrid band run)."""
    nspin, nk, nb = eig.shape
    lines = [f"{natoms:5d}{natoms:5d}    1{nspin:5d}",
             "  0.1190700E+04  0.6300000E-09  0.6300000E-09  0.3000000E-08  0.5000000E-15",
             "  1.000000000000000E-004", "  CAR", " synthetic", f"{int(nelect):7d}{nk:7d}{nb:7d}"]
    for ik in range(nk):
        lines.append("")
        lines.append("  " + "  ".join(f"{x:.7E}" for x in kpts[ik]) + f"  {weights[ik]:.7E}")
        for ib in range(nb):
            if nspin == 1:
                lines.append(f"{ib + 1:5d}   {eig[0, ik, ib]:13.6f}   {occ[0, ik, ib]:9.6f}")
            else:
                lines.append(f"{ib + 1:5d}   {eig[0, ik, ib]:13.6f}   {eig[1, ik, ib]:13.6f}"
                             f"   {occ[0, ik, ib]:9.6f}   {occ[1, ik, ib]:9.6f}")
    write_text(path, lines)


def write_outcar_bands(path, *, natoms, efermi, ispin=1, ncl=False, nelect=NELECT) -> None:
    """The minimal OUTCAR of synthetic_vasp.py plus the ISPIN, LNONCOLLINEAR and NELECT lines."""
    write_outcar(path, natoms=natoms, efermi=efermi, lsorbit=ncl)
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    extra = [f"   ISPIN  =      {ispin}    spin polarized calculation?",
             f"   LNONCOLLINEAR =      {'T' if ncl else 'F'} non collinear calculations",
             f"   NELECT =       {nelect:.4f}    total number of electrons"]
    write_text(path, lines[:2] + extra + lines[2:])


def write_doscar(path, energy, efermi, total, pdos, natoms, ncl=False, lorbit=11) -> None:
    """DOSCAR. ``total`` (nspin, nE); ``pdos`` (nspin, natoms, nE, 9) in the LORBIT = 11 lm order.

    LORBIT = 10 sums the lm columns of each l (s, p, d). Spin-polarized columns are interleaved
    (s_up s_down ...); non-collinear columns carry (total, m_x, m_y, m_z) for every orbital, and
    the magnetization components are given arbitrary non-zero values so that a reader that used
    them would be caught."""
    nspin, nE = total.shape
    head = f"  {energy.max():14.8f}  {energy.min():14.8f}  {nE:5d}  {efermi:14.8f}  1.00000000"
    lines = [f"{natoms:4d}{natoms:4d}   1   0",
             "  0.1190700E+04  0.6300000E-09  0.6300000E-09  0.3000000E-08  0.5000000E-15",
             "  1.000000000000000E-004", "  CAR", " synthetic", head]
    de = energy[1] - energy[0]
    integ = np.cumsum(total, axis=1) * de
    for i in range(nE):
        vals = [total[s, i] for s in range(nspin)] + [integ[s, i] for s in range(nspin)]
        lines.append(f"  {energy[i]:9.4f}  " + "  ".join(f"{v:.4E}" for v in vals))
    if lorbit == 10:
        pdos = np.stack([pdos[..., LM_COLUMNS[l]].sum(axis=-1) for l in (0, 1, 2)], axis=-1)
    for a in range(natoms):
        lines.append(head)
        for i in range(nE):
            vals = []
            for o in range(pdos.shape[-1]):
                if ncl:
                    t = pdos[0, a, i, o]
                    vals += [t, 0.5 * t, -0.3 * t, 0.2 * t]
                else:
                    vals += [pdos[s, a, i, o] for s in range(nspin)]
            lines.append(f"  {energy[i]:9.4f}  " + "  ".join(f"{v:.4E}" for v in vals))
    write_text(path, lines)


# --------------------------------------------------------------------------- example folders
def _structure():
    symbols, cell, frac = toy_slab()
    return symbols, cell, frac


def make_band_run(folder, level: str = "nosoc", *, kind: str = "line", ispin: int = 1, spin_split: float = 0.0,
                  zero_occ_on_path: bool = False, gamma: str = "G", n: int = 40) -> dict:
    """Write one band-structure folder (POSCAR, INCAR, KPOINTS, EIGENVAL, OUTCAR) and return its truth.

    ``kind`` = 'line' (line-mode KPOINTS) or 'explicit' (weighted 4x4 mesh + zero-weight path, the layout
    of a hybrid-functional band run). ``zero_occ_on_path`` writes zero occupations on the path points,
    which forces the reader to fall back on NELECT."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    symbols, cell, frac = _structure()
    write_poscar(folder / "POSCAR", symbols, cell, frac, comment="toy lead-iodide layer (synthetic)")
    soc = level != "nosoc"
    tags = {"SYSTEM": "toy band structure (synthetic)", "ICHARG": 11, "GGA": "PE", "IVDW": 11, "ISMEAR": 0,
            "SIGMA": 0.01, "LORBIT": 11, "ISPIN": ispin}
    if level == "hybrid":
        tags = {"SYSTEM": "toy band structure (synthetic)", "LHFCALC": ".TRUE.", "HFSCREEN": 0.2, "AEXX": 0.25,
                "ALGO": "Normal", "ISMEAR": 0, "SIGMA": 0.01, "ISPIN": ispin}
    if soc:
        tags.update({"LSORBIT": ".TRUE.", "SAXIS": "0 0 1"})
        tags.pop("ISPIN")
    write_incar(folder / "INCAR", tags)
    kpath = line_path(n)
    labels_path = [None] * len(kpath)
    for s, (la, _, lb, _) in enumerate(PATH):
        labels_path[s * n], labels_path[s * n + n - 1] = la, lb
    if kind == "line":
        write_kpoints_line(folder / "KPOINTS", n, gamma)
        eig, occ = toy_bands(kpath, level, ispin, spin_split)
        write_eigenval(folder / "EIGENVAL", eig, occ, kpath, NELECT, len(symbols))
    elif kind == "explicit":
        km = mesh(4)
        kpts = np.vstack([km, kpath])
        w = np.r_[np.ones(len(km)), np.zeros(len(kpath))]
        labels = [None] * len(km) + [gamma if lab == "G" else lab for lab in labels_path]
        write_kpoints_explicit(folder / "KPOINTS", kpts, w, labels)
        eig, occ = toy_bands(kpts, level, ispin, spin_split)
        if zero_occ_on_path:
            occ[:, len(km):] = 0.0
        write_eigenval_weighted(folder / "EIGENVAL", eig, occ, kpts, w / w.sum(), NELECT, len(symbols))
    else:
        raise ValueError(f"unknown kind {kind!r}")
    vbm, cbm = LEVELS[level]
    if ispin == 2 and spin_split:
        cbm -= spin_split
    write_outcar_bands(folder / "OUTCAR", natoms=len(symbols), efermi=vbm + 0.05, ispin=ispin, ncl=soc)
    label = {"nosoc": "PBE+D3", "soc": "PBE+D3+SOC", "hybrid": "HSE06+SOC"}[level]
    return dict(VBM=vbm, CBM=cbm, gap=round(cbm - vbm, 6), k_VBM="S", k_CBM="S", direct=True,
                ticks=["Γ", "X|S", "Y", "Γ"], nk_path=len(kpath), nbands=8 if level == "nosoc" else 16,
                label=label, efermi=vbm + 0.05)


def dos_curves(energy: np.ndarray, vbm: float, cbm: float, symbols: list[str]) -> np.ndarray:
    """Projected DOS (natoms, nE, 9 lm columns) built from DOS_PEAKS."""
    out = np.zeros((len(symbols), len(energy), 9))
    for a, el in enumerate(symbols):
        for l, centre, sigma, area in DOS_PEAKS.get(el, []):
            e0 = cbm + centre[1] if isinstance(centre, tuple) else vbm + centre
            g = area * np.exp(-0.5 * ((energy - e0) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            cols = LM_COLUMNS[l]
            for c in cols:
                out[a, :, c] += g / len(cols)
    return out


def make_dos_run(folder, *, lorbit: int = 11, ispin: int = 1, ncl: bool = False, down_scale: float = 0.9,
                 edge_window: float = 1.0) -> dict:
    """Write a DOS folder (DOSCAR, POSCAR, INCAR, OUTCAR, EIGENVAL) and return its truth.

    With ``ispin = 2`` the spin-down projections are ``down_scale`` times the spin-up ones.
    Truth: per-element states (spin up / single channel) in the whole grid, in the ``edge_window`` eV
    below the VBM and above the CBM."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    symbols, cell, frac = _structure()
    write_poscar(folder / "POSCAR", symbols, cell, frac, comment="toy lead-iodide layer (synthetic)")
    level = "soc" if ncl else "nosoc"
    vbm, cbm = LEVELS[level]
    tags = {"SYSTEM": "toy DOS (synthetic)", "GGA": "PE", "IVDW": 11, "ISMEAR": -5, "LORBIT": lorbit, "NEDOS": 401}
    if ncl:
        tags.update({"LSORBIT": ".TRUE.", "SAXIS": "0 0 1"})
    else:
        tags["ISPIN"] = ispin
    write_incar(folder / "INCAR", tags)
    km = mesh(4)
    eig, occ = toy_bands(km, level, ispin)
    write_eigenval(folder / "EIGENVAL", eig, occ, km, NELECT, len(symbols))
    write_outcar_bands(folder / "OUTCAR", natoms=len(symbols), efermi=vbm + 0.05, ispin=ispin, ncl=ncl)
    energy = np.linspace(vbm - 6.5, vbm + 7.5, 401)
    p = dos_curves(energy, vbm, cbm, symbols)
    pdos = np.stack([p, down_scale * p]) if ispin == 2 else p[None]
    total = pdos.sum(axis=(1, 3))
    write_doscar(folder / "DOSCAR", energy, vbm + 0.05, total, pdos, len(symbols), ncl=ncl, lorbit=lorbit)
    counts = {el: symbols.count(el) for el in dict.fromkeys(symbols)}

    def states(el, which):
        tot = 0.0
        for l, centre, _, area in DOS_PEAKS[el]:
            e0 = cbm + centre[1] if isinstance(centre, tuple) else vbm + centre
            inside = {"all": True, "vb": vbm - edge_window < e0 < vbm, "cb": cbm < e0 < cbm + edge_window}[which]
            tot += area * counts[el] if inside else 0.0
        return tot
    return dict(VBM=vbm, CBM=cbm, gap=round(cbm - vbm, 6), efermi=vbm + 0.05, species=list(counts),
                counts=counts, nedos=len(energy), edge_window=edge_window, down_scale=down_scale,
                states_all={el: states(el, "all") for el in counts},
                states_vb_edge={el: states(el, "vb") for el in counts},
                states_cb_edge={el: states(el, "cb") for el in counts},
                p_states_all={el: sum(ar * counts[el] for l, _, _, ar in DOS_PEAKS[el] if l == 1) for el in counts})


def make_bands_example(out) -> dict:
    """The four example folders of the Band_Structure_DOS tutorial; returns {folder name: truth}."""
    out = Path(out)
    return {
        "bands_nosoc": make_band_run(out / "bands_nosoc", "nosoc"),
        "bands_soc": make_band_run(out / "bands_soc", "soc"),
        "bands_hybrid": make_band_run(out / "bands_hybrid", "hybrid", kind="explicit", gamma="\\Gamma", n=20),
        "dos_lorbit11": make_dos_run(out / "dos_lorbit11"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent), help="parent folder (default: next to this file)")
    a = ap.parse_args()
    for name, t in make_bands_example(a.out).items():
        print(f"{name}: VBM={t['VBM']:.2f} CBM={t['CBM']:.2f} gap={t['gap']:.2f} eV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
