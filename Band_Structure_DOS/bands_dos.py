#!/usr/bin/env python3
"""
bands_dos.py — band structure(s) and element/orbital-projected DOS from VASP, side by side.

Background
----------
A band-structure run (non-self-consistent, ICHARG = 11, k-points along a path) and a DOS run
(a dense uniform mesh, LORBIT = 10/11) answer two halves of the same question: *where* in the
Brillouin zone the band edges are (and whether the gap is direct), and *which atoms and orbitals*
make those edges. This script reads both folders directly (numpy + matplotlib, no pymatgen), puts
every data set on the scale E - E_VBM of its own calculation, and draws the bands next to the
projected DOS. A second band folder (e.g. the same path without and with spin-orbit coupling) is
overlaid on the first.

Files read
    band folder : EIGENVAL, KPOINTS (line mode, or an explicit list with zero-weight path points as
                  in hybrid-functional runs; KPOINTS_OPT/EIGENVAL_OPT are used when present),
                  POSCAR (or CONTCAR), INCAR, OUTCAR
    DOS folder  : DOSCAR (projections need LORBIT = 10 or 11), POSCAR (or CONTCAR), INCAR, OUTCAR,
                  EIGENVAL (for the VBM/CBM of the DOS run)

Detected automatically
    * SOC / non-collinear runs (LNONCOLLINEAR in OUTCAR, else LSORBIT/LNONCOLLINEAR in INCAR), ISPIN 1/2
    * the method label of each data set (PBE+D3, PBE+D3+SOC, HSE06+SOC, ...) from INCAR; see --names
    * the k-path, its labels and its discontinuities (G/Gamma is written Γ; a jump A -> B is labelled A|B)
    * the VBM/CBM of every data set from its own occupations (fallback: NELECT); direct/indirect gap
    * DOS columns for LORBIT = 10 (s p d [f]) and 11 (lm-resolved), ISPIN = 2 (spin down drawn on the
      negative DOS axis) and non-collinear runs (only the total component of each orbital is used)

Usage
-----
    python bands_dos.py --bands BANDS_DIR --dos DOS_DIR
    python bands_dos.py --bands BANDS_noSOC BANDS_SOC --dos DOS_DIR       # two band sets overlaid
    python bands_dos.py --bands BANDS_DIR                                 # bands only
    python bands_dos.py --dos DOS_DIR --proj Pb:p I:p C N H               # DOS only, orbital-resolved
    python bands_dos.py --bands B --dos D -o fig.png --save-data          # one format + CSV of the curves

Output: <out>.png and <out>.pdf (or only the format given as the suffix of -o); --save-data also writes
<out>_bands_<i>.csv and <out>_dos.csv. A short report (VBM, CBM, gap and where they sit on the path;
the element/orbital character of the band edges) is printed for every data set.

Author: Israel C. Ribeiro — https://github.com/israel-c-ribeiro/VASP_tools (MIT licence)
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

__version__ = "1.1.0"

# ----------------------------------------------------------------------------- colours and style
ELEMENT_COLORS = {          # fixed colours for common elements; others cycle through EXTRA
    "I": "#1f77b4", "Pb": "#f28e2b", "C": "#17a589", "H": "#d1361f", "N": "#a8327f",
    "Br": "#6a3d9a", "Cl": "#33a02c", "Sn": "#8c564b", "Cs": "#7f7f7f", "O": "#e31a1c",
    "S": "#b8a000", "F": "#1abc9c", "Ge": "#b15928",
}
EXTRA = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3", "#8c8c8c"]
BAND_STYLES = [dict(color="#1f77b4", lw=1.1, ls="-"),              # first band set: solid blue
               dict(color="#d62728", lw=0.9, ls=(0, (1.2, 1.2))),  # second: dotted red, drawn on top
               dict(color="#2ca02c", lw=0.8, ls="--"),
               dict(color="#9467bd", lw=0.8, ls="-.")]
L_NAMES = {0: "s", 1: "p", 2: "d", 3: "f"}
# number of projected DOSCAR columns (after the energy) for each kind of run
PDOS_COLUMNS = {"collinear": (3, 4, 9, 16), "spin-polarized": (6, 8, 18, 32), "non-collinear": (12, 16, 36, 64)}


def _echo(text: str) -> None:
    """print() that survives a console/pipe whose encoding has no Γ (e.g. cp1252 on Windows)."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.replace("Γ", "G").encode(enc, "replace").decode(enc))


# ----------------------------------------------------------------------------- small readers
def _first(folder: Path, *names: str) -> Path | None:
    for n in names:
        p = Path(folder) / n
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _read(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def read_incar(folder: Path) -> dict[str, str]:
    """INCAR tags (upper-case keys, raw string values); {} if there is no INCAR."""
    p = _first(folder, "INCAR")
    tags: dict[str, str] = {}
    if p is None:
        return tags
    for line in _read(p).splitlines():
        line = re.split(r"[!#]", line, maxsplit=1)[0]
        for part in line.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                tags[k.strip().upper()] = v.strip()
    return tags


def _true(v: str | None) -> bool:
    return v is not None and v.strip().strip(".").upper().startswith("T")


def read_outcar(folder: Path) -> dict:
    """ISPIN, LNONCOLLINEAR, NELECT and the last E-fermi (streamed, so a large OUTCAR is fine)."""
    out = {"ispin": None, "ncl": None, "nelect": None, "efermi": None, "found": False}
    p = _first(folder, "OUTCAR")
    if p is None:
        return out
    out["found"] = True
    with open(p, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.lstrip()
            try:
                if out["ispin"] is None and s.startswith("ISPIN") and "=" in s:
                    out["ispin"] = int(s.split("=")[1].split()[0])
                elif out["ncl"] is None and s.startswith("LNONCOLLINEAR") and "=" in s:
                    out["ncl"] = s.split("=")[1].split()[0].upper().startswith("T")
                elif out["nelect"] is None and s.startswith("NELECT") and "=" in s:
                    out["nelect"] = float(s.split("=")[1].split()[0])
                elif "E-fermi" in s:
                    out["efermi"] = float(s.split(":")[1].split()[0])
            except (ValueError, IndexError):
                continue
    return out


def read_poscar(folder: Path) -> tuple[np.ndarray, list[str], list[int]]:
    """Cell (3x3, Å), species and counts of the POSCAR (or CONTCAR) of ``folder``."""
    p = _first(folder, "POSCAR", "CONTCAR")
    if p is None:
        raise FileNotFoundError(f"no POSCAR/CONTCAR in {folder} (needed for the reciprocal lattice "
                                "and the element of every atom)")
    lines = _read(p).splitlines()
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    if scale < 0:                                       # negative scale = target volume
        scale = (-scale / abs(np.linalg.det(cell))) ** (1 / 3)
    cell *= scale
    tok5, tok6 = lines[5].split(), lines[6].split()
    if all(t.isdigit() for t in tok5):                  # VASP 4 file: no species line
        raise ValueError(f"{p} has no species line (VASP 4 format); add the element names on line 6")
    species = [t.split("/")[0].split("_")[0] for t in tok5]
    counts = [int(t) for t in tok6]
    return cell, species, counts


def spin_flags(incar: dict, outcar: dict) -> tuple[int, bool]:
    """(ISPIN, non-collinear) from OUTCAR, falling back on INCAR."""
    ispin = outcar["ispin"] or int(incar.get("ISPIN", "1").split()[0])
    ncl = outcar["ncl"] if outcar["ncl"] is not None else (_true(incar.get("LSORBIT"))
                                                            or _true(incar.get("LNONCOLLINEAR")))
    return ispin, ncl


def method_label(incar: dict, ncl: bool) -> str:
    """Short name of the level of theory, e.g. PBE+D3+SOC or HSE06, from the INCAR tags."""
    def num(key, default=None):
        try:
            return float(incar[key].split()[0])
        except (KeyError, ValueError, IndexError):
            return default
    parts = []
    if _true(incar.get("LHFCALC")):
        aexx, screen = num("AEXX", 0.25), num("HFSCREEN", 0.0)
        if screen and screen > 0:
            parts.append("HSE06" if abs(screen - 0.2) < 0.01 and abs(aexx - 0.25) < 1e-3
                         else f"HSE({aexx:g},{screen:g})")
        else:
            parts.append("PBE0" if abs(aexx - 0.25) < 1e-3 else f"PBE0({aexx:g})")
    elif "METAGGA" in incar:
        parts.append(incar["METAGGA"].split()[0].upper().replace("R2SCAN", "r²SCAN"))
    else:
        gga = incar.get("GGA", "PE").split()[0].upper()
        parts.append({"PE": "PBE", "PS": "PBEsol", "RP": "RPBE", "91": "PW91", "AM": "AM05"}.get(gga, gga))
    ivdw = incar.get("IVDW", "0").split()[0]
    vdw = {"1": "D2", "10": "D2", "11": "D3", "12": "D3(BJ)", "13": "D4", "2": "TS", "20": "TS",
           "21": "TS-H", "202": "MBD", "4": "dDsC"}.get(ivdw)
    if vdw:
        parts.append(vdw)
    if _true(incar.get("LUSE_VDW")):
        parts.append("vdW-DF")
    if ncl and _true(incar.get("LSORBIT")):
        parts.append("SOC")
    elif ncl:
        parts.append("ncl")
    return "+".join(parts)


# ----------------------------------------------------------------------------- EIGENVAL / KPOINTS
def read_eigenval(path: Path) -> dict:
    """EIGENVAL: k-points (fractional), weights, eigenvalues and occupations as (spin, k, band)."""
    lines = _read(path).splitlines()
    try:
        ispin = int(lines[0].split()[3])
        nelect, nk, nb = (int(float(x)) for x in lines[5].split()[:3])
        tok = np.array(" ".join(lines[6:]).split(), dtype=float)
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cannot parse {path}: not an EIGENVAL file ({exc})") from None
    per_k = tok.size // nk
    t = (per_k - 4) // nb                                # tokens per band line
    if t < 2 or 4 + nb * t != per_k:
        raise ValueError(f"cannot parse {path}: {tok.size} numbers do not fit {nk} k-points x {nb} bands "
                         "(truncated file?)")
    blk = tok[: nk * per_k].reshape(nk, per_k)
    kpts, w = blk[:, :3], blk[:, 3]
    bands = blk[:, 4:].reshape(nk, nb, t)
    if ispin == 1:
        eig = bands[:, :, 1][None]
        occ = bands[:, :, 2][None] if t >= 3 else None
    else:
        eig = np.stack([bands[:, :, 1], bands[:, :, 2]])
        occ = np.stack([bands[:, :, 3], bands[:, :, 4]]) if t >= 5 else None
    return dict(ispin=ispin, nelect=nelect, kpts=kpts, weights=w, eig=eig, occ=occ)


def pretty_label(label: str | None) -> str:
    """'G', 'GAMMA', '\\Gamma' -> 'Γ'; 'S_0' -> S with subscript 0 (matplotlib mathtext)."""
    if not label:
        return ""
    s = label.strip().strip("'\"")
    if "|" in s:
        return "|".join(pretty_label(p) for p in s.split("|"))
    if s.upper() in ("G", "GAMMA", "\\GAMMA", "Γ", "$\\GAMMA$"):
        return "Γ"
    s = s.replace("\\Gamma", "Γ").replace("GAMMA", "Γ").replace("$", "")
    m = re.fullmatch(r"([A-Za-zΓ]+)_?(\d)", s)
    return f"{m.group(1)}$_{{{m.group(2)}}}$" if m else s


def _plain(label: str) -> str:
    return label.replace("$_{", "").replace("}$", "")


def _isfloat(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _label_of(line: str, ncoord: int = 3) -> str | None:
    if "!" in line:
        return line.split("!", 1)[1].strip() or None
    tok = line.split()
    return tok[ncoord] if len(tok) > ncoord and not _isfloat(tok[ncoord]) else None


def read_kpoints(path: Path | None) -> dict:
    """KPOINTS (or KPOINTS_OPT): line mode -> segments with their labels; explicit list -> weights + labels."""
    if path is None:
        return {"mode": "none"}
    lines = _read(path).splitlines()
    if len(lines) < 4:
        return {"mode": "none"}
    if lines[2].strip()[:1].upper() == "L":            # line mode
        n = int(lines[1].split()[0])
        pts = [ln for ln in lines[4:] if len(ln.split()) >= 3 and all(_isfloat(x) for x in ln.split()[:3])]
        if len(pts) % 2:
            raise ValueError(f"{path}: line mode needs pairs of end points, found {len(pts)} points")
        labels = [_label_of(ln) for ln in pts]
        segs = [(labels[i], labels[i + 1]) for i in range(0, len(pts) - 1, 2)]
        return {"mode": "line", "n": n, "segments": segs}
    try:
        nk = int(lines[1].split()[0])
    except (ValueError, IndexError):
        return {"mode": "none"}
    if nk == 0:
        return {"mode": "auto"}
    rows = lines[3:3 + nk]
    w = np.array([float(r.split()[3]) if len(r.split()) > 3 and _isfloat(r.split()[3]) else 1.0 for r in rows])
    labels = [_label_of(r, 4) for r in rows]
    return {"mode": "explicit", "weights": w, "labels": labels}


def build_path(ev: dict, kp: dict, cell: np.ndarray, user_labels: list[str] | None = None) -> dict:
    """x coordinate along the path (1/Å), line breaks, tick positions and labels.

    Zero-weight k-points (the path of a hybrid-functional band run) are kept and the weighted
    self-consistent mesh is dropped. ``brk[i]`` is True when the path jumps between k-points i-1 and i."""
    kpts, eig, occ, w = ev["kpts"], ev["eig"], ev["occ"], ev["weights"]
    labels_pt = None
    if (w == 0).any() and (w > 0).any():                # hybrid band run: keep the zero-weight path points
        keep = w == 0
        if kp.get("mode") == "explicit" and len(kp["labels"]) == len(w):
            labels_pt = [lab for lab, k in zip(kp["labels"], keep) if k]
        kpts, eig = kpts[keep], eig[:, keep]
        occ = occ[:, keep] if occ is not None else None
        kp = {"mode": "explicit-zero"}
    elif kp.get("mode") == "explicit" and len(kp["labels"]) == len(kpts):
        labels_pt = kp["labels"]
    B = 2 * np.pi * np.linalg.inv(cell).T               # rows = reciprocal lattice vectors (1/Å)
    cart = kpts @ B
    step = np.r_[0.0, np.linalg.norm(np.diff(cart, axis=0), axis=1)]
    nk = len(kpts)
    brk = np.zeros(nk, bool)
    ticks: list[int] = []
    tlabels: list[str] = []
    if kp.get("mode") == "line" and kp["n"] * len(kp["segments"]) == nk:
        n, segs = kp["n"], kp["segments"]
        for s, (a, b) in enumerate(segs):
            i0 = s * n
            if s == 0:
                ticks.append(i0)
                tlabels.append(pretty_label(a))
            elif step[i0] > 1e-5:                        # the segment does not start where the last ended
                brk[i0] = True
                tlabels[-1] = f"{pretty_label(segs[s - 1][1])}|{pretty_label(a)}"
            ticks.append(i0 + n - 1)
            tlabels.append(pretty_label(b))
    else:
        pos = step[1:][step[1:] > 1e-9]
        med = np.median(pos) if pos.size else 0.0
        brk[1:] = step[1:] > 6 * med if med > 0 else False
        if labels_pt and any(labels_pt):
            for i, lab in enumerate(labels_pt):
                if not lab:
                    continue
                lab = pretty_label(lab)
                if ticks and i - ticks[-1] == 1 and brk[i]:          # A then B across a jump -> A|B
                    tlabels[-1] = f"{tlabels[-1]}|{lab}"
                elif ticks and i - ticks[-1] == 1 and step[i] < 1e-9:  # repeated corner point
                    if lab != tlabels[-1]:
                        tlabels[-1] = f"{tlabels[-1]}|{lab}"
                elif not ticks or ticks[-1] != i:
                    ticks.append(i)
                    tlabels.append(lab)
        else:                                           # no labels: ticks at the kinks of the path
            d = np.diff(cart, axis=0)
            ticks = [0]
            for i in range(1, nk - 1):
                a, b = d[i - 1], d[i]
                if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9 or brk[i] or brk[i + 1]:
                    if brk[i + 1] or (np.linalg.norm(b) < 1e-9 and i - ticks[-1] > 1):
                        ticks.append(i)
                    continue
                if np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b) < 1 - 1e-4:
                    ticks.append(i)
            ticks.append(nk - 1)
            tlabels = [""] * len(ticks)
    step[brk] = 0.0
    x = np.cumsum(step)
    if user_labels:
        if len(user_labels) == len(ticks):
            tlabels = [pretty_label(lab) for lab in user_labels]
        else:
            _echo(f"  ! --labels has {len(user_labels)} entries but the path has {len(ticks)} ticks; ignored")
    return dict(x=x, brk=brk, ticks=ticks, tlabels=tlabels, kpts=kpts, eig=eig, occ=occ)


def band_edges(eig: np.ndarray, occ: np.ndarray | None, nelect: float, ncl: bool) -> dict:
    """VBM, CBM, their k indices and the smallest direct gap. Occupations first, NELECT otherwise.

    ``eig``/``occ`` have shape (spin, k, band). EIGENVAL occupations are 0..1 per state."""
    nspin = eig.shape[0]
    if occ is not None and np.nanmax(occ) > 0:
        filled = occ > 0.5
    elif nspin == 1:
        nocc = int(round(nelect if ncl else nelect / 2))
        filled = np.zeros(eig.shape, bool)
        filled[..., :nocc] = True
    else:                                               # ISPIN = 2 without occupations: lowest NELECT states per k
        e = np.sort(eig.transpose(1, 0, 2).reshape(eig.shape[1], -1), axis=1)
        vb = e[:, int(round(nelect)) - 1].max()
        filled = eig <= vb + 1e-9
    if not filled.any():
        raise ValueError("no occupied state found (check NELECT / occupations)")
    if filled.all():
        raise ValueError("every band is occupied: add empty bands (NBANDS) to see the conduction band")
    vbm = float(eig[filled].max())
    cbm = float(eig[~filled].min())
    kv = int(np.argwhere(filled & np.isclose(eig, vbm))[0][1])
    kc = int(np.argwhere(~filled & np.isclose(eig, cbm))[0][1])
    vbk = np.where(filled, eig, -np.inf).max(axis=(0, 2))
    cbk = np.where(~filled, eig, np.inf).min(axis=(0, 2))
    kd = int(np.argmin(cbk - vbk))
    return dict(vbm=vbm, cbm=cbm, gap=cbm - vbm, kv=kv, kc=kc, direct_gap=float(cbk[kd] - vbk[kd]), kd=kd)


# ----------------------------------------------------------------------------- DOSCAR
def read_doscar(folder: Path, ispin: int | None = None, ncl: bool = False) -> dict:
    """Total and projected DOS. ``pdos`` has shape (spin, atom, energy, orbital) or is None.

    ISPIN is taken from the number of total-DOS columns when not given; non-collinear runs keep
    only the total (first) of the four components of each orbital."""
    p = _first(folder, "DOSCAR")
    if p is None:
        raise FileNotFoundError(f"no DOSCAR in {folder}")
    lines = _read(p).splitlines()
    natoms = int(lines[0].split()[0])
    head = lines[5].split()
    nedos, efermi = int(head[2]), float(head[3])
    tot = np.array(" ".join(lines[6:6 + nedos]).split(), float).reshape(nedos, -1)
    spin_from_file = 2 if tot.shape[1] == 5 else 1
    if ispin is not None and ispin != spin_from_file:
        raise ValueError(f"{p}: {tot.shape[1]} total-DOS columns mean ISPIN = {spin_from_file}, "
                         f"but OUTCAR/INCAR say ISPIN = {ispin}")
    ispin = spin_from_file
    if ncl and ispin == 2:
        raise ValueError(f"{p}: a non-collinear run cannot have ISPIN = 2 columns")
    energy = tot[:, 0]
    total = np.stack([tot[:, 1], tot[:, 2]]) if ispin == 2 else tot[:, 1][None]
    pdos, norb = None, 0
    start = 6 + nedos
    if len(lines) >= start + natoms * (nedos + 1) and natoms > 0:
        ncol = len(lines[start + 1].split()) - 1
        kind = "non-collinear" if ncl else ("spin-polarized" if ispin == 2 else "collinear")
        if ncol not in PDOS_COLUMNS[kind]:
            raise ValueError(f"{p}: {ncol} projected columns per atom do not match a {kind} run with "
                             f"LORBIT = 10/11 (expected one of {PDOS_COLUMNS[kind]}); "
                             "check ISPIN/LNONCOLLINEAR in OUTCAR or INCAR")
        rows = [ln for a in range(natoms) for ln in lines[start + a * (nedos + 1) + 1: start + (a + 1) * (nedos + 1)]]
        arr = np.array(" ".join(rows).split(), float).reshape(natoms, nedos, ncol + 1)[:, :, 1:]
        if ncl:
            arr = arr[:, :, 0::4][None]                 # total component of each orbital
        elif ispin == 2:
            arr = np.stack([arr[:, :, 0::2], arr[:, :, 1::2]])
        else:
            arr = arr[None]
        pdos = arr
        norb = arr.shape[-1]
    return dict(energy=energy, total=total, pdos=pdos, efermi=efermi, natoms=natoms, ispin=ispin, norb=norb)


def orbital_l(norb: int) -> list[int]:
    """Angular momentum of each projected column: LORBIT = 10 (s p d [f]) or 11 (s, 3 p, 5 d [, 7 f])."""
    if norb in (3, 4):
        return list(range(norb))
    if norb in (9, 16):
        return [0] + [1] * 3 + [2] * 5 + ([3] * 7 if norb == 16 else [])
    raise ValueError(f"unexpected number of orbital columns in DOSCAR: {norb}")


def gaussian_smooth(y: np.ndarray, de: float, sigma: float) -> np.ndarray:
    """Extra Gaussian broadening (eV) along the last axis; area-conserving."""
    if sigma <= 0 or de <= 0:
        return y
    half = int(np.ceil(4 * sigma / de))
    g = np.exp(-0.5 * (np.arange(-half, half + 1) * de / sigma) ** 2)
    g /= g.sum()
    return np.apply_along_axis(lambda v: np.convolve(v, g, mode="same"), -1, y)


def project(dos: dict, species: list[str], counts: list[int], tokens: list[str] | None = None):
    """Curves [(label, array(spin, energy))] for each element ('I') or element:orbitals ('Pb:p', 'I:sp')."""
    pd = dos["pdos"]
    if pd is None:
        raise ValueError("the DOSCAR has no projections: rerun with LORBIT = 10 or 11")
    owner = np.repeat(species, counts)
    if len(owner) != pd.shape[1]:
        raise ValueError(f"POSCAR has {len(owner)} atoms but DOSCAR has {pd.shape[1]}")
    lvals = np.array(orbital_l(pd.shape[-1]))
    if not tokens:
        tokens = list(dict.fromkeys(species))
    curves = []
    for tk in tokens:
        el, _, orb = tk.partition(":")
        m = owner == el
        if not m.any():
            _echo(f"  ! element {el} not in the structure; skipped")
            continue
        if orb:
            want = [k for k, v in L_NAMES.items() if v in orb.lower()]
            if not want:
                raise ValueError(f"unknown orbital in '{tk}': use s, p, d, f (e.g. Pb:p or I:sp)")
            y = pd[:, m][:, :, :, np.isin(lvals, want)].sum(axis=(1, 3))
            curves.append((f"{el}-{orb}", y))
        else:
            curves.append((el, pd[:, m].sum(axis=(1, 3))))
    return curves


def integrate(energy: np.ndarray, y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Number of states of ``y`` (…, energy) between ``lo`` and ``hi`` (trapezoid rule)."""
    m = (energy >= lo) & (energy <= hi)
    if m.sum() < 2:
        return np.zeros(y.shape[:-1])
    e, v = energy[m], y[..., m]
    return (0.5 * (v[..., 1:] + v[..., :-1]) * np.diff(e)).sum(axis=-1)


# ----------------------------------------------------------------------------- loading data sets
def load_bandset(folder: Path, user_labels: list[str] | None = None) -> dict:
    """Read one band-structure folder: path, eigenvalues, band edges, method label."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"band folder {folder} does not exist")
    incar, outcar = read_incar(folder), read_outcar(folder)
    if not incar and not outcar["found"]:
        _echo(f"  ! no INCAR or OUTCAR in {folder}: assuming a collinear run")
    ispin, ncl = spin_flags(incar, outcar)
    cell, _, _ = read_poscar(folder)
    evp = _first(folder, "EIGENVAL_OPT", "EIGENVAL.OPT") if _first(folder, "KPOINTS_OPT") else None
    kpp = _first(folder, "KPOINTS_OPT") if evp else _first(folder, "KPOINTS")
    evp = evp or _first(folder, "EIGENVAL")
    if evp is None:
        raise FileNotFoundError(f"no EIGENVAL in {folder}")
    ev = read_eigenval(evp)
    path = build_path(ev, read_kpoints(kpp), cell, user_labels)
    nelect = outcar["nelect"] or ev["nelect"]
    edges = band_edges(path["eig"], path["occ"], nelect, ncl)
    label = method_label(incar, ncl) if incar else folder.name
    return dict(folder=folder, ispin=ev["ispin"], ncl=ncl, label=label, efermi=outcar["efermi"],
                eigenval=evp.name, **edges, **path)


def load_dos(folder: Path, tokens: list[str] | None = None, ref: str = "vbm", smear: float = 0.0) -> dict:
    """Read one DOS folder: total and projected curves, its own VBM/CBM, the energy zero."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"DOS folder {folder} does not exist")
    incar, outcar = read_incar(folder), read_outcar(folder)
    ispin, ncl = spin_flags(incar, outcar)
    _, species, counts = read_poscar(folder)
    dos = read_doscar(folder, ispin if (outcar["ispin"] or "ISPIN" in incar) else None, ncl)
    evp = _first(folder, "EIGENVAL")
    edges = None
    if evp is not None:
        ev = read_eigenval(evp)
        edges = band_edges(ev["eig"], ev["occ"], outcar["nelect"] or ev["nelect"], ncl)
    zero = {"vbm": edges["vbm"] if edges else dos["efermi"], "fermi": dos["efermi"], "none": 0.0}[ref]
    de = float(np.median(np.diff(dos["energy"])))
    curves = []
    if dos["pdos"] is not None:
        curves = [(lab, gaussian_smooth(y, de, smear)) for lab, y in project(dos, species, counts, tokens)]
    total = gaussian_smooth(dos["total"], de, smear)
    return dict(folder=folder, label=method_label(incar, ncl) if incar else folder.name, ncl=ncl,
                edges=edges, zero=zero, de=de, curves=curves, total=total, ref_used=ref if edges or ref != "vbm"
                else "fermi", **{k: dos[k] for k in ("energy", "efermi", "natoms", "ispin", "norb", "pdos")})


def edge_character(d: dict, width: float) -> list[tuple[str, float, float]]:
    """States of each curve within ``width`` eV below the VBM and above the CBM of the DOS run."""
    if d["edges"] is None or not d["curves"]:
        return []
    vbm, cbm = d["edges"]["vbm"], d["edges"]["cbm"]
    return [(lab, float(integrate(d["energy"], y, vbm - width, vbm).sum()),
             float(integrate(d["energy"], y, cbm, cbm + width).sum())) for lab, y in d["curves"]]


def where(bs: dict, i: int) -> str:
    """Name of k-point ``i`` of a band set: the tick label if it sits on one, else its index and x."""
    if i < 0:
        return "?"
    x, ticks = bs["x"], bs["ticks"]
    j = int(np.argmin([abs(x[i] - x[t]) for t in ticks]))
    lab = _plain(bs["tlabels"][j])
    if abs(x[i] - x[ticks[j]]) > 1e-9 or not lab:
        return f"k #{i} (x = {x[i]:.3f} 1/Å)"
    if "|" in lab:                                      # a jump: left or right of the break?
        left, right = lab.split("|", 1)
        b = [k for k in np.flatnonzero(bs["brk"]) if abs(x[k] - x[ticks[j]]) < 1e-9]
        return (right if i >= b[0] else left) if b else lab
    return lab


def map_x(x2, t2, x1, t1):
    """Put a second band set on the x axis of the first (same path, possibly a slightly different cell)."""
    if len(t1) == len(t2):
        xs2 = [x2[t] for t in t2]
        if np.all(np.diff(xs2) > 0):
            return np.interp(x2, xs2, [x1[t] for t in t1])
    return x2 * (x1[-1] / x2[-1] if x2[-1] > 0 else 1.0)


def report_bands(i: int, bs: dict) -> None:
    _echo(f"[bands {i}] {bs['folder']}")
    _echo(f"  {bs['label']}: ISPIN = {bs['ispin']}, non-collinear = {bs['ncl']}, {len(bs['x'])} k-points "
          f"({bs['eigenval']}), {bs['eig'].shape[-1]} bands")
    _echo(f"  path: {' '.join(_plain(t) or '·' for t in bs['tlabels'])}")
    _echo(f"  VBM = {bs['vbm']:.4f} eV at {where(bs, bs['kv'])}; CBM = {bs['cbm']:.4f} eV at {where(bs, bs['kc'])}")
    if bs["gap"] <= 0:
        _echo("  no gap: the conduction band overlaps the valence band (metal or semimetal)")
        return
    direct = np.allclose(bs["kpts"][bs["kv"]] - np.round(bs["kpts"][bs["kv"]] - bs["kpts"][bs["kc"]]),
                         bs["kpts"][bs["kc"]], atol=1e-6)
    line = f"  gap = {bs['gap']:.3f} eV ({'direct' if direct else 'indirect'})"
    if not direct:
        line += f"; smallest direct gap {bs['direct_gap']:.3f} eV at {where(bs, bs['kd'])}"
    if bs["efermi"] is not None:
        line += f"; E_F(OUTCAR) = {bs['efermi']:.4f} eV"
    _echo(line)


def report_dos(d: dict, width: float) -> None:
    lorbit = {3: "10", 4: "10", 9: "11", 16: "11"}.get(d["norb"], "none")
    _echo(f"[dos] {d['folder']}")
    _echo(f"  {d['label']}: ISPIN = {d['ispin']}, non-collinear = {d['ncl']}, LORBIT = {lorbit}, "
          f"NEDOS = {len(d['energy'])}, dE = {d['de'] * 1000:.1f} meV")
    zname = {"vbm": "VBM of the DOS run", "fermi": "E_F of the DOSCAR", "none": "raw energies"}[d["ref_used"]]
    _echo(f"  energy zero = {d['zero']:.4f} eV ({zname})")
    if d["edges"] is not None:
        e = d["edges"]
        _echo(f"  VBM = {e['vbm']:.4f} eV, CBM = {e['cbm']:.4f} eV, gap = {e['gap']:.3f} eV (EIGENVAL of the DOS run)")
    else:
        _echo("  ! no EIGENVAL in the DOS folder: zero = E_F of the DOSCAR, no edge character")
    if d["pdos"] is None:
        _echo("  ! DOSCAR has no projections (LORBIT not set): only the total DOS is drawn")
    if d["de"] > 0.05:
        _echo("  ! coarse DOS grid; raise NEDOS (e.g. 3001) or add --smear")
    rows = edge_character(d, width)
    if rows:
        sv, sc = sum(r[1] for r in rows) or 1.0, sum(r[2] for r in rows) or 1.0
        _echo(f"  character of the band edges (states within {width:g} eV of the VBM / CBM):")
        _echo(f"    {'curve':<10s} {'VB edge':>8s} {'share':>6s} {'CB edge':>8s} {'share':>6s}")
        for lab, v, c in rows:
            _echo(f"    {lab:<10s} {v:8.3f} {100 * v / sv:5.0f}% {c:8.3f} {100 * c / sc:5.0f}%")


# ----------------------------------------------------------------------------- figure
def set_style(fontsize: float) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": fontsize, "axes.labelsize": fontsize + 1, "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize, "legend.fontsize": fontsize, "axes.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "ytick.minor.width": 0.6,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5, "ytick.minor.size": 2.0,
        "mathtext.default": "regular", "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def plot(out: Path, sets: list[dict], dos: dict | None, a) -> list[Path]:
    """Draw the figure and save it; returns the files written."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import AutoMinorLocator, FuncFormatter, NullLocator

    set_style(a.fontsize)
    MM = 1 / 25.4
    if sets and dos:
        fig, (axb, axd) = plt.subplots(1, 2, sharey=True, figsize=(a.width_mm * MM, a.height_mm * MM),
                                       gridspec_kw=dict(width_ratios=[a.ratio, 1], wspace=0.04))
    elif sets:
        fig, axb = plt.subplots(figsize=(a.width_mm * MM * a.ratio / (a.ratio + 1) + 12 * MM, a.height_mm * MM))
        axd = None
    else:
        fig, axd = plt.subplots(figsize=(a.width_mm * MM / (a.ratio + 1) + 14 * MM, a.height_mm * MM))
        axb = None
    ylab = {"vbm": r"E $-$ E$_{\mathrm{VBM}}$ (eV)", "fermi": r"E $-$ E$_{\mathrm{F}}$ (eV)", "none": "E (eV)"}

    if axb is not None:
        handles = []
        for i, bs in enumerate(sets):
            st = BAND_STYLES[i % len(BAND_STYLES)]
            cuts = [0] + list(np.flatnonzero(bs["brk"])) + [len(bs["x"])]
            for s in range(bs["eig"].shape[0]):
                sty = dict(st, ls=st["ls"] if s == 0 else (0, (4, 1.5)))
                for c0, c1 in zip(cuts[:-1], cuts[1:]):
                    axb.plot(bs["x"][c0:c1], bs["eig"][s, c0:c1] - bs["zero"], zorder=2 + i, **sty)
            handles.append(Line2D([], [], **st))
        ref = sets[0]
        for t in ref["ticks"][1:-1]:
            axb.axvline(ref["x"][t], color="0.25", lw=0.6, zorder=1)
        axb.set_xticks([ref["x"][t] for t in ref["ticks"]])
        axb.set_xticklabels(ref["tlabels"])
        axb.tick_params(axis="x", which="both", length=0, pad=4)
        axb.set_xlim(ref["x"][0], ref["x"][-1])
        names = [bs["label"] for bs in sets]
        if any(bs["eig"].shape[0] == 2 for bs in sets):
            handles.append(Line2D([], [], color="k", lw=0.8, ls=(0, (4, 1.5))))
            names.append("spin down")
        axb.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(names), frameon=False,
                   handlelength=2.0, columnspacing=1.6, borderaxespad=0.2)
        axb.set_ylabel(ylab[a.ref])

    if axd is not None:
        e = dos["energy"] - dos["zero"]
        win = (e >= a.emin) & (e <= a.emax)
        two = dos["total"].shape[0] == 2
        ymax = 0.0
        colors = iter(EXTRA)
        handles, names = [], []
        draw_total = a.total or dos["pdos"] is None
        if draw_total:
            for s, sign in zip(range(dos["total"].shape[0]), (1, -1)):
                axd.fill_betweenx(e, 0, sign * dos["total"][s], color="0.85", lw=0, zorder=1)
                axd.plot(sign * dos["total"][s], e, color="0.55", lw=0.7, zorder=1)
            ymax = max(ymax, dos["total"][:, win].max() if win.any() else 0)
            handles.append(Line2D([], [], color="0.55", lw=1.6))
            names.append("total")
        for lab, y in dos["curves"]:
            el = lab.split("-")[0]
            col = ELEMENT_COLORS.get(lab) or (ELEMENT_COLORS.get(el) if lab == el else None) or next(colors)
            for s, sign in zip(range(y.shape[0]), (1, -1)):
                axd.plot(sign * y[s], e, color=col, lw=1.4, zorder=3)
            ymax = max(ymax, y[:, win].max() if win.any() else 0)
            handles.append(Line2D([], [], color=col, lw=1.6))
            names.append(lab)
        top = a.dosmax or (1.05 * ymax if ymax > 0 else 1.0)
        axd.set_xlim(-top if two else 0, top)
        if two:
            axd.axvline(0, color="k", lw=0.6)
        axd.set_xlabel("DOS (states/eV)")
        if axb is not None and not two:                 # no "0" under the last k label
            axd.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "" if abs(v) < 1e-12 else f"{v:g}"))
        axd.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=min(len(names), 5),
                   frameon=False, handlelength=1.2, columnspacing=0.9, handletextpad=0.4, borderaxespad=0.2)
        if axb is None:
            axd.set_ylabel(ylab[a.ref])
        else:
            axd.tick_params(axis="y", labelleft=False)

    for ax in (axb, axd):
        if ax is None:
            continue
        ax.axhline(0, color="0.55", lw=0.9, ls=(0, (4, 2.5)), zorder=0)
        ax.set_ylim(a.emin, a.emax)
        if ax is axd:
            ax.minorticks_on()
        else:
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.xaxis.set_minor_locator(NullLocator())

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.90)
    suffix = out.suffix.lower()
    targets = [out] if suffix in (".png", ".pdf", ".svg", ".jpg", ".jpeg", ".tif", ".tiff") else \
        [out.with_name(out.name + ".png"), out.with_name(out.name + ".pdf")]
    for t in targets:
        fig.savefig(t, dpi=a.dpi, bbox_inches="tight")
    plt.close(fig)
    return targets


def save_data(base: Path, sets: list[dict], dos: dict | None) -> list[Path]:
    """CSV of every plotted curve, on the energy scale of the figure."""
    written = []
    for i, bs in enumerate(sets):
        p = base.with_name(f"{base.name}_bands_{i + 1}.csv")
        nspin, nb = bs["eig"].shape[0], bs["eig"].shape[-1]
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["x_inv_A"] + [f"s{s + 1}_b{b + 1}" for s in range(nspin) for b in range(nb)])
            for k in range(len(bs["x"])):
                w.writerow([f"{bs['x'][k]:.6f}"] + [f"{v:.5f}" for v in (bs["eig"][:, k] - bs["zero"]).ravel()])
        written.append(p)
    if dos:
        p = base.with_name(f"{base.name}_dos.csv")
        tot = dos["total"]
        cols = [("total" if tot.shape[0] == 1 else f"total_s{s + 1}", tot[s]) for s in range(tot.shape[0])]
        cols += [(lab if y.shape[0] == 1 else f"{lab}_s{s + 1}", y[s]) for lab, y in dos["curves"]
                 for s in range(y.shape[0])]
        np.savetxt(p, np.column_stack([dos["energy"] - dos["zero"]] + [c for _, c in cols]), delimiter=",",
                   header=",".join(["E"] + [n for n, _ in cols]), comments="", fmt="%.5f", encoding="utf-8")
        written.append(p)
    return written


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bands", nargs="+", type=Path, default=[], help="band-structure run folder(s); the next ones are overlaid")
    ap.add_argument("--dos", type=Path, help="DOS run folder")
    ap.add_argument("-o", "--out", default="bands_dos",
                    help="output name; without a suffix both .png and .pdf are written (default bands_dos)")
    ap.add_argument("--emin", type=float, default=-4.0, help="lower energy limit (eV, default -4)")
    ap.add_argument("--emax", type=float, default=4.0, help="upper energy limit (eV, default 4)")
    ap.add_argument("--ref", choices=["vbm", "fermi", "none"], default="vbm", help="energy zero of every data set")
    ap.add_argument("--names", nargs="+", help="legend names of the band sets (default: from INCAR)")
    ap.add_argument("--labels", nargs="+", help="k-point labels at the ticks, e.g. G X S Y G (A|B for a jump)")
    ap.add_argument("--proj", nargs="+", help="DOS curves: elements or element:orbitals, e.g. Pb I C or Pb:p I:p")
    ap.add_argument("--total", action="store_true", help="also draw the total DOS (grey, filled)")
    ap.add_argument("--smear", type=float, default=0.0, help="extra Gaussian broadening of the DOS (eV)")
    ap.add_argument("--dosmax", type=float, help="upper limit of the DOS axis")
    ap.add_argument("--edge-window", type=float, default=1.0,
                    help="width (eV) below the VBM / above the CBM for the edge-character report (default 1)")
    ap.add_argument("--width-mm", type=float, default=120.0, help="figure width (mm, default 120)")
    ap.add_argument("--height-mm", type=float, default=85.0, help="figure height (mm, default 85)")
    ap.add_argument("--fontsize", type=float, default=8.0)
    ap.add_argument("--ratio", type=float, default=1.6, help="width of the band panel / width of the DOS panel")
    ap.add_argument("--dpi", type=int, default=600, help="resolution of raster output (default 600)")
    ap.add_argument("--save-data", action="store_true", help="write the plotted curves as CSV next to the figure")
    ap.add_argument("--no-plot", action="store_true", help="report (and --save-data) only, no figure")
    ap.add_argument("--version", action="version", version=__version__)
    a = ap.parse_args(argv)
    if not a.bands and not a.dos:
        ap.error("give --bands and/or --dos")

    sets = []
    for i, f in enumerate(a.bands):
        bs = load_bandset(f, a.labels)
        bs["zero"] = {"vbm": bs["vbm"], "fermi": bs["efermi"] if bs["efermi"] is not None else bs["vbm"],
                      "none": 0.0}[a.ref]
        if a.names and i < len(a.names):
            bs["label"] = a.names[i]
        sets.append(bs)
        report_bands(i + 1, bs)
    for bs in sets[1:]:
        if len(bs["ticks"]) != len(sets[0]["ticks"]):
            _echo("  ! the band sets have different paths; the second one is scaled to the length of the first")
        bs["x"] = map_x(bs["x"], bs["ticks"], sets[0]["x"], sets[0]["ticks"])

    dos = None
    if a.dos:
        dos = load_dos(a.dos, a.proj, a.ref, a.smear)
        report_dos(dos, a.edge_window)

    out = Path(a.out)
    if not a.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        for p in plot(out, sets, dos, a):
            _echo(f"saved {p}")
    if a.save_data:
        base = out.with_suffix("") if out.suffix.lower() in (".png", ".pdf", ".svg", ".jpg", ".jpeg") else out
        for p in save_data(base, sets, dos):
            _echo(f"saved {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
