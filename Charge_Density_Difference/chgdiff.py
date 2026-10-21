#!/usr/bin/env python3
"""
chgdiff.py — charge-density difference of two fragments from three VASP CHGCARs, with the
plane-averaged profile and the charge-displacement curve.

Background
----------
Where does the electron density go when two fragments A and B are put together? Compute the
self-consistent density of the whole system (AB) and of each fragment alone (A, B), with the
fragment atoms at exactly the same positions, in the same cell and on the same FFT grid. Then

    drho(r) = rho_AB(r) - rho_A(r) - rho_B(r)

is positive where electrons accumulate on binding and negative where they are depleted. Averaged
over the planes normal to an axis and integrated over the plane it gives the line density
lambda(z) (e/Å); its running integral

    dQ(z) = integral from z0 to z of lambda(z') dz'          (z0 in the vacuum, where drho = 0)

is the charge-displacement curve: dQ at a plane between A and B is the number of electrons that
crossed that plane on binding (dQ > 0: electrons moved towards -z, into the region below it).

A CHGCAR stores rho(r) * V_cell (electrons), not rho. The difference file CHGDIFF.vasp keeps that
convention so that VESTA reads it directly; every profile printed here is converted to e/Å^3
(value / V_cell) or e/Å (plane integral per unit length). Only the first data block of each file is
read: the total density. Augmentation occupancies and the magnetization blocks of spin-polarized
and non-collinear CHGCARs are skipped, and the grid is streamed slice by slice, so multi-GB files
need little memory.

Usage
-----
    python chgdiff.py AB/CHGCAR A/CHGCAR B/CHGCAR                       # check, write CHGDIFF.vasp, summary
    python chgdiff.py AB/CHGCAR A/CHGCAR B/CHGCAR --plot edd.png --csv edd.csv --cube edd.cube
    python chgdiff.py AB/CHGCAR A/CHGCAR B/CHGCAR --check-only           # are the three files compatible?
    python chgdiff.py AB/CHGCAR A/CHGCAR B/CHGCAR --plane 9.5 --between 1 2
    python chgdiff.py CHGDIFF.vasp --plot edd.png                        # re-analyse an existing difference

Author: Israel C. Ribeiro — https://github.com/israel-c-ribeiro/VASP_tools (MIT licence)
"""
from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path

import numpy as np

__version__ = "1.1.0"

BOHR = 0.529177210903                       # Å
COULOMB = 14.399645                         # e^2 / (4 pi eps0), eV·Å
DEBYE_PER_EA = 4.80320                      # 1 e·Å in debye
ELEMENTS = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge "
            "As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm "
            "Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U "
            "Np Pu Am Cm Bk Cf Es Fm Md No Lr").split()


# ----------------------------------------------------------------------------------------------- reading
def read_header(fh) -> dict:
    """Parse the POSCAR-like header of a CHGCAR/CHG/LOCPOT up to and including the grid line.

    ``fh`` needs only a ``readline`` returning str. Returns comment, cell (Å), names, counts,
    symbols, fractional positions and the grid shape (nx, ny, nz)."""
    def line():
        ln = fh.readline()
        if not ln:
            raise ValueError("file ends inside the header (not a CHGCAR?)")
        return ln
    comment = line().strip()
    try:
        scale = float(line().split()[0])
        cell = np.array([[float(x) for x in line().split()[:3]] for _ in range(3)])
    except (ValueError, IndexError):
        raise ValueError("lines 2-5 are not a scale factor and three lattice vectors (not a CHGCAR?)") from None
    if scale < 0:                                   # negative scale = target volume
        scale = (-scale / abs(np.linalg.det(cell))) ** (1 / 3)
    cell = cell * scale
    tok = line().split()
    if all(t.isdigit() for t in tok):
        raise ValueError("no species line (VASP 4 format): add the element names on line 6")
    names = [t.split("/")[0].split("_")[0] for t in tok]
    counts = [int(x) for x in line().split()]
    if len(counts) != len(names):
        raise ValueError(f"{len(names)} species names but {len(counts)} counts in the header")
    mode = line().strip()
    if mode[:1] in "Ss":                            # selective dynamics
        mode = line().strip()
    nat = sum(counts)
    pos = np.array([[float(x) for x in line().split()[:3]] for _ in range(nat)]).reshape(nat, 3)
    frac = pos if mode[:1] in "Dd" else (pos * scale) @ np.linalg.inv(cell)
    ln = line()
    while not ln.strip():
        ln = line()
    shape = tuple(int(x) for x in ln.split()[:3])
    if len(shape) != 3:
        raise ValueError(f"expected the grid line 'NGX NGY NGZ' after the positions, got: {ln.strip()!r}")
    symbols = [n for n, c in zip(names, counts) for _ in range(c)]
    return dict(comment=comment, cell=cell, names=names, counts=counts, symbols=symbols,
                frac=frac % 1.0, shape=shape)


class _Text:
    """Text ``readline`` on top of a binary file (the grid itself is read as bytes)."""
    def __init__(self, f):
        self.f = f

    def readline(self):
        return self.f.readline().decode("ascii", "replace")


class ChgcarStream:
    """Open a CHGCAR, parse its header, and yield the first (total-density) grid one xy slice at a time.

    Each slice is an (ny, nx) array of the raw file values (rho * V_cell): x runs fastest in the file."""

    def __init__(self, path, chunk_bytes: int = 32 * 1024 * 1024):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"{self.path}: no such file")
        self.fh = open(self.path, "rb")
        try:
            self.header = read_header(_Text(self.fh))
        except ValueError as exc:
            self.fh.close()
            raise ValueError(f"{self.path}: {exc}") from None
        self.chunk = chunk_bytes
        self.volume = float(abs(np.linalg.det(self.header["cell"])))

    def slices(self):
        nx, ny, nz = self.header["shape"]
        per = nx * ny
        buf, carry, k = np.empty(0), b"", 0
        while k < nz:
            raw = self.fh.read(self.chunk)
            last = not raw
            raw = carry + raw
            # after the density block a CHGCAR continues with text ("augmentation occupancies ...")
            m = re.search(rb"[A-DF-Za-df-z*]", raw)
            if m:
                if raw[m.start():m.start() + 1] == b"*":
                    raise ValueError(f"{self.path}: a grid value overflowed its Fortran field ('***')")
                raw, last = raw[:m.start()], True
            cut = len(raw) if last else max(raw.rfind(b" "), raw.rfind(b"\n"))   # never cut a number
            carry = raw[cut:] if cut > 0 else raw
            if cut > 0:
                try:
                    vals = np.array(raw[:cut].split(), dtype=float)
                except ValueError as exc:
                    raise ValueError(f"{self.path}: unreadable grid value ({exc})") from None
                buf = np.concatenate([buf, vals])
            while len(buf) >= per and k < nz:
                yield buf[:per].reshape(ny, nx)
                buf = buf[per:]
                k += 1
            if last:
                break
        if k < nz:
            raise ValueError(f"{self.path}: grid truncated ({k} of {nz} slices, {k * per + len(buf)} of "
                             f"{nz * per} values)")

    def close(self):
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_chgcar(path):
    """Whole first grid of a (small) CHGCAR as an (nx, ny, nz) array of rho * V_cell, plus its header."""
    with ChgcarStream(path) as s:
        grid = np.stack(list(s.slices()), axis=-1)            # (ny, nx, nz)
        return grid.transpose(1, 0, 2), s.header


# ----------------------------------------------------------------------------------------------- checks
def _mic_distances(f1, f2, cell):
    d = f1[:, None, :] - f2[None, :, :]
    d -= np.round(d)
    return np.linalg.norm(d @ cell, axis=-1)


def match_atoms(ab: dict, frags: dict, tol: float = 1e-3):
    """Assign every atom of each fragment to one atom of AB (same element, same position within ``tol`` Å).

    Returns (labels, problems, max_deviation): ``labels[i]`` is the fragment name of AB atom i
    (None if no fragment claims it)."""
    sym_ab = np.array(ab["symbols"])
    labels = [None] * len(sym_ab)
    problems, dev = [], 0.0
    for name, h in frags.items():
        if len(h["symbols"]) == 0:
            continue
        d = _mic_distances(h["frac"], ab["frac"], ab["cell"])
        for i, s in enumerate(h["symbols"]):
            row = np.where((sym_ab == s) & np.array([lab is None for lab in labels]), d[i], np.inf)
            j = int(np.argmin(row))
            if row[j] <= tol:
                labels[j] = name
                dev = max(dev, float(row[j]))
                continue
            k = int(np.argmin(d[i]))
            what = (f"closest AB atom {k + 1} ({sym_ab[k]}) at {d[i, k]:.3f} Å" if np.isfinite(d[i, k])
                    else "AB has no atom")
            if d[i, k] <= tol and sym_ab[k] != s:
                what += ": same place, different element"
            elif d[i, k] <= tol:
                what += ": already taken by another fragment atom"
            problems.append(f"{name} atom {i + 1} ({s}) has no partner in AB within {tol} Å — {what}")
    for j, lab in enumerate(labels):
        if lab is None:
            problems.append(f"AB atom {j + 1} ({sym_ab[j]}) belongs to no fragment")
    return labels, problems, dev


def check_compatible(ab: dict, frags: dict, tol: float = 1e-3, cell_tol: float = 1e-4):
    """List every difference that makes AB - A - B meaningless (empty list = compatible)."""
    problems = []
    for name, h in frags.items():
        if h["shape"] != ab["shape"]:
            problems.append(f"grid of {name} is {' x '.join(map(str, h['shape']))}, AB is "
                            f"{' x '.join(map(str, ab['shape']))} — set the same NGXF NGYF NGZF in all three INCARs")
        dc = np.abs(h["cell"] - ab["cell"]).max()
        if dc > cell_tol:
            la, lb = np.linalg.norm(h["cell"], axis=1), np.linalg.norm(ab["cell"], axis=1)
            problems.append(f"cell of {name} differs from AB by up to {dc:.4f} Å (lengths "
                            f"{' '.join(f'{x:.4f}' for x in la)} vs {' '.join(f'{x:.4f}' for x in lb)})")
    if not problems:                                   # positions only make sense in the same cell
        labels, prob, dev = match_atoms(ab, frags, tol)
        problems += prob
    n_ab, n_fr = len(ab["symbols"]), sum(len(h["symbols"]) for h in frags.values())
    if n_ab != n_fr:
        problems.insert(0, f"AB has {n_ab} atoms, the fragments together {n_fr}")
    return problems


# ----------------------------------------------------------------------------------------------- writing
class _GridWriter:
    """Write grid values 5 per line (CHGCAR style) from a stream of arrays."""

    def __init__(self, fh, per_line: int = 5):
        self.fh, self.per, self.pending = fh, per_line, np.empty(0)

    def add(self, vals):
        p = np.concatenate([self.pending, np.ravel(vals)])
        m = len(p) // self.per * self.per
        if m:
            np.savetxt(self.fh, p[:m].reshape(-1, self.per), fmt="%18.11E", delimiter=" ")
        self.pending = p[m:]

    def close(self):
        if len(self.pending):
            self.fh.write(" ".join(f"{v:18.11E}" for v in self.pending) + "\n")
        self.pending = np.empty(0)


def header_text(h: dict, comment: str | None = None) -> str:
    """POSCAR-style header + blank line + grid line, as in a CHGCAR (direct coordinates, scale 1)."""
    lines = [comment or h["comment"], "   1.00000000000000"]
    lines += ["  " + "  ".join(f"{x:12.6f}" for x in v) for v in h["cell"]]
    lines += ["   " + "   ".join(h["names"]), "   " + "   ".join(str(c) for c in h["counts"]), "Direct"]
    lines += ["  " + "  ".join(f"{x:.6f}" for x in f) for f in h["frac"]]
    lines += ["", "  " + "  ".join(f"{n:4d}" for n in h["shape"])]
    return "\n".join(lines) + "\n"


def write_cube(path, h: dict, drho):
    """Gaussian cube of drho (input in e/Å^3, written in e/bohr^3, lengths in bohr)."""
    nat = len(h["symbols"])
    cart = h["frac"] @ h["cell"] / BOHR
    with open(path, "w", newline="\n") as f:
        f.write("charge-density difference AB - A - B (chgdiff.py)\nunits: e/bohr^3, bohr\n")
        f.write(f"{nat:5d} {0.0:12.6f} {0.0:12.6f} {0.0:12.6f}\n")
        for n, v in zip(h["shape"], h["cell"]):
            f.write(f"{n:5d} " + " ".join(f"{x:12.6f}" for x in v / n / BOHR) + "\n")
        for s, p in zip(h["symbols"], cart):
            z = ELEMENTS.index(s) if s in ELEMENTS else 0
            f.write(f"{z:5d} {float(z):12.6f} " + " ".join(f"{x:12.6f}" for x in p) + "\n")
        vals = (np.asarray(drho) * BOHR ** 3).reshape(-1)       # x outer, z inner = C order of (nx, ny, nz)
        nz = h["shape"][2]
        for row in vals.reshape(-1, nz):
            for i in range(0, nz, 6):
                f.write(" ".join(f"{v:13.5E}" for v in row[i:i + 6]) + "\n")


# ----------------------------------------------------------------------------------------------- core
def layer_height(cell, axis: int = 2) -> float:
    """Distance between lattice planes normal to ``axis`` (Å): the length of the profile."""
    other = [cell[a] for a in range(3) if a != axis]
    return float(abs(np.linalg.det(cell)) / np.linalg.norm(np.cross(*other)))


def difference(paths, out=None, keep_grid: bool = False, scales=(1.0, -1.0, -1.0)):
    """Stream the files in ``paths`` in lockstep and form sum(scale_i * value_i) slice by slice.

    With three paths (AB, A, B) and the default scales this is AB - A - B; with one path it simply
    reads that file (e.g. an existing CHGDIFF.vasp). Writes the result in CHGCAR format to ``out``
    (if given). Returns a dict with the header of the first file, the volume, the plane sums of the
    result and of every input along each grid axis, the electron count of every input, and the full
    difference grid in e/Å^3 if ``keep_grid``."""
    streams = [ChgcarStream(p) for p in paths]
    try:
        h = streams[0].header
        nx, ny, nz = h["shape"]
        vol = streams[0].volume
        n_in = len(streams)
        sums = [[np.zeros(nx), np.zeros(ny), np.zeros(nz)] for _ in range(n_in + 1)]   # inputs ..., result
        full = np.empty((nx, ny, nz)) if keep_grid else None
        fo = writer = None
        if out is not None:
            fo = open(out, "w", newline="\n")
            fo.write(header_text(h, "charge-density difference AB - A - B (chgdiff.py); values = drho * V_cell"
                                 if n_in == 3 else None))
            writer = _GridWriter(fo)
        k, dmax = 0, 0.0
        for sl in zip(*(s.slices() for s in streams)):
            d = sum(c * x for c, x in zip(scales, sl)) if n_in > 1 else sl[0]
            dmax = max(dmax, float(np.abs(d).max()))
            for i, x in enumerate(list(sl) + [d]):
                sums[i][0] += x.sum(axis=0)
                sums[i][1] += x.sum(axis=1)
                sums[i][2][k] = x.sum()
            if writer:
                writer.add(d)
            if keep_grid:
                full[:, :, k] = d.T / vol
            k += 1
        if writer:
            writer.close()
            fo.close()
    finally:
        for s in streams:
            s.close()
    ntot = nx * ny * nz
    return dict(header=h, volume=vol, sums=sums, electrons=[float(s[2].sum() / ntot) for s in sums],
                grid=full, max_abs=dmax / vol)


def heights(h: dict, axis: int = 2):
    """Height of every atom along the normal to the plane of the other two cell vectors (Å, in [0, L))."""
    return (h["frac"][:, axis] % 1.0) * layer_height(h["cell"], axis)


def vacuum_origin(z_atoms, length):
    """Middle of the largest atom-free gap along the normal, and the gap width."""
    zs = np.sort(np.asarray(z_atoms) % length)
    if len(zs) == 0:
        return 0.0, length
    gaps = np.diff(np.r_[zs, zs[0] + length])
    k = int(np.argmax(gaps))
    return float((zs[k] + gaps[k] / 2) % length), float(gaps[k])


def profile(res: dict, axis: int = 2, origin: float | None = None):
    """Plane-averaged difference along ``axis`` and the charge-displacement curve.

    Returns x (Å, a window of one period starting at the origin, which lies in the vacuum),
    drho_avg (e/Å^3), lam (e/Å, plane integral), dQ (e, running integral from the origin),
    the line densities of every input (e/Å), the origin and the window start."""
    h = res["header"]
    n = h["shape"][axis]
    L = layer_height(h["cell"], axis)
    dz = L / n
    za = heights(h, axis)
    z0 = vacuum_origin(za, L)[0] if origin is None else float(origin) % L
    start = z0 if z0 < L / 2 else z0 - L                        # keeps the window near [0, L)
    z = np.arange(n) * dz
    x = start + (z - start) % L
    order = np.argsort(x)
    npl = int(np.prod(h["shape"])) // n                         # grid points per plane
    lams = [s[axis] / npl / L for s in res["sums"]]            # mean of rho*V over the plane / L = e/Å
    lam = lams[-1][order]
    x = x[order]
    dQ = np.cumsum(lam) * dz - 0.5 * lam * dz                  # midpoint rule, 0 at the origin
    return dict(x=x, lam=lam, drho_avg=lam * L / res["volume"], dQ=dQ, inputs=[v[order] for v in lams[:-1]],
                origin=z0, start=start, length=L, dz=dz, z_atoms=start + (za - start) % L)


def charge_at(prof: dict, z: float) -> float:
    """dQ at a plane (cell-frame height z in Å): electrons that crossed it towards -z."""
    zw = prof["start"] + (z - prof["start"]) % prof["length"]
    return float(np.interp(zw, prof["x"], prof["dQ"]))


def fragment_boundaries(z_atoms, labels):
    """Mid-planes between consecutive atom layers that belong to different fragments (window frame)."""
    idx = [i for i, lab in enumerate(labels) if lab is not None]
    order = sorted(idx, key=lambda i: z_atoms[i])
    out = []
    for i, j in zip(order[:-1], order[1:]):
        if labels[i] != labels[j]:
            out.append((float(0.5 * (z_atoms[i] + z_atoms[j])), f"{labels[i]}|{labels[j]} boundary"))
    return out


def dipole(prof: dict):
    """Dipole of the rearrangement per cell along +normal (e·Å; electrons are negative) and the
    resulting step of the vacuum level, top minus bottom (eV) — exact only if the integral is zero."""
    xc = 0.5 * (prof["z_atoms"].min() + prof["z_atoms"].max()) if len(prof["z_atoms"]) else prof["x"].mean()
    mu = -float(np.sum((prof["x"] - xc) * prof["lam"]) * prof["dz"])
    return mu, xc


def mirror_asymmetry(prof: dict, center: float) -> float:
    """0 = lambda(z) mirror-symmetric about ``center``, 1 = fully antisymmetric/one-sided."""
    x, lam, L = prof["x"], prof["lam"], prof["length"]
    xm = prof["start"] + (2 * center - x - prof["start"]) % L
    xp = np.r_[x - L, x, x + L]
    mir = np.interp(xm, xp, np.r_[lam, lam, lam])
    den = np.abs(lam).sum() + np.abs(mir).sum()
    return float(np.abs(lam - mir).sum() / den) if den > 0 else 0.0


# ----------------------------------------------------------------------------------------------- plot
def plot(path, prof: dict, planes, symbols, labels, axis_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, lam, dq = prof["x"], prof["lam"], prof["dQ"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.8, 4.4), sharex=True, gridspec_kw=dict(height_ratios=[1.15, 1]))
    a1.fill_between(x, 0, lam, where=lam >= 0, interpolate=True, color="#0072B2", alpha=0.35, lw=0, label="electron gain")
    a1.fill_between(x, 0, lam, where=lam < 0, interpolate=True, color="#D55E00", alpha=0.35, lw=0, label="electron loss")
    a1.plot(x, lam, color="0.15", lw=0.9, label=r"$\Delta\lambda(z)$")
    a1.axhline(0, color="0.6", lw=0.6)
    span = max(np.abs(lam).max(), 1e-12)
    a1.set_ylim(-1.45 * span, 1.25 * span)
    groups = ([(lab, np.array([l_ == lab for l_ in labels])) for lab in dict.fromkeys(l_ for l_ in labels if l_)]
              if labels and any(labels) else
              [(e, np.array(symbols) == e) for e in dict.fromkeys(symbols)])
    for (name, m), c in zip(groups, itertools.cycle(["#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#000000"])):
        a1.plot(prof["z_atoms"][m], np.full(m.sum(), -1.25 * span), "|", ms=9, mew=1.6, color=c,
                label=f"{name} atoms")
    a1.set_ylabel(r"$\Delta\lambda$ (e/Å)")
    a2.plot(x, dq, color="#0072B2", lw=1.5, label=r"$\Delta Q(z)$")
    a2.axhline(0, color="0.6", lw=0.6)
    for zp, name, q in planes:
        for ax in (a1, a2):
            ax.axvline(zp, color="0.35", ls="--", lw=0.9)
        a2.plot([zp], [q], "o", color="#D55E00", ms=4, zorder=3)
        a2.annotate(f"{q:+.3f} e", (zp, q), xytext=(5, 4), textcoords="offset points", fontsize=7.5, color="#D55E00")
    a2.set_ylabel(r"$\Delta Q$ (e)")
    a2.set_xlabel(f"Position along {axis_name} (Å)")
    lo, hi = min(dq.min(), 0.0), max(dq.max(), 0.0)
    pad = 0.18 * max(hi - lo, 1e-12)
    a2.set_ylim(lo - pad, hi + 1.6 * pad)
    a2.set_xlim(x[0], x[-1])
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"figure -> {path}")


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="AB A B CHGCARs (whole system, fragment A, fragment B), "
                                             "or one existing difference file")
    ap.add_argument("-o", "--out", default="CHGDIFF.vasp", help="difference file in CHGCAR format (default CHGDIFF.vasp)")
    ap.add_argument("--no-write", action="store_true", help="do not write the difference file")
    ap.add_argument("--cube", help="also write a Gaussian cube of drho (e/bohr^3; holds the grid in memory)")
    ap.add_argument("--check-only", action="store_true", help="only check that the three CHGCARs are compatible")
    ap.add_argument("--allow-mismatch", action="store_true", help="continue (with warnings) if the atoms do not match")
    ap.add_argument("--tol", type=float, default=1e-3, help="position tolerance (Å) for matching atoms (default 1e-3)")
    ap.add_argument("--axis", type=int, default=2, choices=[0, 1, 2], help="profile direction: 0=a, 1=b, 2=c")
    ap.add_argument("--origin", type=float, help="start of the dQ integration (Å; default: middle of the vacuum)")
    ap.add_argument("--plane", type=float, nargs="+", default=[], help="heights (Å, cell frame) where dQ is reported")
    ap.add_argument("--between", type=int, nargs=2, action="append", default=[], metavar=("I", "J"),
                    help="report dQ at the midpoint between atoms I and J (1-based, AB numbering); repeatable")
    ap.add_argument("--plot", help="save the profile figure (PNG/PDF/SVG)")
    ap.add_argument("--csv", help="save z, drho_avg, lambda, dQ and the input line densities")
    ap.add_argument("--version", action="version", version=__version__)
    a = ap.parse_args(argv)

    if len(a.files) not in (1, 3):
        print("give three files (AB A B) or one difference file", file=sys.stderr)
        return 1
    try:
        heads = []
        for p in a.files:
            with ChgcarStream(p) as s:
                heads.append(s.header)
    except (OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    ab = heads[0]
    labels = None
    for tag, p, h in zip(("AB", "A", "B") if len(heads) == 3 else ("in",), a.files, heads):
        print(f"{tag:<3s} {p}: grid {' x '.join(map(str, h['shape']))}, {len(h['symbols'])} atoms")
    if len(heads) == 3:
        frags = {"A": heads[1], "B": heads[2]}
        problems = check_compatible(ab, frags, a.tol)
        if problems:
            print("[FAIL] the three CHGCARs are not compatible:" if not a.allow_mismatch else "[WARN] mismatch:",
                  file=sys.stderr)
            for msg in problems:
                print(f"  - {msg}", file=sys.stderr)
            if not a.allow_mismatch or any(m.startswith(("grid", "cell")) for m in problems):
                return 1
        labels, _, dev = match_atoms(ab, frags, a.tol)
        if not problems:
            print(f"compatible: same cell and grid; AB atoms = A ({len(heads[1]['symbols'])}) + "
                  f"B ({len(heads[2]['symbols'])}), largest position difference {dev:.1e} Å")
        if a.check_only:
            return 0
    elif a.check_only:
        print("--check-only needs three files", file=sys.stderr)
        return 1

    out = None if a.no_write else Path(a.out)
    try:
        res = difference(a.files, out=out, keep_grid=bool(a.cube))
    except ValueError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    e = res["electrons"]
    if len(heads) == 3:
        print(f"electrons: AB {e[0]:.4f}, A {e[1]:.4f}, B {e[2]:.4f}")
        integ = e[3]
        print(f"integral of drho = N_AB - N_A - N_B = {integ:+.5f} e"
              + ("" if abs(integ) < 1e-3 * max(e[0], 1) else "   <-- not zero: charged fragments or different NELECT?"))
    else:
        integ = e[-1]
        print(f"integral of the grid = {integ:+.5f} e" + ("   <-- not a difference density?" if abs(integ) > 0.05 else ""))
    if out is not None:
        print(f"difference -> {out}  (CHGCAR format: drho * V_cell, as VESTA expects)")
    if a.cube:
        write_cube(a.cube, ab, res["grid"])
        print(f"cube -> {a.cube}  (e/bohr^3)")

    prof = profile(res, a.axis, a.origin)
    L = prof["length"]
    gmax = res["max_abs"]
    print(f"max |drho| on the 3D grid {gmax:.3e} e/Å^3 = {gmax * BOHR ** 3:.3e} e/bohr^3 "
          f"(isosurface levels of ~5-10 % of it show the main features)")
    print(f"profile along {'abc'[a.axis]}: {L:.3f} Å, {len(prof['x'])} points; dQ integrated from "
          f"z0 = {prof['origin']:.3f} Å" + (" (middle of the vacuum)" if a.origin is None else ""))
    print(f"max |lambda| {np.abs(prof['lam']).max():.4f} e/Å, "
          f"max |plane-averaged drho| {np.abs(prof['drho_avg']).max():.3e} e/Å^3")
    print(f"dQ at the end of the window {prof['dQ'][-1] + 0.5 * prof['lam'][-1] * prof['dz']:+.5f} e (= integral)")
    mu, xc = dipole(prof)
    area = res["volume"] / L
    print(f"dipole of drho {mu:+.4f} e·Å ({mu * DEBYE_PER_EA:+.3f} D) per cell -> vacuum-level step "
          f"(top - bottom) {-4 * np.pi * COULOMB * mu / area:+.3f} eV")
    asym = mirror_asymmetry(prof, xc)
    print(f"mirror symmetry about the slab centre ({xc:.3f} Å): "
          + ("symmetric" if asym < 0.1 else "asymmetric") + f" (index {asym:.2f}; 0 = symmetric)")

    planes = []
    if labels is not None:
        planes += fragment_boundaries(prof["z_atoms"], labels)
    for i, j in a.between:
        n = len(ab["symbols"])
        if not (1 <= i <= n and 1 <= j <= n):
            print(f"[FAIL] --between {i} {j}: atom indices run from 1 to {n}", file=sys.stderr)
            return 1
        planes.append((float(0.5 * (prof["z_atoms"][i - 1] + prof["z_atoms"][j - 1])),
                       f"between {ab['symbols'][i - 1]}{i} and {ab['symbols'][j - 1]}{j}"))
    planes += [(prof["start"] + (z - prof["start"]) % L, "--plane") for z in a.plane]
    rows = [(zp, name, charge_at(prof, zp)) for zp, name in planes]
    if rows:
        print(f"{'plane (Å)':>10s}  {'where':<26s} {'dQ (e)':>9s}")
        for zp, name, q in rows:
            print(f"{zp:10.3f}  {name:<26s} {q:+9.4f}")
        print("dQ > 0: electrons crossed the plane towards -z (into the region below it)")
    if a.csv:
        cols = [prof["x"], prof["drho_avg"], prof["lam"], prof["dQ"]] + prof["inputs"]
        names = ["z_A", "drho_avg_e_A3", "lambda_e_A", "dQ_e"] + (
            ["lambda_AB_e_A", "lambda_A_e_A", "lambda_B_e_A"] if len(heads) == 3 else [])
        np.savetxt(a.csv, np.column_stack(cols[:len(names)]), delimiter=",", fmt="%.8g",
                   header=",".join(names), comments="")
        print(f"profile -> {a.csv}")
    if a.plot:
        plot(a.plot, prof, rows, ab["symbols"], labels, "abc"[a.axis])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
