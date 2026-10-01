#!/usr/bin/env python3
"""
planar_average.py — planar and macroscopic averages of a VASP LOCPOT, vacuum level and work function.

Background
----------
LOCPOT stores a potential on the 3D FFT grid. Averaging it over the planes parallel to a
surface gives the planar average V(z); averaging V(z) once more over a window equal to the
period of the structure (e.g. the interlayer distance) removes the atomic oscillations and
gives the macroscopic average, the quantity used for band offsets at interfaces.
In the vacuum region of a slab V(z) is flat: that plateau is the vacuum level E_vac, and

    work function   Phi = E_vac - E_F

Write the electrostatic (ionic + Hartree) potential with LVHAR = .TRUE. in the INCAR.
(LVTOT = .TRUE. also adds the exchange-correlation potential, which decays slowly into the
vacuum and is not the right quantity for a vacuum level.)

Usage
-----
    python planar_average.py LOCPOT                          # summary on screen
    python planar_average.py run/LOCPOT --outcar run/OUTCAR --plot pot.png --csv pot.csv
    python planar_average.py LOCPOT --axis 2 --macro 6.4     # macroscopic average, window 6.4 Å

Author: Israel C. Ribeiro — https://github.com/israel-c-ribeiro/VASP_tools (MIT licence)
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

__version__ = "1.0.0"


def read_locpot(path):
    """Return (grid (nx, ny, nz), cell 3x3 Å, symbols, cartesian positions).

    Reads only the first data block (non-collinear LOCPOTs carry four)."""
    with open(path) as f:
        head = [next(f) for _ in range(6)]
        scale = float(head[1].split()[0])
        cell = np.array([[float(t) for t in head[i].split()[:3]] for i in (2, 3, 4)]) * scale
        names = head[5].split()
        ln = head[5]
        if not all(t.isdigit() for t in names):
            ln = next(f)
        else:
            names = [f"X{i}" for i in range(len(names))]
        counts = [int(x) for x in ln.split()]
        symbols = [n for n, c in zip(names, counts) for _ in range(c)]
        ln = next(f)
        if ln.strip()[:1] in "Ss":
            ln = next(f)
        direct = ln.strip()[:1] in "Dd"
        coords = np.array([[float(t) for t in next(f).split()[:3]] for _ in symbols])
        ln = next(f)
        while not ln.strip():
            ln = next(f)
        nx, ny, nz = (int(x) for x in ln.split())
        n = nx * ny * nz
        first = next(f)
        per = len(first.split())
        txt = first + "".join(itertools.islice(f, -(-n // per) - 1))
    vals = np.array(txt.split()[:n], dtype=float)
    if vals.size < n:
        raise ValueError(f"grid truncated ({vals.size} of {n} values)")
    cart = coords @ cell if direct else coords * scale
    return vals.reshape((nz, ny, nx)).T, cell, symbols, cart


def layer_height(cell, axis):
    """Distance between lattice planes normal to ``axis`` (Å): the length of the z axis of the profile."""
    other = [cell[a] for a in range(3) if a != axis]
    return abs(np.linalg.det(cell)) / np.linalg.norm(np.cross(*other))


def planar_average(grid, axis=2):
    """Average over the two in-plane grid directions."""
    return grid.mean(axis=tuple(a for a in range(3) if a != axis))


def macroscopic_average(profile, height, window):
    """Running average of a periodic profile over ``window`` Å (one period of the structure)."""
    n = len(profile)
    w = max(1, int(round(window / height * n)))
    kernel = np.ones(w) / w
    padded = np.r_[profile[-w:], profile, profile[:w]]
    return np.convolve(padded, kernel, mode="same")[w:w + n]


def vacuum_plateaus(profile, cell, cart, axis=2, margin=4.0):
    """Vacuum level on each side of the slab (see band_alignment.py for the same algorithm)."""
    n = len(profile)
    z = np.arange(n) / n
    frac = np.sort((cart @ np.linalg.inv(cell))[:, axis] % 1.0)
    gaps = np.diff(np.r_[frac, frac[0] + 1.0])
    k = int(np.argmax(gaps))
    start, length = frac[k], gaps[k]
    h = layer_height(cell, axis)
    m = min(margin / h, 0.35 * length)
    s = (z - start) % 1.0
    upper = (s >= m) & (s <= length / 2 - 0.05 * length)
    lower = (s >= length / 2 + 0.05 * length) & (s <= length - m)
    if upper.sum() < 2 or lower.sum() < 2:
        raise RuntimeError(f"vacuum too thin for a plateau ({length * h:.1f} Å)")
    return dict(V_upper=float(profile[upper].mean()), V_lower=float(profile[lower].mean()),
                spread=float(max(np.ptp(profile[upper]), np.ptp(profile[lower]))),
                vac_width=float(length * h), windows=(upper, lower))


def fermi_from_outcar(path):
    ef = None
    with open(path, errors="replace") as f:
        for ln in f:
            if "E-fermi" in ln:
                ef = float(ln.split()[2])
    return ef


def plot(path, zgrid, prof, macro, vac, symbols, zatoms, efermi, axis_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.plot(zgrid, prof, lw=0.8, color="#0072B2", label="planar average")
    if macro is not None:
        ax.plot(zgrid, macro, lw=1.6, color="#D55E00", label="macroscopic average")
    up, lo = vac["windows"]
    ax.plot(zgrid[up], np.full(up.sum(), vac["V_upper"]), lw=3, color="#009E73", alpha=0.6, label="vacuum plateaus")
    ax.plot(zgrid[lo], np.full(lo.sum(), vac["V_lower"]), lw=3, color="#009E73", alpha=0.6)
    if efermi is not None:
        ax.axhline(efermi, color="0.3", ls=":", lw=1)
        ax.text(zgrid[0], efermi, r"$E_F$", va="bottom", ha="left", fontsize=8)
    ymin = prof.min()
    elements = list(dict.fromkeys(symbols))
    colors = ["#E69F00", "#56B4E9", "#CC79A7", "#000000", "#F0E442", "#0072B2"]
    for e, c in zip(elements, itertools.cycle(colors)):
        za = zatoms[np.array(symbols) == e]
        ax.plot(za, np.full(len(za), ymin - 0.6), "|", ms=8, color=c, label=e)
    ax.set_xlabel(f"Position along {axis_name} (Å)")
    ax.set_ylabel("Electrostatic potential (eV)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(ymin - 1.2, max(prof.max(), efermi or prof.max()) + 1.0)
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    print(f"figure -> {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("locpot", nargs="?", default="LOCPOT")
    ap.add_argument("--axis", type=int, default=2, choices=[0, 1, 2], help="averaging direction: 0=a, 1=b, 2=c")
    ap.add_argument("--macro", type=float, help="window (Å) of the macroscopic average, e.g. the interlayer period")
    ap.add_argument("--outcar", help="OUTCAR to read E_fermi from (default: next to LOCPOT, if present)")
    ap.add_argument("--margin", type=float, default=4.0, help="distance (Å) from the slab ignored in the plateau")
    ap.add_argument("--plot", help="save the profile figure (PNG/PDF/SVG)")
    ap.add_argument("--csv", help="save z, planar and macroscopic averages")
    ap.add_argument("--version", action="version", version=__version__)
    a = ap.parse_args(argv)

    grid, cell, symbols, cart = read_locpot(a.locpot)
    h = layer_height(cell, a.axis)
    prof = planar_average(grid, a.axis)
    zgrid = np.arange(len(prof)) / len(prof) * h
    macro = macroscopic_average(prof, h, a.macro) if a.macro else None
    outcar = Path(a.outcar) if a.outcar else Path(a.locpot).with_name("OUTCAR")
    efermi = fermi_from_outcar(outcar) if outcar.is_file() else None
    normal = np.cross(*[cell[i] for i in range(3) if i != a.axis])
    zat = (cart @ (normal / np.linalg.norm(normal))) % h

    print(f"grid {grid.shape}, {len(symbols)} atoms, profile length {h:.3f} Å along axis {a.axis}")
    try:
        vac = vacuum_plateaus(prof, cell, cart, a.axis, a.margin)
    except RuntimeError as exc:
        print(f"no vacuum plateau: {exc}")
        vac = None
    if vac:
        v = 0.5 * (vac["V_upper"] + vac["V_lower"])
        print(f"vacuum gap        {vac['vac_width']:8.2f} Å")
        print(f"V_vac upper side  {vac['V_upper']:8.3f} eV")
        print(f"V_vac lower side  {vac['V_lower']:8.3f} eV   (difference {vac['V_upper'] - vac['V_lower']:+.3f} eV)")
        print(f"plateau spread    {vac['spread']:8.3f} eV" + ("   <-- not flat" if vac["spread"] > 0.05 else ""))
        if efermi is not None:
            print(f"E_fermi           {efermi:8.3f} eV")
            print(f"work function     {v - efermi:8.3f} eV  (upper {vac['V_upper'] - efermi:.3f}, "
                  f"lower {vac['V_lower'] - efermi:.3f})")
    if a.csv:
        cols = [zgrid, prof] + ([macro] if macro is not None else [])
        np.savetxt(a.csv, np.column_stack(cols), delimiter=",", fmt="%.6f",
                   header="z_A,planar_eV" + (",macro_eV" if macro is not None else ""), comments="")
        print(f"profile -> {a.csv}")
    if a.plot and vac:
        plot(a.plot, zgrid, prof, macro, vac, symbols, zat, efermi, "abc"[a.axis])
    return 0


if __name__ == "__main__":
    sys.exit(main())
