"""Regenerate every figure of the README files from the synthetic examples.

    python docs/make_figures.py          # writes docs/img/*.png

The examples are written first (examples/synthetic_vasp.py), then each tool is run exactly as a
user would run it, so the figures double as an end-to-end test of the command-line interfaces.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
EX = ROOT / "examples"
sys.path.insert(0, str(EX))


def load(relpath):
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def slab_figure(out):
    """Side view of the naive and the whole-molecule slab of examples/layered_bulk."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ase.io import read
    from ase.data import covalent_radii
    ms = load("Slab_Whole_Molecules/make_slab.py")
    bulk = read(EX / "layered_bulk" / "POSCAR", format="vasp")
    slabs = {"naive: wrap + center": ms.naive_slab(bulk, 15.0),
             "make_slab.py: whole molecules": ms.whole_molecule_slab(bulk, ["Pb", "I"], "Pb", "N", 15.0)}
    d = bulk.get_all_distances(mic=True)
    r = covalent_radii[bulk.numbers]
    pairs = [(i, j) for i in range(len(bulk)) for j in range(i + 1, len(bulk)) if d[i, j] < 1.2 * (r[i] + r[j])]
    colors = {"Pb": "#4D4D4D", "I": "#CC79A7", "C": "#E69F00", "N": "#0072B2", "H": "#BBBBBB"}
    size = {"Pb": 70, "I": 55, "C": 30, "N": 30, "H": 10}
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.6), sharey=True)
    for ax, (title, s) in zip(axes, slabs.items()):
        p = s.positions
        h = (p[:, 0] + 0.35 * p[:, 1]) % 6.3                   # oblique projection: no atom hides another
        for i, j in pairs:
            if {bulk[i].symbol, bulk[j].symbol} <= {"Pb", "I"}:
                continue
            if abs(p[i, 2] - p[j, 2]) < 2.0:
                ax.plot(h[[i, j]], p[[i, j], 2], color="0.55", lw=0.8, zorder=1)
            else:                                            # the bond now runs through the vacuum
                up, lo = (i, j) if p[i, 2] > p[j, 2] else (j, i)
                ax.plot([h[up]] * 2, [p[up, 2], p[up, 2] + 3.0], color="#D55E00", lw=1.2, ls="--")
                ax.plot([h[lo]] * 2, [p[lo, 2], p[lo, 2] - 3.0], color="#D55E00", lw=1.2, ls="--")
                ax.text(-0.6, p[up, 2] + 3.3, f"{bulk[up].symbol}-{bulk[lo].symbol} bond cut:\n"
                        "it runs across the vacuum", color="#D55E00", fontsize=6.5, va="bottom", ha="left")
        for e in colors:
            m = np.array(s.get_chemical_symbols()) == e
            ax.scatter(h[m], p[m, 2], s=size[e], color=colors[e], label=e, zorder=2,
                       edgecolor="white", linewidth=0.4)
        ax.axhspan(p[:, 2].max(), s.cell.lengths()[2], color="#56B4E9", alpha=0.08, lw=0)
        ax.axhspan(0, p[:, 2].min(), color="#56B4E9", alpha=0.08, lw=0)
        ax.set_title(title, fontsize=8)
        ax.set_xlim(-0.8, 7.1)
        ax.set_ylim(0, s.cell.lengths()[2] + 0.5)
        ax.set_xlabel("in-plane position (Å)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("z (Å)")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main() -> int:
    import synthetic_vasp
    IMG.mkdir(parents=True, exist_ok=True)
    sys.argv = ["synthetic_vasp.py"]
    synthetic_vasp.main()
    load("LOCPOT_Planar_Average/planar_average.py").main(
        [str(EX / "synthetic_slab" / "LOCPOT"), "--macro", "6.4", "--plot", str(IMG / "planar_average.png")])
    load("Band_Alignment_Vacuum/band_alignment.py").main(
        [str(EX / "synthetic_slab*"), "--fragment", "Pb,I", "--labels", "inorganic,organic",
         "--plot", str(IMG / "band_alignment.png")])
    load("SCF_Rescue_Hybrid/scf_rescue.py").main(
        ["diagnose", str(EX / "hybrid_scf_oscillating" / "OSZICAR"), "--ediff", "1e-6", "--nelm", "60",
         "--plot", str(IMG / "scf_oscillating.png")])
    slab_figure(IMG / "slab_whole_molecules.png")
    print(f"figures -> {IMG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
