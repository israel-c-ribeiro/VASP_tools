"""
synthetic_vasp.py — write small, format-faithful VASP output files with a KNOWN answer.

Why this exists
---------------
Real VASP outputs are large, licensed-software products and usually tied to
unpublished work, so they cannot ship with a public repository. The tutorials
and the test-suite of VASP_tools therefore run on synthetic files that follow
the exact layout of POSCAR, EIGENVAL, PROCAR, LOCPOT, OUTCAR and OSZICAR, but
whose content is designed so that the correct result is known in advance
(vacuum level, band edges, forces, convergence state ...).

The toy system is a single-layer "hybrid perovskite-like" slab:
one Pb and four I atoms (the inorganic layer, fragment A) sandwiched between
two methylammonium-like cations (fragment B), with ~19 Å of vacuum along c.
The numbers are illustrative only; they are NOT the result of a calculation.

Usage
-----
    python synthetic_vasp.py                 # writes every example folder next to this file
    python synthetic_vasp.py --out my_dir    # somewhere else

    from synthetic_vasp import make_toy_slab_run
    truth = make_toy_slab_run("run_dir")     # returns the ground-truth dictionary
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ORBITALS = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "x2-y2"]


# --------------------------------------------------------------------------- geometry
def toy_slab(a: float = 6.3, c: float = 30.0, layer_z: float = 15.0):
    """Symbols, cell (3x3, Å) and fractional coordinates of the toy slab."""
    cell = np.diag([a, a, c])
    cart = [("Pb", (0.0, 0.0, layer_z)),
            ("I", (a / 2, 0.0, layer_z)), ("I", (0.0, a / 2, layer_z)),
            ("I", (0.0, 0.0, layer_z + 3.2)), ("I", (0.0, 0.0, layer_z - 3.2))]
    cart += cation(np.array([a / 2, a / 2, layer_z + 3.6]), up=True)
    cart += cation(np.array([a / 2, a / 2, layer_z - 3.6]), up=False)
    order = {"Pb": 0, "I": 1, "C": 2, "N": 3, "H": 4}
    cart.sort(key=lambda t: order[t[0]])
    symbols = [s for s, _ in cart]
    pos = np.array([p for _, p in cart])
    return symbols, cell, (pos @ np.linalg.inv(cell)) % 1.0


def cation(n_pos: np.ndarray, up: bool = True):
    """CH3NH3+-like cation with its N at ``n_pos``; the C-N bond points away from the layer."""
    s = 1.0 if up else -1.0
    out = [("N", tuple(n_pos)), ("C", tuple(n_pos + [0, 0, s * 1.47]))]
    for k in range(3):                                  # three H on N, three H on C
        ang = 2 * np.pi * k / 3
        out.append(("H", tuple(n_pos + [0.96 * np.cos(ang), 0.96 * np.sin(ang), -s * 0.34])))
        out.append(("H", tuple(n_pos + [1.03 * np.cos(ang + 0.5), 1.03 * np.sin(ang + 0.5), s * (1.47 + 0.36)])))
    return out


# --------------------------------------------------------------------------- writers
def write_poscar(path, symbols, cell, frac, sd=None, comment="synthetic structure"):
    """VASP 5 POSCAR (species line present), direct coordinates, optional selective dynamics.

    ``sd`` is an (N, 3) boolean array, True = the coordinate may move ("T")."""
    names = list(dict.fromkeys(symbols))
    counts = [symbols.count(n) for n in names]
    lines = [comment, "   1.00000000000000"]
    lines += ["  " + "  ".join(f"{x:20.16f}" for x in v) for v in cell]
    lines += ["   " + "   ".join(f"{n:<2s}" for n in names), "   " + "   ".join(f"{n:4d}" for n in counts)]
    if sd is not None:
        lines.append("Selective dynamics")
    lines.append("Direct")
    for i, f in enumerate(frac):
        row = "  " + "  ".join(f"{x:19.16f}" for x in f)
        if sd is not None:
            row += "   " + "   ".join("T" if m else "F" for m in sd[i])
        lines.append(row)
    Path(path).write_text("\n".join(lines) + "\n", newline="\n")


def write_eigenval(path, eig, occ, kpts, nelect, natoms):
    """EIGENVAL. ``eig``/``occ`` have shape (nspin, nk, nb); nspin = 1 or 2."""
    nspin, nk, nb = eig.shape
    lines = [f"{natoms:5d}{natoms:5d}    1{nspin:5d}",
             "  0.1190700E+04  0.6300000E-09  0.6300000E-09  0.3000000E-08  0.5000000E-15",
             "  1.000000000000000E-004", "  CAR", " synthetic", f"{int(nelect):7d}{nk:7d}{nb:7d}"]
    w = 1.0 / nk
    for ik in range(nk):
        lines.append("")
        lines.append("  " + "  ".join(f"{x:.7E}" for x in kpts[ik]) + f"  {w:.7E}")
        for ib in range(nb):
            if nspin == 1:
                lines.append(f"{ib + 1:5d}   {eig[0, ik, ib]:13.6f}   {occ[0, ik, ib]:9.6f}")
            else:
                lines.append(f"{ib + 1:5d}   {eig[0, ik, ib]:13.6f}   {eig[1, ik, ib]:13.6f}"
                             f"   {occ[0, ik, ib]:9.6f}   {occ[1, ik, ib]:9.6f}")
    Path(path).write_text("\n".join(lines) + "\n", newline="\n")


def write_procar(path, eig, occ, kpts, proj, soc=False):
    """PROCAR in the LORBIT = 11 layout (one spin channel).

    ``proj`` has shape (nk, nb, natoms, 9). With ``soc`` each band carries four blocks
    (total, m_x, m_y, m_z) after a single 'ion' header, as in non-collinear runs."""
    nk, nb, nat, norb = proj.shape
    out = ["PROCAR lm decomposed",
           f"# of k-points:  {nk:4d}         # of bands:  {nb:4d}         # of ions:  {nat:4d}", ""]
    for ik in range(nk):
        out.append(f" k-point  {ik + 1:4d} :    " + " ".join(f"{x:.8f}" for x in kpts[ik])
                   + f"     weight = {1.0 / nk:.8f}")
        out.append("")
        for ib in range(nb):
            out.append(f"band  {ib + 1:4d} # energy  {eig[ik, ib]:13.8f} # occ.  {occ[ik, ib]:.8f}")
            out.append("")
            out.append("ion  " + "".join(f"{o:>7s}" for o in ORBITALS) + "    tot")
            blocks = [proj[ik, ib]] + ([0.01 * proj[ik, ib]] * 3 if soc else [])
            for blk in blocks:
                for ia in range(nat):
                    row = blk[ia]
                    out.append(f"{ia + 1:5d}" + "".join(f"{x:7.3f}" for x in row) + f"{row.sum():7.3f}")
                tot = blk.sum(axis=0)
                out.append("tot  " + "".join(f"{x:7.3f}" for x in tot) + f"{tot.sum():7.3f}")
            out.append("")
        out.append("")
    Path(path).write_text("\n".join(out) + "\n", newline="\n")


def write_locpot(path, symbols, cell, frac, grid, nblocks=1):
    """LOCPOT/CHGCAR layout: POSCAR header, blank line, grid size, values (x fastest, 5 per line).

    ``nblocks = 4`` mimics a non-collinear LOCPOT (the extra blocks are written as zeros)."""
    names = list(dict.fromkeys(symbols))
    counts = [symbols.count(n) for n in names]
    head = ["synthetic LOCPOT", "   1.00000000000000"]
    head += ["  " + "  ".join(f"{x:12.6f}" for x in v) for v in cell]
    head += ["   " + "   ".join(names), "   " + "   ".join(str(n) for n in counts), "Direct"]
    head += ["  " + "  ".join(f"{x:.6f}" for x in f) for f in frac]
    nx, ny, nz = grid.shape
    body = []
    for b in range(nblocks):
        vals = grid.T.ravel() if b == 0 else np.zeros(grid.size)
        rows = [" ".join(f"{v:.11E}" for v in vals[i:i + 5]) for i in range(0, vals.size, 5)]
        body += ["", f"  {nx:4d}  {ny:4d}  {nz:4d}"] + rows
    Path(path).write_text("\n".join(head + body) + "\n", newline="\n")


def write_outcar(path, *, natoms, efermi=None, energies=(-100.0,), forces=None, positions=None,
                 lsorbit=False, finished=True, scf_converged=None, ionic_converged=False,
                 nsw=0, ibrion=-1, isif=2, ediffg=-0.02, nbands=20, volume=1190.7, volumes=None,
                 pressure=0.5, elapsed=1234.5):
    """A minimal OUTCAR containing every line the VASP_tools parsers look for.

    ``energies`` = free energy (TOTEN) of each ionic step; ``scf_converged`` = one bool per
    ionic step (default all True); ``forces``/``positions`` = (natoms, 3) of the LAST step;
    ``volumes`` = cell volume after each ionic step (variable-cell runs)."""
    n = len(energies)
    scf_converged = [True] * n if scf_converged is None else list(scf_converged)
    L = [" vasp.6.4.1 synthetic OUTCAR (written by VASP_tools/examples/synthetic_vasp.py)",
         f"   number of dos      NEDOS =    301   number of ions     NIONS = {natoms:6d}",
         f"   NBANDS=  {nbands:6d}",
         f"   NSW    = {nsw:6d}    number of steps for IOM",
         f"   IBRION = {ibrion:6d}    ionic relax: 0-MD 1-quasi-New 2-CG",
         f"   ISIF   = {isif:6d}    stress and relaxation",
         f"   EDIFFG = {ediffg:.1E}   stopping-criterion for IOM",
         "   EDIFF  = 0.1E-05   stopping-criterion for ELM",
         f"   LSORBIT =      {'T' if lsorbit else 'F'}    spin-orbit coupling",
         f"  volume of cell : {volume:12.2f}"]
    for i, e in enumerate(energies):
        L.append("-" * 40 + " Ionic step " + str(i + 1) + " " + "-" * 40)
        if scf_converged[i]:
            L.append("------------------------ aborting loop because EDIFF is reached ----------------------------------------")
        else:
            L.append("------------------------ aborting loop EDIFF was not reached (unconverged)  ----------------------------")
        if efermi is not None:
            L.append(f" E-fermi : {efermi:9.4f}     XC(G=0):  -1.0000     alpha+bet : -1.0000")
        L.append(f"  external pressure = {pressure:11.2f} kB  Pullay stress =        0.00 kB")
        if volumes is not None:
            L.append(f"  volume of cell : {volumes[i]:12.2f}")
        if forces is not None and i == n - 1:
            L.append(" POSITION                                       TOTAL-FORCE (eV/Angst)")
            L.append(" " + "-" * 83)
            for p, f in zip(positions, forces):
                L.append(f" {p[0]:12.5f} {p[1]:12.5f} {p[2]:12.5f}    {f[0]:13.6f} {f[1]:13.6f} {f[2]:13.6f}")
            L.append(" " + "-" * 83)
            L.append("    total drift:                                0.000001     -0.000001      0.000002")
        L.append("  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)")
        L.append(f"  free  energy   TOTEN  = {e:18.8f} eV")
        L.append(f"  energy  without entropy= {e + 0.001:18.8f}  energy(sigma->0) = {e + 0.0005:18.8f}")
    if ionic_converged:
        L.append(" reached required accuracy - stopping structural energy minimisation")
    if finished:
        L += ["", "                  General timing and accounting informations for this job:",
              "                  ========================================================",
              f"                            Elapsed time (sec): {elapsed:12.3f}",
              "                      Voluntary context switches:        1234"]
    Path(path).write_text("\n".join(L) + "\n", newline="\n")


def write_oszicar(path, scf_energies, algo="DAV"):
    """OSZICAR with one list of SCF total energies per ionic step."""
    L = ["       N       E                     dE             d eps       ncg     rms          rms(c)"]
    for istep, es in enumerate(scf_energies):
        prev = 0.0
        for it, e in enumerate(es):
            L.append(f"{algo}: {it + 1:3d}    {e: .12E}   {e - prev: .5E}   {0.1 * (e - prev): .5E}   960   0.123E+01")
            prev = e
        L.append(f"   {istep + 1} F= {es[-1]: .8E} E0= {es[-1] + 0.001: .8E}  d E ={es[-1]: .6E}")
    Path(path).write_text("\n".join(L) + "\n", newline="\n")


# --------------------------------------------------------------------------- the toy run
def toy_potential(cell, frac, shape=(8, 8, 160), v_vac=2.0, depth=8.0, dipole_step=0.0):
    """Electrostatic potential (eV) on a grid: flat vacuum plateau at ``v_vac``, a smooth well
    where the slab is, Gaussian dips at the atoms and an in-plane modulation whose planar average
    is zero. ``dipole_step`` adds a potential step in the middle of the vacuum (dipole correction)."""
    nx, ny, nz = shape
    x, y, z = np.meshgrid(np.arange(nx) / nx, np.arange(ny) / ny, np.arange(nz) / nz, indexing="ij")
    c = np.linalg.norm(cell[2])
    zf = frac[:, 2]
    zmin, zmax = zf.min(), zf.max()
    box = 0.5 * (np.tanh((z - zmin) * c / 1.0) - np.tanh((z - zmax) * c / 1.0))
    v = v_vac - depth * box
    for f in frac:
        dz = (z - f[2] + 0.5) % 1.0 - 0.5
        v -= 1.5 * np.exp(-(dz * c) ** 2 / 0.5)
    v += 0.8 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * box
    if dipole_step:
        zmid = (zmax + (zmin + 1.0 - zmax) / 2.0) % 1.0
        v += dipole_step * (((z - zmid) % 1.0) < ((zmin + 1.0 - zmid) % 1.0))
    return v


def make_toy_slab_run(folder, *, homo_B=-3.95, v_vac=2.0, soc=False, nblocks_locpot=None,
                      dipole_step=0.0, efermi=-3.70):
    """Write POSCAR, CONTCAR, EIGENVAL, PROCAR, LOCPOT, OUTCAR and OSZICAR of the toy slab.

    Fragment A = Pb + I (the "inorganic" layer), fragment B = C, N, H (the cations).
    Bands at two k-points (Gamma and X). Returns the ground truth as a dict.
    ``homo_B`` > -3.80 puts the cation HOMO above the inorganic VBM (a type-II alignment)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    symbols, cell, frac = toy_slab()
    nat = len(symbols)
    is_A = np.array([s in ("Pb", "I") for s in symbols])
    kpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    # (energy at Gamma, energy at X, inorganic fraction) for each band
    bands = [(-12.0, -11.9, 0.5), (-11.0, -10.9, 0.2), (-10.0, -10.1, 0.8), (-9.0, -8.8, 0.3),
             (-8.0, -8.1, 0.6), (-7.0, -7.2, 0.1), (-6.0, -6.3, 0.9), (-5.5, -5.4, 0.4),
             (-5.0, -5.2, 0.85), (-4.60, -4.65, 0.50), (homo_B, homo_B - 0.02, 0.05),
             (-3.80, -4.30, 0.92),                                      # inorganic VBM at Gamma
             (-1.60, -1.20, 0.90),                                      # inorganic CBM at Gamma
             (0.50, 0.52, 0.06),                                        # cation LUMO
             (1.20, 1.50, 0.50), (2.00, 2.10, 0.70), (2.50, 2.40, 0.30), (3.00, 3.20, 0.60),
             (3.50, 3.60, 0.40), (4.00, 4.20, 0.50)]
    n_occ = 12
    nb, nk = len(bands), 2
    eig = np.array([[b[0] for b in bands], [b[1] for b in bands]])       # (nk, nb)
    occ = np.zeros((nk, nb))
    occ[:, :n_occ] = 1.0
    fA = np.array([b[2] for b in bands])
    proj = np.zeros((nk, nb, nat, 9))
    for ib in range(nb):
        wa = fA[ib] / is_A.sum()
        wb = (1.0 - fA[ib]) / (~is_A).sum()
        for ia in range(nat):
            w = wa if is_A[ia] else wb
            proj[:, ib, ia, 0] = 0.3 * w                          # s
            proj[:, ib, ia, 1:4] = 0.7 * w / 3                    # p
    write_poscar(folder / "POSCAR", symbols, cell, frac, comment="toy hybrid slab (synthetic)")
    write_poscar(folder / "CONTCAR", symbols, cell, frac, comment="toy hybrid slab (synthetic)")
    write_eigenval(folder / "EIGENVAL", eig[None], occ[None], kpts, nelect=2 * n_occ, natoms=nat)
    write_procar(folder / "PROCAR", eig, occ, kpts, proj, soc=soc)
    grid = toy_potential(cell, frac, v_vac=v_vac, dipole_step=dipole_step)
    write_locpot(folder / "LOCPOT", symbols, cell, frac, grid,
                 nblocks=nblocks_locpot or (4 if soc else 1))
    write_outcar(folder / "OUTCAR", natoms=nat, efermi=efermi, energies=[-123.456789], lsorbit=soc,
                 nbands=nb, volume=float(abs(np.linalg.det(cell))))
    write_oszicar(folder / "OSZICAR", [list(-123.456789 + 50.0 * np.exp(-np.arange(32) / 1.5))])
    occ_mask = occ > 0.5
    vbm, cbm = eig[occ_mask].max(), eig[~occ_mask].min()
    vbm_A = -3.80
    vbm_B = homo_B
    return dict(v_vac=v_vac, efermi=efermi, VBM=vbm, CBM=cbm, gap=cbm - vbm, IP=v_vac - vbm, EA=v_vac - cbm,
                work_function=v_vac - efermi, VBM_A=vbm_A, CBM_A=-1.60, VBM_B=vbm_B, CBM_B=0.50,
                natoms=nat, vac_width=30.0 * (1 - (frac[:, 2].max() - frac[:, 2].min())))


# --------------------------------------------------------------------------- runs with problems
def make_diagnostic_examples(out):
    """Three runs for the Run_Diagnostics and SCF_Rescue_Hybrid tutorials.

    relax_inherited_constraints  ISIF = 3 relaxation that "converged" with Pb and I still fixed
                                 by selective-dynamics flags copied from an earlier POSCAR
    relax_time_limit             relaxation killed by the Slurm wall time before converging
    hybrid_scf_oscillating       hybrid-functional single point whose SCF oscillates and crashes"""
    out = Path(out)
    symbols, cell, frac = toy_slab(c=13.5, layer_z=6.75)            # bulk-like layered cell, no vacuum
    nat = len(symbols)
    inorganic = np.array([s in ("Pb", "I") for s in symbols])
    rng = np.random.default_rng(42)

    # 1. converged, but with inherited constraints and a large volume change
    d = out / "relax_inherited_constraints"
    d.mkdir(parents=True, exist_ok=True)
    sd = np.repeat(~inorganic[:, None], 3, axis=1)                   # Pb and I: F F F
    write_poscar(d / "POSCAR", symbols, cell, frac, sd=sd, comment="restart from an earlier CONTCAR (synthetic)")
    cell1 = cell.copy()
    cell1[2] *= 1.042
    write_poscar(d / "CONTCAR", symbols, cell1, frac, sd=sd, comment="restart from an earlier CONTCAR (synthetic)")
    F = rng.normal(0, 0.004, (nat, 3))
    F[inorganic] = rng.normal(0, 0.6, (inorganic.sum(), 3))
    v0 = abs(np.linalg.det(cell))
    vols = list(np.linspace(v0, v0 * 1.042, 6))
    write_outcar(d / "OUTCAR", natoms=nat, efermi=-2.1, energies=list(-140.0 - 0.8 * (1 - np.exp(-np.arange(6)))),
                 forces=F, positions=frac @ cell1, nsw=150, ibrion=2, isif=3, ediffg=-0.02,
                 ionic_converged=True, volume=v0, volumes=vols, pressure=-0.8)
    write_lines(d / "INCAR", ["SYSTEM = toy layered cell", "ISIF = 3", "IBRION = 2", "NSW = 150", "EDIFFG = -0.02"])

    # 2. killed by the wall time
    d = out / "relax_time_limit"
    d.mkdir(parents=True, exist_ok=True)
    write_poscar(d / "POSCAR", symbols, cell, frac, comment="toy layered cell (synthetic)")
    write_poscar(d / "CONTCAR", symbols, cell, frac, comment="toy layered cell (synthetic)")
    F = rng.normal(0, 0.04, (nat, 3))
    write_outcar(d / "OUTCAR", natoms=nat, efermi=-2.1, energies=list(-140.0 - 0.5 * (1 - np.exp(-np.arange(40) / 10))),
                 forces=F, positions=frac @ cell, nsw=100, ibrion=2, isif=2, ediffg=-0.02,
                 scf_converged=[True] * 23 + [False] + [True] * 16, finished=False)
    write_lines(d / "slurm-123456.out", [
        " running on  128 total cores", " ...",
        "slurmstepd: error: *** JOB 123456 ON node042 CANCELLED AT 2026-10-01T09:59:59 DUE TO TIME LIMIT ***"])

    # 3. hybrid single point: SCF converges, then oscillates, then crashes
    d = out / "hybrid_scf_oscillating"
    d.mkdir(parents=True, exist_ok=True)
    s_symbols, s_cell, s_frac = toy_slab()
    write_poscar(d / "POSCAR", s_symbols, s_cell, s_frac, comment="toy hybrid slab (synthetic)")
    k = np.arange(1, 61)
    E = np.where(k <= 10, -1226.5 + 300 * np.exp(-k / 2.0),
                 -1226.5 + 20 * np.sin(2 * np.pi * k / 3) * (1 + 0.02 * k))
    write_oszicar(d / "OSZICAR", [list(E)])
    write_outcar(d / "OUTCAR", natoms=len(s_symbols), energies=[], nsw=0, ibrion=-1, nbands=96,
                 finished=False, scf_converged=[])
    write_lines(d / "vasp.out", [" running on  128 total cores", " DAV:  58 ...",
                                 " LAPACK: Routine ZPOTRF failed!            1"])
    write_lines(d / "INCAR", [
        "SYSTEM  = toy slab, PBE0 single point",
        "ENCUT   = 500", "PREC    = Accurate", "EDIFF   = 1E-6", "NELM    = 60",
        "ISMEAR  = 0", "SIGMA   = 0.01",
        "LHFCALC = .TRUE.", "AEXX    = 0.25", "ALGO    = Normal",
        "AMIX    = 0.1", "BMIX    = 0.0001     ! linear mixing: Kerker preconditioning switched off",
        "LVHAR   = .TRUE.", "LORBIT  = 11", "NSW     = 0", "IBRION  = -1"])


def write_lines(path, lines):
    Path(path).write_text("\n".join(lines) + "\n", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent), help="parent folder (default: next to this file)")
    a = ap.parse_args()
    out = Path(a.out)
    for name, kw in (("synthetic_slab", {}), ("synthetic_slab_typeII", {"homo_B": -3.60, "efermi": -3.50})):
        truth = make_toy_slab_run(out / name, **kw)
        print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in truth.items() if isinstance(v, float)))
    make_diagnostic_examples(out)
    print("relax_inherited_constraints, relax_time_limit, hybrid_scf_oscillating: written")
    (out / "layered_bulk").mkdir(parents=True, exist_ok=True)       # one cation crosses the c boundary
    write_poscar(out / "layered_bulk" / "POSCAR", *toy_slab(c=13.5, layer_z=4.3),
                 comment="toy layered hybrid bulk cell (synthetic)")
    print("layered_bulk: written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
