# Example data (synthetic)

Every file in this folder is **synthetic**: written by [`synthetic_vasp.py`](synthetic_vasp.py) in the
exact layout of VASP 6 outputs, with content designed so that the correct result is known in advance.
Nothing here comes from a real calculation, so the numbers are illustrative only; what they are good
for is learning the tools and testing them.

Regenerate everything with:

```bash
python examples/synthetic_vasp.py
```

This also runs the generators of the v1.1 examples, which live in their own modules and can be run on
their own as well: `synthetic_bands.py`, `synthetic_vacuum.py`, `synthetic_chgdiff.py`,
`synthetic_hbonds.py` and `synthetic_octahedra.py` (each one imports the writers of `synthetic_vasp.py`).

| Folder | What it contains | Used in | Known answer |
|---|---|---|---|
| `synthetic_slab/` | toy hybrid slab (1 Pb, 4 I, 2 CH3NH3-like cations, ~19 Å vacuum): POSCAR, CONTCAR, OUTCAR, EIGENVAL, PROCAR, LOCPOT, OSZICAR | Band_Alignment_Vacuum, LOCPOT_Planar_Average, Run_Diagnostics | vacuum level +2.00 eV, VBM −3.80, CBM −1.60 eV → IP 5.80, EA 3.60 eV; work function 5.70 eV; type I (inorganic inside organic) |
| `synthetic_slab_typeII/` | the same slab with the cation HOMO raised to −3.60 eV | Band_Alignment_Vacuum | true VBM is the organic HOMO; type II; IP 5.60 eV |
| `relax_inherited_constraints/` | ISIF = 3 relaxation, converged, with Pb and I fixed by `F F F` flags; +4.2 % volume | Run_Diagnostics, Selective_Dynamics | 5 fixed atoms with forces up to ~1 eV/Å; max force on free atoms 0.009 eV/Å |
| `relax_time_limit/` | relaxation killed by the Slurm wall time after 40 steps, one ionic step with an unconverged SCF | Run_Diagnostics | stopped; "DUE TO TIME LIMIT" in the Slurm log |
| `hybrid_scf_oscillating/` | PBE0 single point: INCAR with `BMIX = 0.0001`, OSZICAR with a period-3 oscillation, `ZPOTRF` crash in `vasp.out` | SCF_Rescue_Hybrid, Run_Diagnostics | verdict OSCILLATING; NBANDS = 96 |
| `layered_bulk/` | toy layered bulk cell in which one cation crosses the c boundary | Slab_Whole_Molecules | 2 molecules + 1 layer; the naive slab cuts a C–N bond |
| `bands_nosoc/`, `bands_soc/`, `bands_hybrid/`, `dos_lorbit11/` | toy lead-iodide layer: line-mode band runs Γ–X\|S–Y–Γ without and with SOC, a hybrid-style explicit KPOINTS (16 weighted + 60 zero-weight path points), and a LORBIT = 11 DOS run (Gaussian peaks of known area); written by [`synthetic_bands.py`](synthetic_bands.py) | Band_Structure_DOS | direct gap at S: 1.80 eV (PBE+D3), 1.05 eV (+SOC), 1.70 eV (HSE06+SOC); VB edge 4.0 I + 0.6 Pb states, CB edge 1.5 Pb + 0.4 I states (1 eV windows) |
| `vacuum_clean/`, `vacuum_residual/`, `vacuum_ramp/`, `vacuum_dipole_step/` | LOCPOT + INCAR of the toy slab (c = 30 Å, 19.1 Å of vacuum) with a known defect in the vacuum, written by [`synthetic_vacuum.py`](synthetic_vacuum.py); `vacuum_residual` is a four-block (non-collinear) LOCPOT | Vacuum_Potential_Check | cosine amplitude 0 / 50 meV / 0 / 0; ramp 0 / 0 / +10 meV/Å / 0; step 0 / 0 / 0 / 0.30 eV (IDIPOL = 3, LDIPOL) → verdicts OK, BAD, WARN, OK |
| `charge_difference/` | three CHGCARs (AB, A, B) of Gaussian charge blobs, fragment A = "Pb"+"I", B = "N"+"H" (A written in the spin-polarized layout), plus `B_other_grid/` on a different FFT grid; written by [`synthetic_chgdiff.py`](synthetic_chgdiff.py) | Charge_Density_Difference | ∫Δρ = 0; 0.25 e moved from B to A across z = 9.5 Å → ΔQ(9.5 Å) = +0.25 e; dipole +0.45 e·Å; `B_other_grid` is rejected |
| `hbond_toy/` | toy cell (39 atoms): 2 NH3-like cations, water, methanol-like OH, I/Br/Cl acceptors, Pb making iodides terminal/bridging; `POSCAR` (orthorhombic) and `POSCAR_triclinic` (same fragments, skewed cell); written by [`synthetic_hbonds.py`](synthetic_hbonds.py) | Hydrogen_Bonds | hydrogen bonds (angle ≥ 120°) 2 / 3 / 4 / 6 under 2.8 / 3.0 / 3.2 / 3.5 Å; 7 contacts ≤ 3.5 Å, one (2.90 Å, 95°) rejected by the angle; one 3.10 Å contact only across the a boundary; the C–H···I contact at 2.60 Å is ignored; same answer in both cells |
| `octahedra_ideal/`, `octahedra_distorted/`, `octahedra_skewed/`, `octahedra_slab/` | 2 × 2 corner-sharing PbI6 sheet with toy cations, written by [`synthetic_octahedra.py`](synthetic_octahedra.py): ideal; rotated ±12°, axial bond 3.30 vs 3.20 Å, axial tilt 8°; the same in a triclinic cell; the same as a slab with ~19 Å vacuum | Octahedral_Distortion | ideal: Δ_d 0, ⟨λ⟩ 1, σ² 0, B–X–B 180°; distorted/skewed/slab: Δ_d 0.01375, ⟨λ⟩ 1.00697, σ² 23.27 deg², B–X–B 156°, tilt 8° |

The toy structures are deliberately minimal (one octahedron per cell, small cations) so that the files
stay small. No POTCAR is included: POTCAR files are licensed by VASP Software GmbH and must not be
redistributed.
