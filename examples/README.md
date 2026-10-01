# Example data (synthetic)

Every file in this folder is **synthetic**: written by [`synthetic_vasp.py`](synthetic_vasp.py) in the
exact layout of VASP 6 outputs, with content designed so that the correct result is known in advance.
Nothing here comes from a real calculation, so the numbers are illustrative only; what they are good
for is learning the tools and testing them.

Regenerate everything with:

```bash
python examples/synthetic_vasp.py
```

| Folder | What it contains | Used in | Known answer |
|---|---|---|---|
| `synthetic_slab/` | toy hybrid slab (1 Pb, 4 I, 2 CH3NH3-like cations, ~19 Å vacuum): POSCAR, CONTCAR, OUTCAR, EIGENVAL, PROCAR, LOCPOT, OSZICAR | Band_Alignment_Vacuum, LOCPOT_Planar_Average, Run_Diagnostics | vacuum level +2.00 eV, VBM −3.80, CBM −1.60 eV → IP 5.80, EA 3.60 eV; work function 5.70 eV; type I (inorganic inside organic) |
| `synthetic_slab_typeII/` | the same slab with the cation HOMO raised to −3.60 eV | Band_Alignment_Vacuum | true VBM is the organic HOMO; type II; IP 5.60 eV |
| `relax_inherited_constraints/` | ISIF = 3 relaxation, converged, with Pb and I fixed by `F F F` flags; +4.2 % volume | Run_Diagnostics, Selective_Dynamics | 5 fixed atoms with forces up to ~1 eV/Å; max force on free atoms 0.009 eV/Å |
| `relax_time_limit/` | relaxation killed by the Slurm wall time after 40 steps, one ionic step with an unconverged SCF | Run_Diagnostics | stopped; "DUE TO TIME LIMIT" in the Slurm log |
| `hybrid_scf_oscillating/` | PBE0 single point: INCAR with `BMIX = 0.0001`, OSZICAR with a period-3 oscillation, `ZPOTRF` crash in `vasp.out` | SCF_Rescue_Hybrid, Run_Diagnostics | verdict OSCILLATING; NBANDS = 96 |
| `layered_bulk/` | toy layered bulk cell in which one cation crosses the c boundary | Slab_Whole_Molecules | 2 molecules + 1 layer; the naive slab cuts a C–N bond |

The toy structures are deliberately minimal (one octahedron per cell, small cations) so that the files
stay small. No POTCAR is included: POTCAR files are licensed by VASP Software GmbH and must not be
redistributed.
