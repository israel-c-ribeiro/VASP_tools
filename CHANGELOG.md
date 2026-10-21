# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [1.1.0] - 2026-10-21

### Added
- `Band_Structure_DOS/bands_dos.py`: band structure(s) and element/orbital-projected DOS in one figure,
  from EIGENVAL/KPOINTS (line mode, hybrid zero-weight paths, KPOINTS_OPT) and DOSCAR (LORBIT 10/11,
  ISPIN 1/2, non-collinear); SOC vs no-SOC overlay, method labels from INCAR, VBM/CBM/gap report with
  k-location, band-edge character, CSV export.
- `Vacuum_Potential_Check/vacuum_check.py`: quality of the vacuum reference — line (+ dipole step) plus
  long-wavelength cosine fit of the planar-averaged potential in the vacuum, ramp, two-side difference,
  RMS residual and the INCAR mixing/EDIFF/dipole settings, with an OK/WARN/BAD verdict; streaming LOCPOT
  reader (four-block and gzipped files), batch mode.
- `Charge_Density_Difference/chgdiff.py`: Δρ = ρ_AB − ρ_A − ρ_B from three CHGCARs (streamed, multi-GB),
  grid/cell/atom compatibility checks, CHGDIFF.vasp and cube output, plane-averaged profile and
  charge-displacement curve ΔQ(z) with the charge moved across fragment boundaries or chosen planes.
- `Hydrogen_Bonds/hbonds.py`: geometric D–H···A hydrogen bonds and short contacts in periodic structures
  (all images), several cutoffs with the angle criterion, per-donor-group aggregation, terminal/bridging
  acceptor tags, soft count, CSVs and batch mode.
- `Octahedral_Distortion/octahedra.py`: B–X octahedra of bulk cells and slabs — Baur distortion index,
  quadratic elongation, bond-angle variance, bridging B–X–B angles, axial tilts, area per centre; batch
  mode and before/after comparison.
- Synthetic examples with known answers for each new tool (`examples/synthetic_*.py`, run by
  `synthetic_vasp.py`), their tests, and their README figures in `docs/make_figures.py`.

## [1.0.0] - 2026-10-01

### Added
- `Run_Diagnostics/vasp_diagnose.py`: per-run health check (status, SCF of every ionic step, forces on
  free atoms vs EDIFFG, selective-dynamics constraints, cell change, known VASP/Slurm error messages).
- `Selective_Dynamics/sd_tool.py`: report, release and set selective-dynamics flags.
- `Slab_Whole_Molecules/make_slab.py`: slabs of layered hybrid crystals with every molecule kept whole.
- `SCF_Rescue_Hybrid/scf_rescue.py`: OSZICAR diagnosis and the two-step semilocal -> hybrid rescue job.
- `LOCPOT_Planar_Average/planar_average.py`: planar and macroscopic averages, vacuum level, work function.
- `Band_Alignment_Vacuum/band_alignment.py`: vacuum-referenced band edges, IP/EA, fragment-resolved
  edges and band-alignment type, batch mode.
- `examples/`: synthetic, format-faithful VASP outputs with known answers, and their generator.
- Test suite (pytest) and continuous integration on Linux and Windows.
- `docs/make_figures.py`: regenerates every README figure through the command-line tools.
- LICENSE (MIT), CITATION.cff, requirements files.

### Changed
- README rewritten as an index of tutorials with a suggested learning path.

## [0.1.0] - 2024-12-17

### Added
- `MAGMOM_Vasp/magmom_input.py` and `Vacuum_Slab_Control/center_vacum.py`.
