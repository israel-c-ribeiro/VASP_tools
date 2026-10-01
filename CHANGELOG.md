# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

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
