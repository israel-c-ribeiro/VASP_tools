# Run diagnostics: did my VASP run really work?

**A per-run health check: status, SCF convergence of every ionic step, residual forces on the free
atoms, inherited constraints, cell change, and the known error messages of VASP and Slurm, each with
its usual cause and what to try.**

## Why

"The job finished" and "the result is usable" are different statements. Some classic traps:

| Trap | What you see | What really happened |
|---|---|---|
| SCF hit `NELM` | the run ends normally, all files are written | the last electronic loop never reached `EDIFF`; energies, forces and eigenvalues are meaningless |
| inherited selective dynamics | "reached required accuracy" | part of the structure was fixed by `F F F` flags copied from an earlier CONTCAR, and the forces on it are large |
| large cell change (ISIF = 3) | converged | the plane-wave basis was set up for the initial cell (Pulay stress): restart from CONTCAR |
| wall time | no error in OUTCAR | Slurm killed the job; only the Slurm log says so |

`vasp_diagnose.py` checks all of this in one pass, for one run or a hundred.

## Quick start with the example data

[`examples/`](../examples) contains three runs with typical problems and two clean single points
(all synthetic, written by [`synthetic_vasp.py`](../examples/synthetic_vasp.py)).

```bash
python Run_Diagnostics/vasp_diagnose.py examples/relax_inherited_constraints
```

```text
=== examples/relax_inherited_constraints  ->  OK (2 warnings)
  [OK  ] finished normally in 0.34 h
  [OK  ] all 6 electronic loops reached EDIFF
  [OK  ] relaxation converged in 6 ionic steps, max force on free atoms 0.009 eV/Å (EDIFFG = -0.02)
  [INFO] energy change of the last ionic steps: -68.44 meV, -25.18 meV, -9.26 meV
  [WARN] selective dynamics: 5 atom(s) with fixed coordinates (1 Pb, 4 I); max force on fully fixed atoms 1.030 eV/Å. Intended? (flags are copied along when you restart from a CONTCAR)
  [INFO] the cell changed while fractional coordinates of fixed atoms stayed put: their only displacement is the image of the cell strain
  [WARN] volume changed by +4.20 %: restart from CONTCAR so the plane-wave basis matches the new cell (Pulay stress)
  [INFO] final external pressure -0.80 kB
```

A whole campaign, one line per run:

```bash
python Run_Diagnostics/vasp_diagnose.py "examples/*" --summary --csv health.csv
```

```text
run                              state                    ionic SCFfail    Fmax fixed    dV%  errors
hybrid_scf_oscillating           stopped                      0       0     nan     0    nan  ZPOTRF (orthonormalisation)
relax_inherited_constraints      OK (2 warnings)              6       0   0.009     5   4.20
relax_time_limit                 stopped                     40       1   0.142     0    nan  wall time reached
synthetic_slab                   OK                           1       0     nan     0    nan
synthetic_slab_typeII            OK                           1       0     nan     0    nan
```

The exit code is 0 only if every run is OK (warnings allowed), so the script can gate a workflow:
`python vasp_diagnose.py run && python band_alignment.py run`.

## What is checked

| Check | Source | Verdict |
|---|---|---|
| status | final timing block of OUTCAR, age of OUTCAR, logs | finished / running? / stopped |
| SCF of every ionic step | `aborting loop because EDIFF is reached` vs `... EDIFF was not reached (unconverged)` | FAIL if the last loop failed, WARN for intermediate ones |
| ionic convergence | `reached required accuracy`, last `TOTAL-FORCE` block, `EDIFFG` | max force on the **free** coordinates (what VASP compares with EDIFFG) |
| constraints | `Selective dynamics` flags in POSCAR, POSCAR vs CONTCAR | number of fixed atoms per element, forces on them, whether they moved |
| cell | `volume of cell` (first and last), `ISIF` | WARN above 3 % volume change |
| logs | `*.out`, `*.err`, `slurm-*`, `stdout*`, `vasp.out` | known messages (table below) |

### Known messages

| Message | Usual cause | What to try |
|---|---|---|
| `DUE TO TIME LIMIT` | wall time reached | CONTCAR → POSCAR and resubmit; an auto-restart job script |
| `oom-kill`, `Out Of Memory` | not enough memory per task | more nodes, fewer tasks per node, NCORE/KPAR, `LREAL = Auto` |
| `ZBRENT: fatal error` | the CG line minimisation lost its bracket | restart from CONTCAR; `IBRION = 1` or smaller `POTIM`; tighter `EDIFF` |
| `Error EDDDAV` | Davidson diagonalisation failed | `ALGO = Normal`/`All`; check for atoms too close |
| `BRMIX: very serious problems` | density mixing broke down | start without WAVECAR/CHGCAR; change `AMIX`/`BMIX` |
| `ZPOTRF` | orbitals became linearly dependent (diverging SCF, overlapping atoms) | see [`SCF_Rescue_Hybrid`](../SCF_Rescue_Hybrid) |
| `Sub-Space-Matrix is not hermitian` | Davidson instability | `ALGO = Normal`/`All`, `PREC = Accurate` |
| `internal error in subroutine SGRCON / IBZKPT / PRICEL` | symmetry analysis failed | symmetrise, change `SYMPREC`, or `ISYM = 0` |
| `FEXCF` / `FEXCP` | density outside the XC table: atoms far too close | fix the geometry |
| `segmentation fault` | stack size, memory, binary/MPI mismatch | `ulimit -s unlimited` in the job script |

## Options

| Option | Meaning |
|---|---|
| `--summary` | one line per run |
| `--only-problems` | hide runs whose state is exactly OK |
| `--running-minutes 20` | an unfinished OUTCAR younger than this counts as still running |
| `--csv FILE` | one row per run |
| `--no-color` | plain output (also when `NO_COLOR` is set or the output is not a terminal) |

## Good practice

- Run the check **before** any analysis script, and keep its CSV with the results.
- After any restart chain, run `python ../Selective_Dynamics/sd_tool.py report CONTCAR` to see the flags
  that will travel to the next stage.
- A relaxation is converged when the forces on the free atoms are below |EDIFFG| *and* the structure is
  the one you meant to relax.
