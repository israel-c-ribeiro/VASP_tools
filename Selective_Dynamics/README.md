# Selective dynamics: inspect, release and set constraints

**Report which atoms of a POSCAR/CONTCAR are fixed (and the forces left on them), remove every
constraint, or fix atoms by height, element or index.**

## Why

With `Selective dynamics`, each coordinate in POSCAR carries a flag: `T` (may move) or `F` (fixed).
VASP writes the same flags into CONTCAR, so they silently travel to every calculation built from it:
a restart, a cell relaxation, a slab cut from the relaxed bulk. Because the forces on fixed coordinates
are not part of the convergence criterion, a relaxation can report *reached required accuracy* while
part of the structure never moved and still feels forces of 1 eV/Å.

The two-minute habit that prevents it: **report the flags at the start of every new stage.**

## Quick start with the example data

```bash
python Selective_Dynamics/sd_tool.py report examples/relax_inherited_constraints/CONTCAR \
       --outcar examples/relax_inherited_constraints/OUTCAR
```

```text
21 atoms, species Pb1 I4 C2 N2 H12
selective dynamics: 5 atom(s) with at least one fixed coordinate, 5 completely fixed

element   atoms  fixed x  fixed y  fixed z   indices (1-based)
Pb            1        1        1        1   1
I             4        4        4        4   2-5
C             2        0        0        0   -
N             2        0        0        0   -
H            12        0        0        0   -

last ionic step: max |F| on free coordinates 0.009 eV/Å (the quantity compared with EDIFFG)
                 max |F| on fixed atoms     1.030 eV/Å (atom 3, I) — not part of the convergence test
```

Release everything and restart:

```bash
python Selective_Dynamics/sd_tool.py free examples/relax_inherited_constraints/CONTCAR -o POSCAR
```

## Fixing atoms on purpose

All selections are intersected, so they can be combined:

```bash
# freeze the bottom of a slab (fractional c < 0.30): the usual surface-calculation setup
python sd_tool.py fix POSCAR -o POSCAR_fixed --below 0.30

# freeze the bottom 8 Å, but only the inorganic atoms
python sd_tool.py fix POSCAR -o POSCAR_fixed --below-A 8.0 --elements Pb,I

# allow in-plane motion only (fix the third coordinate) for atoms 1-12 and 20
python sd_tool.py fix POSCAR -o POSCAR_fixed --indices 1-12,20 --directions z

# add to the constraints already present instead of replacing them
python sd_tool.py fix CONTCAR -o POSCAR_fixed --elements N --keep-existing
```

Notes:

- Indices are 1-based, as in VESTA and in the OUTCAR force table.
- The flags act on the coordinates as they are written in the file. With `Direct` coordinates
  (what this tool writes), "x y z" means the fractional coordinates along a, b and c. In a variable-cell
  relaxation (ISIF = 3), fixed atoms keep their *fractional* positions and therefore still move with the
  cell.
- The output is always a VASP 5 POSCAR in `Direct` coordinates; the comment line, cell and species
  order are preserved.

## Commands

| Command | Purpose | Main options |
|---|---|---|
| `report FILE` | flags per element and direction | `--outcar OUTCAR` for the residual forces |
| `free FILE -o OUT` | remove the `Selective dynamics` line | — |
| `fix FILE -o OUT` | fix a selection | `--below/--above` (fractional), `--below-A/--above-A` (Å), `--axis`, `--elements`, `--indices`, `--directions`, `--keep-existing` |

See also [`Run_Diagnostics`](../Run_Diagnostics), which runs the same constraint check as part of a
full run diagnosis.
