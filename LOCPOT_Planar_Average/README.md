# Planar and macroscopic average of LOCPOT

**Electrostatic-potential profiles, vacuum level and work function from a VASP LOCPOT, with a
publication-ready figure.**

<p align="center"><img src="../docs/img/planar_average.png" width="560" alt="Planar and macroscopic average of a synthetic slab"/></p>

## Why

The planar-averaged potential is the first thing to look at after a slab or interface calculation:

- it **shows whether the vacuum is thick enough** (a flat plateau) and whether the slab carries a
  dipole (a slope, or a step between the two sides);
- its plateau is the **vacuum level**, the common reference for ionization potentials, electron
  affinities and work functions;
- its **macroscopic average** — the planar average averaged once more over one period of the
  structure — removes the atomic oscillations and gives the potential lineup used for band offsets
  at interfaces.

## Setting up the VASP run

```
LVHAR = .TRUE.     ! LOCPOT = ionic + Hartree potential: the right quantity for a vacuum level
```

`LVTOT = .TRUE.` writes the total local potential instead, which also contains the
exchange–correlation potential; it decays slowly into the vacuum and is not used for vacuum levels.

## Quick start with the example data

```bash
python LOCPOT_Planar_Average/planar_average.py examples/synthetic_slab/LOCPOT \
       --macro 6.4 --plot profile.png --csv profile.csv
```

```
grid (8, 8, 160), 21 atoms, profile length 30.000 Å along axis 2
vacuum gap           19.14 Å
V_vac upper side     2.000 eV
V_vac lower side     2.000 eV   (difference +0.000 eV)
plateau spread       0.002 eV
E_fermi             -3.700 eV
work function        5.700 eV  (upper 5.700, lower 5.700)
```

`E_fermi` is read from the `OUTCAR` next to the LOCPOT (or from `--outcar`).

## Step by step

1. **Planar average.** The 3D grid is averaged over the two in-plane directions (`--axis` selects the
   surface normal; default c). The x axis of the profile is the distance along the normal, so tilted
   cells are handled correctly.
2. **Vacuum plateaus.** The largest atom-free gap along the normal is the vacuum. Slices closer than
   `--margin` Å (default 4) to the slab are discarded, and the rest is split in an *upper* half (above
   the top surface) and a *lower* half (below the bottom surface). Their means are the two vacuum
   levels; the largest peak-to-peak variation inside them is the `plateau spread`.
3. **Macroscopic average** (`--macro L`): a running average of the planar average over a window of
   L Å. Choose L equal to the period of the structure along the normal (the interlayer spacing, or the
   lattice parameter of each side of an interface); then the bulk-like regions become flat.
4. **Work function** = vacuum level − E_F, reported for each side.

## Reading the figure

| Feature | Meaning |
|---|---|
| flat plateaus at both ends, equal height | enough vacuum, no net dipole |
| plateaus that are not flat (large spread) | vacuum too thin: the slab images still interact |
| a linear slope across the vacuum | net dipole without correction: use `IDIPOL`/`LDIPOL` |
| two flat plateaus at different heights | dipole-corrected asymmetric slab: each surface has its own work function |

## Options

| Option | Meaning |
|---|---|
| `--axis 0/1/2` | surface normal along a, b or c (default c) |
| `--macro L` | window (Å) of the macroscopic average |
| `--outcar FILE` | where to read E_F (default: `OUTCAR` next to the LOCPOT) |
| `--margin 4.0` | distance (Å) from the slab excluded from the plateau |
| `--plot FILE`, `--csv FILE` | figure (PNG/PDF/SVG) and profile table (`z_A, planar_eV, macro_eV`) |

## References

- VASP wiki: [LVHAR](https://www.vasp.at/wiki/index.php/LVHAR), [LOCPOT](https://www.vasp.at/wiki/index.php/LOCPOT).
- A. Baldereschi, S. Baroni and R. Resta, *Phys. Rev. Lett.* **61**, 734 (1988) — the macroscopic average.
- See also [`Band_Alignment_Vacuum`](../Band_Alignment_Vacuum), which uses the same plateau algorithm
  to put band edges on an absolute scale.
