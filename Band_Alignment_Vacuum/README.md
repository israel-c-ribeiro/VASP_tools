# Band alignment from the vacuum level

**Absolute band edges (ionization potential, electron affinity, work function) and fragment-resolved
band offsets from VASP slab calculations.**

<p align="center"><img src="../docs/img/band_alignment.png" width="430" alt="Energy-level diagram of two synthetic slabs"/></p>

## Why

Every periodic DFT code sets the average electrostatic potential of the cell to zero, so the
Kohn–Sham eigenvalues of two different calculations do not share a common energy zero. A VBM of
−1.23 eV in one run and −0.98 eV in another tells you nothing about which material holds its holes
more tightly. To compare runs — two surfaces, a family of molecules on the same substrate, a hundred
conformers of a hybrid material, the two sides of a heterostructure — every eigenvalue must first be
referred to a common level. For a slab with vacuum, that level is the **vacuum level**.

`band_alignment.py` does this in one command for one run or a whole campaign, and refuses runs whose
numbers cannot be trusted (unfinished, SCF not converged, no vacuum).

## The physics in five lines

1. Average the electrostatic potential over planes parallel to the surface: $\bar V(z)$.
2. Far from the slab $\bar V(z)$ is flat; that plateau is the vacuum level $E_\mathrm{vac}$.
3. Ionization potential $\mathrm{IP} = E_\mathrm{vac} - E_\mathrm{VBM}$, electron affinity $\mathrm{EA} = E_\mathrm{vac} - E_\mathrm{CBM}$.
4. Work function $\Phi = E_\mathrm{vac} - E_F$ (meaningful for metals; for a semiconductor report IP and EA).
5. With `--fragment`, each state $\psi_{nk}$ gets the weight of fragment A from PROCAR,
   $f_A = \sum_{i\in A} w_i \big/ \sum_{i} w_i$, and the edges of A and B are found separately.

The fragment edges give the **band-alignment type** between A and B:

| Type | Condition | Consequence |
|---|---|---|
| I (A inside B) | VBM_B ≤ VBM_A and CBM_A ≤ CBM_B | electrons and holes both confined in A |
| I (B inside A) | VBM_A ≤ VBM_B and CBM_B ≤ CBM_A | electrons and holes both confined in B |
| II (staggered) | both edges of one fragment above those of the other | electrons in one fragment, holes in the other |
| III (broken gap) | the CBM of one fragment lies below the VBM of the other | charge transfer in the ground state |

In a hybrid perovskite, A = the inorganic layer (Pb, I) and B = the organic cations: type I confines
electrons and holes in the inorganic layer, type II separates them.

## Setting up the VASP run

```
LVHAR  = .TRUE.    ! write the electrostatic potential (ionic + Hartree) to LOCPOT
LORBIT = 11        ! site- and lm-projected PROCAR (needed only for --fragment)
! asymmetric slabs only:
IDIPOL = 3         ! dipole correction along c
LDIPOL = .TRUE.
```

- Use at least ~15 Å of vacuum and check that the plateau is flat (`vac_spread` < 0.05 eV).
- Include Γ in the k-mesh if you use `--gamma-only`, and enough empty bands (`NBANDS`) to reach the CBM
  of both fragments.
- Semilocal functionals underestimate gaps; hybrid functionals and, for Pb/Bi/Sn compounds, spin–orbit
  coupling (SOC) change the edges substantially. The script reads collinear, spin-polarized and
  non-collinear (SOC) outputs, including the four-block LOCPOT of a non-collinear run.

Files needed in each run folder: `OUTCAR`, `EIGENVAL`, `CONTCAR` (or `POSCAR`), `LOCPOT`, and `PROCAR`
for `--fragment`.

## Quick start with the example data

The folder [`examples/`](../examples) contains two synthetic slabs written by
[`synthetic_vasp.py`](../examples/synthetic_vasp.py), with a known answer: vacuum level +2.00 eV,
VBM −3.80 eV, CBM −1.60 eV, so IP = 5.80 eV and EA = 3.60 eV.

```bash
cd VASP_tools
python Band_Alignment_Vacuum/band_alignment.py "examples/synthetic_slab*" \
       --fragment Pb,I --labels inorganic,organic --csv edges.csv --plot levels.png
```

```
run                            VBM_al   CBM_al    gap     IP     EA    Phi  dV_dip
synthetic_slab                 -5.800   -3.600  2.200  5.800  3.600  5.700   0.000
synthetic_slab_typeII          -5.600   -3.600  2.000  5.600  3.600  5.500   0.000

run                           VBM_inorganic    VBM_organic  CBM_inorganic    CBM_organic  alignment
synthetic_slab                       -5.800         -5.950         -3.600         -1.500  I (A inside B)
synthetic_slab_typeII                -5.800         -5.600         -3.600         -1.500  II (staggered)
```

In the second slab the cation HOMO lies 0.20 eV above the inorganic VBM: the true VBM of the slab is
now an organic state (`fA_VBM` ≈ 0.04 in the CSV) and the alignment is type II.

## Using it on your own runs

```bash
python band_alignment.py run_dir                                   # one run, true edges only
python band_alignment.py "slabs/*" --csv edges.csv --jobs 8        # a campaign, in parallel
python band_alignment.py "slabs/*" --fragment Pb,I --labels inorganic,organic --gamma-only
python band_alignment.py run_dir --side upper                      # vacuum above the top surface only
```

| Option | Meaning |
|---|---|
| `--fragment Pb,I` | elements of fragment A; every other atom is fragment B |
| `--threshold 0.70` | minimum fragment weight for a state to count as an A or B edge (mixed states are skipped) |
| `--gamma-only` | take the fragment-A edges at Γ only (e.g. a direct-gap inorganic layer) |
| `--side mean/upper/lower` | which vacuum plateau to use when the two sides differ (dipole) |
| `--min-vacuum 8` | runs whose largest atom-free gap is thinner than this (Å) are rejected as bulk cells |
| `--allow-unconverged` | keep runs whose last SCF loop did not reach EDIFF (not recommended) |

## Output columns

| Column | Meaning |
|---|---|
| `VBM`, `CBM`, `gap` | true edges (any k, any character), raw Kohn–Sham energies (eV) |
| `V_vac`, `V_upper`, `V_lower`, `dipole_step` | vacuum level used, the two plateaus and their difference |
| `*_al` | energy referred to the vacuum level (`E − V_vac`) |
| `IP`, `EA`, `work_function` | `V_vac − VBM`, `V_vac − CBM`, `V_vac − E_F` |
| `VBM_A`, `CBM_A`, `VBM_B`, `CBM_B` | fragment edges (with `--fragment`) |
| `offset_VB`, `offset_CB`, `alignment` | `VBM_B − VBM_A`, `CBM_B − CBM_A`, type I/II/III |
| `fA_VBM`, `fA_CBM` | weight of fragment A in the true VBM and CBM |
| `vac_spread`, `vac_width` | flatness of the plateau (eV) and thickness of the vacuum (Å) |

## Good practice and pitfalls

- **Converge the vacuum, not only the k-points.** If `vac_spread` exceeds a few 0.01 eV, add vacuum.
- **Asymmetric slabs carry a dipole.** Without a dipole correction the vacuum potential is sloped; with
  it, the two plateaus differ by a step, and each surface has its own IP. The script reports both.
- **Band edges come from the k-points of the run.** For an indirect or off-mesh gap, compute a band
  structure along a path and use those eigenvalues.
- **The character threshold is a choice.** Mixed states (0.3 < f_A < 0.7) are excluded from the
  fragment edges; check `fA_VBM`/`fA_CBM` and repeat with another `--threshold` before drawing conclusions.
- **Compare like with like:** same functional, ENCUT, PAW set, SOC treatment and vacuum for every run
  of a series.
- **A bulk cell has no vacuum level.** Bulk band offsets need an interface calculation or a reference
  such as the macroscopic average or a core level (see [`LOCPOT_Planar_Average`](../LOCPOT_Planar_Average)).

## References

- VASP wiki: [LVHAR](https://www.vasp.at/wiki/index.php/LVHAR), [LORBIT](https://www.vasp.at/wiki/index.php/LORBIT),
  [IDIPOL](https://www.vasp.at/wiki/index.php/IDIPOL).
- C. G. Van de Walle and R. M. Martin, *Phys. Rev. B* **35**, 8154 (1987) — band offsets from
  potential alignment.
- J. Neugebauer and M. Scheffler, *Phys. Rev. B* **46**, 16067 (1992) — dipole correction for slabs.
