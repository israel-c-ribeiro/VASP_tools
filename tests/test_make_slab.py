import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("scipy")

from ase.io import read  # noqa: E402

from conftest import load  # noqa: E402

ms = load("Slab_Whole_Molecules/make_slab.py")


@pytest.fixture
def bulk(examples):
    root, _ = examples
    return read(root / "layered_bulk" / "POSCAR", format="vasp")


def test_naive_recipe_splits_a_molecule(bulk):
    slab = ms.naive_slab(bulk, vacuum=15.0)
    assert ms.count_fragments(slab) > 3                    # 2 molecules + 1 layer expected


def test_whole_molecule_slab(bulk):
    slab = ms.whole_molecule_slab(bulk, ["Pb", "I"], "Pb", "N", vacuum=15.0)
    assert len(slab) == len(bulk)
    assert ms.count_fragments(slab) == 3
    xh = ms.xh_bonds(slab, periodic=False)
    assert xh.max() < 1.15
    gap = slab.cell.lengths()[2] - np.ptp(slab.positions[:, 2])
    assert gap == pytest.approx(15.0, abs=1e-6)
    # atoms moved by lattice vectors only: every covalent bond of the bulk is still there, with the
    # same length, once the stacking axis is open (no periodic image across the vacuum)
    open_slab = slab.copy()
    open_slab.pbc = [True, True, False]
    assert np.allclose(bonds(bulk), bonds(open_slab), atol=1e-6)


def bonds(atoms, cutoff=2.0):
    d = atoms.get_all_distances(mic=True)
    return np.sort(d[(d > 0) & (d < cutoff)])


def test_two_layers_per_cell_are_refused(bulk):
    with pytest.raises(ValueError, match="inorganic layers"):
        ms.whole_molecule_slab(bulk.repeat((1, 1, 2)), ["Pb", "I"], "Pb")


def test_cli_and_species_order(examples, tmp_path):
    root, _ = examples
    out = tmp_path / "POSCAR_slab"
    assert ms.main([str(root / "layered_bulk" / "POSCAR"), "-o", str(out), "--order", "C,H,I,N,Pb"]) == 0
    assert read(out, format="vasp").get_chemical_symbols()[:2] == ["C", "C"]
    assert ms.main([str(root / "layered_bulk" / "POSCAR"), "-o", str(out), "--naive"]) == 1
