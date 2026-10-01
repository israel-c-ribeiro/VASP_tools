import numpy as np

from conftest import load

sd = load("Selective_Dynamics/sd_tool.py")


def test_read_flags(examples):
    root, _ = examples
    p = sd.Poscar.read(root / "relax_inherited_constraints" / "POSCAR")
    assert p.flags is not None and (~p.flags.any(axis=1)).sum() == 5
    assert p.symbols[:5] == ["Pb", "I", "I", "I", "I"]


def test_free_then_fix_roundtrip(examples, tmp_path):
    root, _ = examples
    src = root / "relax_inherited_constraints" / "CONTCAR"
    assert sd.main(["free", str(src), "-o", str(tmp_path / "free")]) == 0
    p = sd.Poscar.read(tmp_path / "free")
    assert p.flags is None
    ref = sd.Poscar.read(src)
    assert np.allclose(p.frac, ref.frac) and np.allclose(p.cell, ref.cell)
    assert sd.main(["fix", str(tmp_path / "free"), "-o", str(tmp_path / "fixed"),
                    "--elements", "Pb,I", "--directions", "z"]) == 0
    q = sd.Poscar.read(tmp_path / "fixed")
    assert (~q.flags[:, 2]).sum() == 5 and q.flags[:, :2].all()


def test_fix_by_height(examples, tmp_path):
    root, _ = examples
    sd.main(["fix", str(root / "synthetic_slab" / "POSCAR"), "-o", str(tmp_path / "f"), "--below-A", "14.0"])
    p = sd.Poscar.read(tmp_path / "f")
    z = p.frac[:, 2] * 30.0                                  # c = 30 Å, no atom near 14 Å
    assert np.array_equal(~p.flags.all(axis=1), z < 14.0)


def test_index_helpers():
    assert sd.compress([1, 2, 3, 7, 9, 10]) == "1-3,7,9-10"
    assert sd.parse_indices("1-3,7", 8).tolist() == [True, True, True, False, False, False, True, False]
