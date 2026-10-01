import numpy as np
import pytest

from conftest import load
import synthetic_vasp as sv

ba = load("Band_Alignment_Vacuum/band_alignment.py")


@pytest.mark.parametrize("name", ["synthetic_slab", "synthetic_slab_typeII"])
def test_true_edges_and_vacuum_level(examples, name):
    root, truth = examples
    t = truth[name]
    r = ba.analyze(root / name)
    assert r["VBM"] == pytest.approx(t["VBM"])
    assert r["CBM"] == pytest.approx(t["CBM"])
    assert r["V_vac"] == pytest.approx(t["v_vac"], abs=0.01)
    assert r["IP"] == pytest.approx(t["IP"], abs=0.01)
    assert r["EA"] == pytest.approx(t["EA"], abs=0.01)
    assert r["work_function"] == pytest.approx(t["work_function"], abs=0.01)
    assert r["vac_width"] == pytest.approx(t["vac_width"], abs=0.01)


def test_fragment_edges_type_I_and_II(examples):
    root, truth = examples
    r1 = ba.analyze(root / "synthetic_slab", fragment=["Pb", "I"], gamma_only=True)
    assert r1["VBM_A"] == pytest.approx(-3.80) and r1["CBM_A"] == pytest.approx(-1.60)
    assert r1["VBM_B"] == pytest.approx(-3.95) and r1["CBM_B"] == pytest.approx(0.50)
    assert r1["alignment"].startswith("I (A inside B)")
    r2 = ba.analyze(root / "synthetic_slab_typeII", fragment=["Pb", "I"])
    assert r2["alignment"].startswith("II")
    assert r2["offset_VB"] == pytest.approx(0.20)
    assert r2["fA_VBM"] < 0.3                       # the true VBM is a cation state


def test_soc_files_are_read(tmp_path):
    t = sv.make_toy_slab_run(tmp_path / "soc", soc=True)
    r = ba.analyze(tmp_path / "soc", fragment=["Pb", "I"])
    assert r["soc"] is True
    assert r["V_vac"] == pytest.approx(t["v_vac"], abs=0.01)   # first of the four LOCPOT blocks
    assert r["VBM_A"] == pytest.approx(-3.80)


def test_dipole_step_is_reported(tmp_path):
    sv.make_toy_slab_run(tmp_path / "dip", dipole_step=0.30)
    r = ba.analyze(tmp_path / "dip")
    assert abs(r["dipole_step"]) == pytest.approx(0.30, abs=0.02)


def test_bulk_cell_is_rejected(examples):
    root, _ = examples
    with pytest.raises(RuntimeError, match="no vacuum"):
        ba.analyze(root / "relax_inherited_constraints")


def test_unconverged_scf_is_rejected(tmp_path):
    sv.make_toy_slab_run(tmp_path / "bad")
    sv.write_outcar(tmp_path / "bad" / "OUTCAR", natoms=21, efermi=-3.7, scf_converged=[False])
    with pytest.raises(RuntimeError, match="EDIFF"):
        ba.analyze(tmp_path / "bad")
    assert ba.analyze(tmp_path / "bad", allow_unconverged=True)["VBM"] == pytest.approx(-3.80)


def test_spin_polarized_eigenval(tmp_path):
    eig = np.array([[[-2.0, -1.0, 1.0]], [[-2.1, -0.8, 1.2]]])
    occ = np.array([[[1.0, 1.0, 0.0]], [[1.0, 1.0, 0.0]]])
    sv.write_eigenval(tmp_path / "EIGENVAL", eig, occ, np.zeros((1, 3)), nelect=4, natoms=2)
    e, o, k = ba.read_eigenval(tmp_path / "EIGENVAL")
    assert e.shape == (2, 1, 3)
    res = ba.classify(e, o, efermi=-0.5)
    assert res["VBM"] == pytest.approx(-0.8) and res["CBM"] == pytest.approx(1.0)


@pytest.mark.parametrize("edges, expected", [
    ((-6, -3, -7, -2), "I (A inside B)"),
    ((-7, -2, -6, -3), "I (B inside A)"),
    ((-6, -3, -5.5, -2), "II (staggered)"),
    ((-6, -5, -4, -3), "III (broken gap)"),
])
def test_alignment_type(edges, expected):
    assert ba.alignment_type(*edges) == expected


def test_cli_writes_csv(examples, tmp_path):
    root, _ = examples
    out = tmp_path / "edges.csv"
    code = ba.main([str(root / "synthetic_slab*"), "--fragment", "Pb,I", "--csv", str(out)])
    assert code == 0
    lines = out.read_text().splitlines()
    assert len(lines) == 3 and lines[0].startswith("folder,VBM,CBM")
