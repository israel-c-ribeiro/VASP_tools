import shutil

import numpy as np
import pytest

from conftest import load
import synthetic_bands as sb

bd = load("Band_Structure_DOS/bands_dos.py")


@pytest.fixture(scope="module")
def ex(tmp_path_factory):
    out = tmp_path_factory.mktemp("bands")
    return out, sb.make_bands_example(out)


@pytest.mark.parametrize("name", ["bands_nosoc", "bands_soc", "bands_hybrid"])
def test_edges_gap_and_k_location(ex, name):
    root, truth = ex
    t = truth[name]
    bs = bd.load_bandset(root / name)
    assert bs["vbm"] == pytest.approx(t["VBM"], abs=1e-5)
    assert bs["cbm"] == pytest.approx(t["CBM"], abs=1e-5)
    assert bs["gap"] == pytest.approx(t["gap"], abs=1e-5)
    assert bd.where(bs, bs["kv"]) == t["k_VBM"]
    assert bd.where(bs, bs["kc"]) == t["k_CBM"]
    assert np.allclose(bs["kpts"][bs["kv"]], [0.5, 0.5, 0.0])
    assert bs["label"] == t["label"]
    assert bs["eig"].shape[-1] == t["nbands"]


@pytest.mark.parametrize("name", ["bands_nosoc", "bands_hybrid"])
def test_ticks_gamma_and_jump(ex, name):
    root, truth = ex
    bs = bd.load_bandset(root / name)
    assert bs["tlabels"] == truth[name]["ticks"]               # ['Γ', 'X|S', 'Y', 'Γ']
    assert bs["brk"].sum() == 1                                # one discontinuity, between X and S
    j = int(np.flatnonzero(bs["brk"])[0])
    assert np.allclose(bs["kpts"][j - 1], [0.5, 0, 0]) and np.allclose(bs["kpts"][j], [0.5, 0.5, 0])
    assert bs["x"][j] == pytest.approx(bs["x"][j - 1])         # no horizontal gap in the plot
    assert np.all(np.diff(bs["x"]) >= 0)


def test_zero_weight_points_are_the_path(ex):
    root, truth = ex
    bs = bd.load_bandset(root / "bands_hybrid")
    assert len(bs["x"]) == truth["bands_hybrid"]["nk_path"]    # the 16 weighted mesh points are dropped
    ev = bd.read_eigenval(root / "bands_hybrid" / "EIGENVAL")
    assert (ev["weights"] > 0).sum() == 16 and (ev["weights"] == 0).sum() == 60


def test_nelect_fallback_without_occupations_on_the_path(tmp_path):
    t = sb.make_band_run(tmp_path / "h", "hybrid", kind="explicit", zero_occ_on_path=True, n=20)
    bs = bd.load_bandset(tmp_path / "h")
    assert bs["vbm"] == pytest.approx(t["VBM"]) and bs["cbm"] == pytest.approx(t["CBM"])


def test_spin_polarized_bands(tmp_path):
    t = sb.make_band_run(tmp_path / "sp", "nosoc", ispin=2, spin_split=0.1)
    bs = bd.load_bandset(tmp_path / "sp")
    assert bs["ispin"] == 2 and bs["eig"].shape[0] == 2
    assert bs["vbm"] == pytest.approx(t["VBM"]) and bs["cbm"] == pytest.approx(t["CBM"])   # CBM in spin down
    e = bd.band_edges(bs["eig"], None, 8, ncl=False)            # NELECT route for ISPIN = 2
    assert e["vbm"] == pytest.approx(t["VBM"]) and e["cbm"] == pytest.approx(t["CBM"])


def test_indirect_gap_is_reported():
    eig = np.array([[[-1.0, 1.0], [-0.5, 1.2], [-0.8, 0.9]]])  # VBM at k1, CBM at k2
    e = bd.band_edges(eig, None, 2, ncl=False)
    assert (e["kv"], e["kc"]) == (1, 2)
    assert e["gap"] == pytest.approx(1.4) and e["direct_gap"] == pytest.approx(1.7)


def test_kpoints_opt_is_preferred(ex, tmp_path):
    root, _ = ex
    d = tmp_path / "opt"
    shutil.copytree(root / "bands_nosoc", d)
    (d / "EIGENVAL").rename(d / "EIGENVAL_OPT")
    (d / "KPOINTS").rename(d / "KPOINTS_OPT")
    (d / "KPOINTS").write_text("mesh\n0\nGamma\n4 4 1\n")
    bs = bd.load_bandset(d)
    assert bs["eigenval"] == "EIGENVAL_OPT" and bs["tlabels"][1] == "X|S"


def test_labels_and_method_names():
    assert bd.pretty_label("\\Gamma") == "Γ" and bd.pretty_label("GAMMA") == "Γ" and bd.pretty_label("G") == "Γ"
    assert bd.pretty_label("X|G") == "X|Γ"
    assert bd.method_label({"LHFCALC": ".TRUE.", "HFSCREEN": "0.2"}, False) == "HSE06"
    assert bd.method_label({"LHFCALC": "T", "AEXX": "0.25"}, True) == "PBE0+ncl"
    assert bd.method_label({"GGA": "PS", "IVDW": "12", "LSORBIT": ".TRUE."}, True) == "PBEsol+D3(BJ)+SOC"


# ----------------------------------------------------------------------------- DOS
def test_dos_per_element_integrals_and_edge_character(ex):
    root, truth = ex
    t = truth["dos_lorbit11"]
    d = bd.load_dos(root / "dos_lorbit11")
    assert d["zero"] == pytest.approx(t["VBM"]) and d["norb"] == 9 and d["ispin"] == 1
    curves = dict(d["curves"])
    assert list(curves) == t["species"]
    for el, n in t["states_all"].items():
        assert bd.integrate(d["energy"], curves[el], -1e9, 1e9)[0] == pytest.approx(n, abs=2e-3)
    for el, v, c in bd.edge_character(d, 1.0):
        assert v == pytest.approx(t["states_vb_edge"][el], abs=2e-3)
        assert c == pytest.approx(t["states_cb_edge"][el], abs=2e-3)
    total = bd.integrate(d["energy"], d["total"], -1e9, 1e9)[0]
    assert total == pytest.approx(sum(t["states_all"].values()), abs=5e-3)


@pytest.mark.parametrize("lorbit", [10, 11])
@pytest.mark.parametrize("ispin,ncl", [(1, False), (2, False), (1, True)])
def test_dos_columns(tmp_path, lorbit, ispin, ncl):
    t = sb.make_dos_run(tmp_path / "d", lorbit=lorbit, ispin=ispin, ncl=ncl)
    d = bd.load_dos(tmp_path / "d", tokens=["Pb", "I:p", "N:s"])
    assert d["norb"] == (3 if lorbit == 10 else 9)
    assert d["pdos"].shape[0] == ispin and d["total"].shape[0] == ispin
    assert d["zero"] == pytest.approx(t["VBM"])
    c = dict(d["curves"])
    assert set(c) == {"Pb", "I-p", "N-s"}
    n_pb = bd.integrate(d["energy"], c["Pb"], -1e9, 1e9)
    n_ip = bd.integrate(d["energy"], c["I-p"], -1e9, 1e9)
    assert n_pb[0] == pytest.approx(t["states_all"]["Pb"], abs=2e-3)     # non-collinear: total component only
    assert n_ip[0] == pytest.approx(t["p_states_all"]["I"], abs=2e-3)
    assert bd.integrate(d["energy"], c["N-s"], -1e9, 1e9)[0] == pytest.approx(0.0, abs=1e-6)
    if ispin == 2:
        assert n_pb[1] == pytest.approx(t["down_scale"] * n_pb[0], rel=1e-3)


def test_dos_column_mismatch_fails_loudly(tmp_path):
    sb.make_dos_run(tmp_path / "d", ncl=True)
    (tmp_path / "d" / "OUTCAR").unlink()
    (tmp_path / "d" / "INCAR").write_text("ISPIN = 1\nLORBIT = 11\n")   # SOC flags lost
    with pytest.raises(ValueError, match="projected columns"):
        bd.load_dos(tmp_path / "d")


# ----------------------------------------------------------------------------- command line
def test_command_line_figure_report_and_csv(ex, tmp_path, capsys):
    root, truth = ex
    out = tmp_path / "fig.png"
    rc = bd.main(["--bands", str(root / "bands_nosoc"), str(root / "bands_soc"), "--dos", str(root / "dos_lorbit11"),
                  "-o", str(out), "--dpi", "80", "--save-data", "--total"])
    assert rc == 0 and out.is_file() and not out.with_suffix(".pdf").exists()
    txt = capsys.readouterr().out
    assert "gap = 1.800 eV (direct)" in txt and "gap = 1.050 eV (direct)" in txt
    b1 = np.loadtxt(tmp_path / "fig_bands_1.csv", delimiter=",", skiprows=1)
    assert b1.shape == (120, 1 + 8)
    assert b1[:, 1:5].max() == pytest.approx(0.0, abs=1e-5)                 # E - E_VBM: VBM at 0
    assert b1[:, 5:].min() == pytest.approx(truth["bands_nosoc"]["gap"], abs=1e-5)
    b2 = np.loadtxt(tmp_path / "fig_bands_2.csv", delimiter=",", skiprows=1)
    assert b2[:, 9:].min() == pytest.approx(truth["bands_soc"]["gap"], abs=1e-5)
    head = (tmp_path / "fig_dos.csv").read_text().splitlines()[0].split(",")
    assert head == ["E", "total", "Pb", "I", "C", "N", "H"]
    dos = np.loadtxt(tmp_path / "fig_dos.csv", delimiter=",", skiprows=1)
    assert dos[:, 0].min() == pytest.approx(-6.5, abs=1e-3)                   # grid starts 6.5 eV below the VBM


def test_command_line_dos_only_spin_polarized(tmp_path):
    sb.make_dos_run(tmp_path / "d", ispin=2)
    assert bd.main(["--dos", str(tmp_path / "d"), "--proj", "Pb:p", "I:p", "-o", str(tmp_path / "x"),
                    "--no-plot", "--save-data", "--ref", "fermi"]) == 0
    head = (tmp_path / "x_dos.csv").read_text().splitlines()[0].split(",")
    assert head == ["E", "total_s1", "total_s2", "Pb-p_s1", "Pb-p_s2", "I-p_s1", "I-p_s2"]


def test_missing_files_fail_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        bd.load_bandset(tmp_path / "nothing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="POSCAR"):
        bd.load_bandset(tmp_path / "empty")
