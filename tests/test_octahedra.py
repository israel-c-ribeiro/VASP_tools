import csv

import numpy as np
import pytest

from conftest import load

import synthetic_octahedra as so
import synthetic_vasp

oc = load("Octahedral_Distortion/octahedra.py")


@pytest.fixture(scope="module")
def octa(tmp_path_factory):
    out = tmp_path_factory.mktemp("octahedra")
    return out, so.make_octahedra_examples(out)


def check(r, t):
    s = r["summary"]
    assert len(r["octahedra"]) == t["n_centres"] and r["n_complete"] == t["n_centres"]
    assert len(r["bridging"]) == t["n_bridging"] and len(r["terminal"]) == t["n_terminal"]
    assert r["n_layers"] == t["n_layers"]
    for key, ref, tol in (("bond", t["bond_mean"], 1e-6), ("baur", t["baur"], 1e-7),
                          ("lambda", t["lambda_"], 1e-7), ("sigma2", t["sigma2"], 1e-5),
                          ("volume", t["volume"], 1e-4), ("d_eq", t["d_eq"], 1e-6), ("d_ax", t["d_ax"], 1e-6),
                          ("bxb", t["bxb"], 1e-5), ("tilt", t["tilt"], 1e-5),
                          ("area", t["area_per_centre"], 1e-4)):
        mean, lo, hi = s[key]
        assert mean == pytest.approx(ref, abs=tol), key
        if key != "bond":
            assert hi - lo < 10 * tol, key                   # every octahedron is the same


def test_ideal_sheet(octa):
    root, truth = octa
    r = oc.analyse(root / "octahedra_ideal" / "POSCAR")
    check(r, truth["octahedra_ideal"])
    s = r["summary"]
    assert s["baur"][0] == pytest.approx(0, abs=1e-9) and s["lambda"][0] == pytest.approx(1, abs=1e-9)
    assert s["sigma2"][0] == pytest.approx(0, abs=1e-8) and s["bxb"][0] == pytest.approx(180)
    assert s["tilt"][0] == pytest.approx(0, abs=1e-6)


def test_known_distortion(octa):
    root, truth = octa
    t = truth["octahedra_distorted"]
    # the closed forms of the generator, written out once more
    assert t["baur"] == pytest.approx(4 / 9 * 0.10 / ((4 * 3.20 + 2 * 3.30) / 6))
    assert t["sigma2"] == pytest.approx(4 * 64 / 11) and t["bxb"] == 156.0
    check(oc.analyse(root / "octahedra_distorted" / "POSCAR"), t)


def test_skewed_cell_and_slab_match_bulk(octa):
    root, truth = octa
    ref = oc.analyse(root / "octahedra_distorted")
    for name in ("octahedra_skewed", "octahedra_slab"):
        r = oc.analyse(root / name)                          # folder -> its POSCAR
        check(r, truth[name])
        for a, b in zip(ref["octahedra"], r["octahedra"]):
            for k in ("d_mean", "baur", "lambda", "sigma2", "bxb", "tilt"):
                assert a[k] == pytest.approx(b[k], abs=1e-6)
    skew = so.octahedral_sheet(3.2, 3.3, 12, 8, skew=True)[1]
    gamma = np.degrees(np.arccos(skew[0] @ skew[1] / np.linalg.norm(skew[0]) / np.linalg.norm(skew[1])))
    assert gamma < 30                                        # really a skewed cell


def test_auto_cutoff_equals_explicit(octa):
    root, _ = octa
    a = oc.analyse(root / "octahedra_slab" / "POSCAR")
    b = oc.analyse(root / "octahedra_slab" / "POSCAR", cutoff=3.8)
    assert 3.3 < a["cutoff"] < 5.1 and a["cutoff_method"] == "largest gap"
    assert a["summary"]["baur"] == pytest.approx(b["summary"]["baur"])


def test_one_octahedron_per_cell_images():
    """1 Pb per cell: each bridging I bonds to the same Pb through two periodic images."""
    sym, cell, frac = synthetic_vasp.toy_slab()
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "POSCAR"
        synthetic_vasp.write_poscar(p, sym, cell, frac)
        r = oc.analyse(p)
    o = r["octahedra"][0]
    assert o["n_ligands"] == 6 and o["n_bridging"] == 4 and o["n_terminal"] == 2
    assert len(r["bridging"]) == 2 and len(r["terminal"]) == 2
    assert o["d_eq"] == pytest.approx(3.15) and o["d_ax"] == pytest.approx(3.2)
    assert r["summary"]["bxb"][0] == pytest.approx(180) and r["summary"]["area"][0] == pytest.approx(6.3 ** 2)
    m = (4 * 3.15 + 2 * 3.2) / 6
    assert o["baur"] == pytest.approx(4 / 9 * 0.05 / m)


def test_off_centre_octahedron_volume():
    """Regular octahedron (vertex distance r) with the centre moved by e along z: volume unchanged."""
    r, e = 3.0, 0.4
    v = np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r]]) - [0, 0, e]
    baur, lam, sigma2, vol = oc.octahedron_shape(v)
    d = np.linalg.norm(v, axis=1)
    assert vol == pytest.approx(4 / 3 * r ** 3)
    assert baur == pytest.approx(np.mean(np.abs(d - d.mean())) / d.mean())
    assert lam == pytest.approx(np.mean(d ** 2) / r ** 2)
    # cis angles: 4 equatorial-equatorial, 4 with the +z ligand, 4 with the -z ligand
    a_up = np.degrees(np.arccos(np.dot(v[4], v[0]) / (d[4] * d[0])))
    a_dn = np.degrees(np.arccos(np.dot(v[5], v[0]) / (d[5] * d[0])))
    a_eq = np.degrees(np.arccos(np.dot(v[0], v[2]) / (d[0] * d[2])))
    ref = (4 * (a_up - 90) ** 2 + 4 * (a_dn - 90) ** 2 + 4 * (a_eq - 90) ** 2) / 11
    assert sigma2 == pytest.approx(ref) and sigma2 > 0


def test_reader_cartesian_negative_scale(octa, tmp_path):
    root, _ = octa
    sym, cell, frac = oc.read_poscar(root / "octahedra_distorted" / "POSCAR")
    vol = abs(np.linalg.det(cell))
    names = list(dict.fromkeys(sym))
    lines = ["cartesian, negative scale", f"{-vol:.10f}"]
    lines += [" ".join(f"{x / 2:.12f}" for x in v) for v in cell]          # cell written at half size
    lines += [" ".join(names), " ".join(str(sym.count(n)) for n in names), "Selective dynamics", "Cartesian"]
    lines += [" ".join(f"{x / 2:.12f}" for x in p) + " T T F" for p in frac @ cell]
    (tmp_path / "POSCAR").write_text("\n".join(lines) + "\n")
    s2, c2, f2 = oc.read_poscar(tmp_path / "POSCAR")
    assert s2 == sym and np.allclose(c2, cell) and np.allclose(f2, frac)


def test_normal_along_a(octa, tmp_path):
    """The same slab with the stacking axis written first: --normal a recovers the tilt."""
    root, truth = octa
    sym, cell, frac = oc.read_poscar(root / "octahedra_slab" / "POSCAR")
    synthetic_vasp.write_poscar(tmp_path / "POSCAR", sym, cell[[2, 0, 1]], frac[:, [2, 0, 1]])
    r = oc.analyse(tmp_path / "POSCAR", normal="a")
    check(r, truth["octahedra_slab"])
    # rows permuted, Cartesian space unchanged: the Cartesian normal is still z
    assert oc.analyse(tmp_path / "POSCAR", normal="0,0,1")["summary"]["tilt"][0] == pytest.approx(8.0)
    assert oc.analyse(tmp_path / "POSCAR")["summary"]["tilt"][0] > 45     # wrong default normal


def test_batch_csv(octa, tmp_path, capsys):
    root, _ = octa
    out = tmp_path / "oct.csv"
    assert oc.main([str(root / "octahedra_*"), "--csv", str(out)]) == 0
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 16 and {r["element"] for r in rows} == {"Pb"}
    ideal = [r for r in rows if "octahedra_ideal" in r["structure"]]
    assert all(float(r["bxb"]) == pytest.approx(180) for r in ideal)
    text = capsys.readouterr().out
    assert "octahedra_skewed" in text and "156.00" in text


def test_compare(octa, tmp_path, capsys):
    root, truth = octa
    out = tmp_path / "cmp.csv"
    assert oc.main(["--compare", str(root / "octahedra_ideal"), str(root / "octahedra_distorted"),
                    "--csv", str(out), "--plot", str(tmp_path / "cmp.png")]) == 0
    text = capsys.readouterr().out
    assert "-24.00" in text and "+8.00" in text
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 4
    for r in rows:
        assert float(r["bxb_change"]) == pytest.approx(-24.0)
        assert float(r["baur_change"]) == pytest.approx(truth["octahedra_distorted"]["baur"])
    assert (tmp_path / "cmp.png").stat().st_size > 10_000


def test_fail_loudly(octa, tmp_path, capsys):
    root, _ = octa
    assert oc.main([str(root / "octahedra_ideal"), "--centers", "Sn"]) == 1
    assert "no centre atom" in capsys.readouterr().err
    assert oc.main([str(tmp_path / "missing")]) == 1
    r = oc.analyse(root / "octahedra_ideal" / "POSCAR", cutoff=2.5)     # too short: no bond
    assert r["n_complete"] == 0 and oc.warnings_for(r)
    assert oc.main([str(root / "octahedra_ideal"), "--cutoff", "2.5"]) == 0
    assert "without exactly 6 ligands" in capsys.readouterr().err
