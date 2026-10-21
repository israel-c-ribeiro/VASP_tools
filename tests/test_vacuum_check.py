import csv
import gzip
import shutil

import numpy as np
import pytest

from conftest import load
import synthetic_vacuum
import synthetic_vasp

vc = load("Vacuum_Potential_Check/vacuum_check.py")
pa = load("LOCPOT_Planar_Average/planar_average.py")


@pytest.fixture(scope="module")
def vac(tmp_path_factory):
    out = tmp_path_factory.mktemp("vacuum")
    return out, synthetic_vacuum.make_vacuum_examples(out)


def test_clean_vacuum_is_ok(vac):
    root, truth = vac
    r = vc.check(root / "vacuum_clean")
    assert abs(r["cos_amp_meV"]) < 0.5 and abs(r["ramp_meV_per_A"]) < 0.05 and r["step_eV"] == 0.0
    assert r["V_vac_mean_eV"] == pytest.approx(truth["vacuum_clean"]["v_vac"], abs=1e-3)
    assert r["vac_gap_A"] == pytest.approx(truth["vacuum_clean"]["vac_gap_A"], abs=1e-6)
    assert r["rms_meV"] < 1.0
    assert r["verdict"] == "OK" and r["reasons"] == []
    assert r["EDIFF"] == "1E-6" and r["LVHAR"] == ".TRUE."


def test_cosine_residual_is_recovered_from_a_noncollinear_locpot(vac):
    root, truth = vac
    t = truth["vacuum_residual"]
    assert t["nblocks"] == 4
    r = vc.check(root / "vacuum_residual" / "LOCPOT")
    assert r["cos_amp_meV"] == pytest.approx(1000 * t["cos_amp_eV"], abs=0.5)
    assert r["cos_amp_free_meV"] == pytest.approx(1000 * t["cos_amp_eV"], abs=1.0)
    assert abs(r["ramp_meV_per_A"]) < 0.05
    assert r["verdict"] == "BAD" and "long-wavelength" in r["reasons"][0]
    assert r["EDIFF"] == "1E-3" and r["NELM"] == "12"


def test_dipole_ramp_is_recovered_and_does_not_leak_into_the_cosine(vac):
    root, truth = vac
    t = truth["vacuum_ramp"]
    r = vc.check(root / "vacuum_ramp")
    assert r["ramp_meV_per_A"] == pytest.approx(1000 * t["ramp_eV_per_A"], abs=0.05)
    assert abs(r["cos_amp_meV"]) < 0.5
    # upper side (just above the top surface) minus lower side: -ramp x distance between the side centres
    assert r["two_side_diff_eV"] == pytest.approx(-t["ramp_eV_per_A"] * (r["window_A"] - 1.0), abs=0.01)
    assert r["verdict"] == "WARN" and "IDIPOL" in r["reasons"][0]


def test_dipole_correction_step_is_fitted(vac):
    root, truth = vac
    t = truth["vacuum_dipole_step"]
    r = vc.check(root / "vacuum_dipole_step")
    assert r["step_eV"] == pytest.approx(t["step_eV"], abs=0.003)
    assert r["two_side_diff_eV"] == pytest.approx(t["step_eV"], abs=0.003)
    assert abs(r["cos_amp_meV"]) < 0.5 and abs(r["ramp_meV_per_A"]) < 0.05
    assert r["LDIPOL"] == ".TRUE." and r["verdict"] == "OK"
    # without the step term the jump wrecks the straight-line model
    off = vc.check(root / "vacuum_dipole_step", step="off")
    assert off["rms_meV"] > 20 and off["verdict"] != "OK"


def test_step_without_ldipol_is_flagged(tmp_path):
    t = synthetic_vacuum.make_vacuum_example(tmp_path / "run", step=0.2)       # default INCAR: no LDIPOL
    r = vc.check(tmp_path / "run")
    assert r["step_eV"] == pytest.approx(t["step_eV"], abs=0.003)
    assert r["verdict"] == "WARN" and "LDIPOL is not set" in " ".join(r["reasons"])


@pytest.mark.parametrize("warn,bad,expected", [(5, 20, "WARN"), (5, 8, "BAD"), (15, 20, "OK")])
def test_thresholds(tmp_path, warn, bad, expected):
    synthetic_vacuum.make_vacuum_example(tmp_path / "run", cos_amp=0.010)
    r = vc.check(tmp_path / "run", thresholds=dict(warn=warn, bad=bad))
    assert r["cos_amp_meV"] == pytest.approx(10.0, abs=0.5)
    assert r["verdict"] == expected


def test_streaming_reader_matches_a_full_read(vac):
    root, _ = vac
    path = root / "vacuum_residual" / "LOCPOT"
    grid, cell, symbols, cart = pa.read_locpot(path)
    for axis in (0, 1, 2):
        prof, cell2, sym2, cart2 = vc.read_locpot_profile(path, axis=axis, chunk_lines=7)   # many tiny chunks
        assert np.allclose(prof, pa.planar_average(grid, axis))
    assert np.allclose(cell, cell2) and sym2 == symbols and np.allclose(cart, cart2)


def test_gzipped_locpot(vac, tmp_path):
    root, _ = vac
    d = tmp_path / "gz"
    d.mkdir()
    with open(root / "vacuum_residual" / "LOCPOT", "rb") as fi, gzip.open(d / "LOCPOT.gz", "wb") as fo:
        shutil.copyfileobj(fi, fo)
    r = vc.check(d)
    assert r["cos_amp_meV"] == pytest.approx(50.0, abs=0.5) and r["incar"] is False


def test_truncated_locpot_fails_loudly(vac, tmp_path):
    root, _ = vac
    text = (root / "vacuum_clean" / "LOCPOT").read_text()
    (tmp_path / "LOCPOT").write_text(text[: len(text) // 2])
    with pytest.raises(ValueError, match="truncated"):
        vc.read_locpot_profile(tmp_path / "LOCPOT")


def test_thin_vacuum_is_rejected(tmp_path):
    symbols, cell, frac = synthetic_vasp.toy_slab(c=16.0, layer_z=8.0)        # ~5 Å of vacuum
    synthetic_vasp.write_locpot(tmp_path / "LOCPOT", symbols, cell, frac, np.zeros((2, 2, 100)))
    with pytest.raises(RuntimeError, match="no fit window"):
        vc.check(tmp_path / "LOCPOT")
    assert vc.main([str(tmp_path / "LOCPOT")]) == 1


def test_incar_parser_and_lvtot_warning(tmp_path):
    synthetic_vacuum.make_vacuum_example(tmp_path / "run")
    (tmp_path / "run" / "INCAR").write_text("ALGO = Fast ; AMIX = 0.3 ! comment\nbmix=0.5 # c\nLVTOT = .TRUE.\n")
    tags = vc.read_incar(tmp_path / "run")
    assert tags == {"ALGO": "Fast", "AMIX": "0.3", "BMIX": "0.5", "LVTOT": ".TRUE."}
    r = vc.check(tmp_path / "run")
    assert r["verdict"] == "WARN" and "LVTOT" in r["reasons"][0]


def test_cli_batch_csv_and_plot(vac, tmp_path, capsys):
    root, _ = vac
    out_csv, out_png = tmp_path / "vacuum.csv", tmp_path / "vacuum.png"
    rc = vc.main([str(root / "vacuum_*"), str(root / "does_not_exist"),
                  "--csv", str(out_csv), "--plot", str(out_png)])
    assert rc == 2                                           # one path failed, the others were analysed
    out = capsys.readouterr().out
    assert "4 runs: 2 OK, 1 WARN, 1 BAD" in out
    rows = list(csv.DictReader(open(out_csv)))
    assert [r["run"] for r in rows] == ["vacuum_clean", "vacuum_dipole_step", "vacuum_ramp", "vacuum_residual"]
    assert {r["run"]: r["verdict"] for r in rows}["vacuum_residual"] == "BAD"
    assert out_png.stat().st_size > 10_000


def test_version_flag(capsys):
    with pytest.raises(SystemExit):
        vc.main(["--version"])
    assert vc.__version__ in capsys.readouterr().out
