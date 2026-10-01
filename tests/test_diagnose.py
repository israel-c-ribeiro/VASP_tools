import pytest

from conftest import load
import synthetic_vasp as sv

dg = load("Run_Diagnostics/vasp_diagnose.py")


def levels(notes):
    return [lvl for lvl, _ in notes]


def test_clean_single_point_is_ok(examples):
    root, _ = examples
    rec, notes = dg.diagnose(root / "synthetic_slab")
    assert rec["state"] == "OK"
    assert "FAIL" not in levels(notes) and "WARN" not in levels(notes)


def test_inherited_constraints_are_flagged(examples):
    root, _ = examples
    rec, notes = dg.diagnose(root / "relax_inherited_constraints")
    assert rec["state"].startswith("OK (")                 # converged, but with warnings
    assert rec["n_fixed"] == 5
    assert rec["fmax_fixed"] > 0.5 > rec["fmax_free"]
    assert rec["dV_percent"] == pytest.approx(4.2, abs=0.01)
    text = " ".join(m for _, m in notes)
    assert "selective dynamics" in text and "Pulay" in text


def test_time_limit_is_recognised(examples):
    root, _ = examples
    rec, notes = dg.diagnose(root / "relax_time_limit")
    assert rec["state"] == "stopped"
    assert "wall time" in rec["errors"]
    assert rec["scf_failed_steps"] == 1
    assert any("NOT converged" in m for _, m in notes)


def test_crash_message_is_recognised(examples):
    root, _ = examples
    rec, _ = dg.diagnose(root / "hybrid_scf_oscillating")
    assert rec["state"] == "stopped" and "ZPOTRF" in rec["errors"]


def test_final_scf_failure_is_not_ok(tmp_path):
    sv.make_toy_slab_run(tmp_path / "r")
    sv.write_outcar(tmp_path / "r" / "OUTCAR", natoms=21, scf_converged=[False])
    rec, notes = dg.diagnose(tmp_path / "r")
    assert rec["state"] == "finished, NOT converged"
    assert "FAIL" in levels(notes)


def test_running_job_is_not_called_a_failure(tmp_path):
    sv.make_toy_slab_run(tmp_path / "r")
    sv.write_outcar(tmp_path / "r" / "OUTCAR", natoms=21, finished=False)
    rec, _ = dg.diagnose(tmp_path / "r", running_minutes=60)
    assert rec["state"] == "running?"


def test_cli_exit_code_and_csv(examples, tmp_path):
    root, _ = examples
    out = tmp_path / "health.csv"
    assert dg.main([str(root / "synthetic_slab"), "--summary"]) == 0
    assert dg.main([str(root / "*"), "--summary", "--csv", str(out)]) == 1
    assert out.read_text().count("\n") >= 6
