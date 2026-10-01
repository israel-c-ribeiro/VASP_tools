import shutil
import subprocess

import numpy as np
import pytest

from conftest import load

sr = load("SCF_Rescue_Hybrid/scf_rescue.py")


def test_oscillating_scf_is_recognised(examples):
    root, _ = examples
    steps = sr.read_oszicar(root / "hybrid_scf_oscillating" / "OSZICAR")
    assert len(steps) == 1 and len(steps[0]["dE"]) == 60
    assert sr.classify_scf(steps[0]["dE"], ediff=1e-6, nelm=60).startswith("OSCILLATING")


def test_other_verdicts():
    k = np.arange(40)
    assert sr.classify_scf(10.0 * np.exp(-k), ediff=1e-5) == "converged"
    assert sr.classify_scf(10.0 * np.exp(-k / 10), ediff=1e-5).startswith("SLOW")
    growing = np.r_[10.0 * np.exp(-k[:20] / 2), 1e-3 * np.exp(k[:20] / 1.5)]
    assert sr.classify_scf(growing).startswith("DIVERGING")
    assert sr.classify_scf([1.0, np.nan]).startswith("DIVERGED")


def test_incar_tag_editing():
    txt = "ENCUT = 500\nALGO = Fast   ! comment\n"
    txt = sr.set_tag(txt, "ALGO", "Damped")
    txt = sr.set_tag(txt, "TIME", "0.4")
    assert sr.get_tag(txt, "algo") == "Damped" and sr.get_tag(txt, "TIME") == "0.4"
    assert sr.get_tag(sr.drop_tag(txt, "ENCUT"), "ENCUT") is None


def test_prepare_writes_both_steps(examples, tmp_path):
    root, _ = examples
    run = root / "hybrid_scf_oscillating"
    out = tmp_path / "rescue"
    assert sr.main(["prepare", str(run / "INCAR"), "--outcar", str(run / "OUTCAR"),
                    "--exe", "vasp_ncl", "-o", str(out)]) == 0
    s1, s2 = (out / "INCAR.step1").read_text(), (out / "INCAR.step2").read_text()
    assert sr.get_tag(s1, "LHFCALC") == ".FALSE." and sr.get_tag(s1, "BMIX") == "1.0"
    assert sr.get_tag(s1, "LORBIT") is None and sr.get_tag(s1, "LWAVE") == ".TRUE."
    assert sr.get_tag(s2, "LHFCALC") == ".TRUE." and sr.get_tag(s2, "ALGO") == "Damped"
    assert sr.get_tag(s1, "NBANDS") == sr.get_tag(s2, "NBANDS") == "96"
    assert sr.get_tag(s2, "ENCUT") == sr.get_tag(s1, "ENCUT")           # same basis in both steps
    job = (out / "job_two_step.sh").read_text()
    assert job.count("vasp_ncl") == 2
    if working_bash():
        assert subprocess.run(["bash", "-n", str(out / "job_two_step.sh")]).returncode == 0


def working_bash():
    """True if a usable bash is on PATH (on Windows 'bash' may be an uninstalled WSL stub)."""
    if not shutil.which("bash"):
        return False
    try:
        return subprocess.run(["bash", "-c", "exit 0"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False
