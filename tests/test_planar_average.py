import numpy as np
import pytest

from conftest import load

pa = load("LOCPOT_Planar_Average/planar_average.py")


def test_vacuum_level_and_work_function(examples, tmp_path, capsys):
    root, truth = examples
    t = truth["synthetic_slab"]
    grid, cell, symbols, cart = pa.read_locpot(root / "synthetic_slab" / "LOCPOT")
    assert grid.shape == (8, 8, 160) and len(symbols) == t["natoms"]
    prof = pa.planar_average(grid)
    vac = pa.vacuum_plateaus(prof, cell, cart)
    assert vac["V_upper"] == pytest.approx(t["v_vac"], abs=0.01)
    assert vac["V_lower"] == pytest.approx(t["v_vac"], abs=0.01)
    csv = tmp_path / "profile.csv"
    assert pa.main([str(root / "synthetic_slab" / "LOCPOT"), "--macro", "6.4", "--csv", str(csv)]) == 0
    assert "work function" in capsys.readouterr().out
    data = np.loadtxt(csv, delimiter=",", skiprows=1)
    assert data.shape == (160, 3)


def test_macroscopic_average_of_a_periodic_signal():
    n, h = 200, 20.0
    z = np.arange(n) / n * h
    prof = 1.0 + np.sin(2 * np.pi * z / 5.0)           # period 5 Å
    macro = pa.macroscopic_average(prof, h, 5.0)
    assert np.allclose(macro, 1.0, atol=1e-2)


def test_planar_average_is_the_in_plane_mean():
    g = np.random.default_rng(0).normal(size=(4, 5, 6))
    assert np.allclose(pa.planar_average(g, axis=2), g.mean(axis=(0, 1)))
    assert np.allclose(pa.planar_average(g, axis=0), g.mean(axis=(1, 2)))
