import numpy as np
import pytest

from conftest import load

import synthetic_chgdiff as sc

cd = load("Charge_Density_Difference/chgdiff.py")


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    root = tmp_path_factory.mktemp("chgdiff")
    truth = sc.make_chgdiff_example(root)
    files = [str(root / s / "CHGCAR") for s in ("AB", "A", "B")]
    return root, files, truth


def test_known_charge_transfer_and_difference_file(example, tmp_path, capsys):
    root, files, t = example
    out, csv = tmp_path / "CHGDIFF.vasp", tmp_path / "edd.csv"
    assert cd.main(files + ["-o", str(out), "--csv", str(csv)]) == 0
    text = capsys.readouterr().out
    assert "compatible" in text and "A|B boundary" in text and "+0.2500" in text
    # the written file is a CHGCAR whose grid / V is exactly the drho put in
    grid, h = cd.read_chgcar(out)
    assert h["shape"] == t["shape"] and h["symbols"] == ["Pb", "I", "N", "H"]
    assert np.allclose(grid / 500.0, t["drho"], atol=1e-10)   # 12 significant digits in the file
    data = np.loadtxt(csv, delimiter=",", skiprows=1)
    assert data.shape == (80, 7)
    z, lam, dq = data[:, 0], data[:, 2], data[:, 3]
    assert np.interp(t["plane_z"], z, dq) == pytest.approx(t["q_moved"], abs=1e-6)
    assert lam.sum() * (z[1] - z[0]) == pytest.approx(0.0, abs=1e-9)


def test_profile_functions_recover_the_truth(example):
    root, files, t = example
    res = cd.difference(files)
    assert res["electrons"][:3] == pytest.approx([t["electrons_AB"], t["electrons_A"], t["electrons_B"]], abs=1e-6)
    assert res["electrons"][3] == pytest.approx(t["integral"], abs=1e-9)
    assert res["max_abs"] == pytest.approx(np.abs(t["drho"]).max(), rel=1e-8)
    prof = cd.profile(res)
    assert prof["origin"] == pytest.approx(18.5)                  # middle of the 12 -> 25 Å vacuum
    assert cd.charge_at(prof, t["plane_z"]) == pytest.approx(t["q_moved"], abs=1e-6)
    assert cd.charge_at(prof, 2.0) == pytest.approx(0.0, abs=1e-9)   # below everything
    mu, xc = cd.dipole(prof)
    assert mu == pytest.approx(t["dipole_eA"], abs=5e-3)
    assert cd.mirror_asymmetry(prof, xc) > 0.3
    labels, problems, dev = cd.match_atoms(res["header"], {"A": cd.read_chgcar(files[1])[1],
                                                           "B": cd.read_chgcar(files[2])[1]})
    assert labels == ["A", "A", "B", "B"] and not problems and dev < 1e-6
    planes = cd.fragment_boundaries(prof["z_atoms"], labels)
    assert len(planes) == 1 and planes[0][0] == pytest.approx(9.5)


def test_rereading_the_difference_file_gives_the_same_profile(example, tmp_path, capsys):
    root, files, t = example
    out = tmp_path / "CHGDIFF.vasp"
    cd.main(files + ["-o", str(out)])
    capsys.readouterr()
    assert cd.main([str(out), "--no-write", "--plane", "9.5", "--between", "2", "3"]) == 0
    text = capsys.readouterr().out
    assert text.count("+0.2500") == 2 and "between I2 and N3" in text


def test_incompatible_grid_is_rejected(example, capsys):
    root, files, t = example
    bad = files[:2] + [str(root / "B_other_grid" / "CHGCAR")]
    assert cd.main(bad) == 1
    err = capsys.readouterr().err
    assert "grid of B is 6 x 6 x 60" in err and "NGXF" in err


def test_moved_and_missing_atoms_are_reported(example, tmp_path):
    root, files, t = example
    sym, fr = sc._structure(sc.ATOMS_B)
    fr = fr.copy()
    fr[1, 2] += 0.1 / 20.0                                            # H displaced by 0.1 Å
    sc.write_chgcar(tmp_path / "CHGCAR", sym, sc.CELL, fr, np.ones(sc.SHAPE))
    ab = cd.read_chgcar(files[0])[1]
    a = cd.read_chgcar(files[1])[1]
    b = cd.read_chgcar(tmp_path / "CHGCAR")[1]
    problems = cd.check_compatible(ab, {"A": a, "B": b})
    assert any("B atom 2 (H)" in p and "0.100 Å" in p for p in problems)
    assert any("AB atom 4 (H) belongs to no fragment" in p for p in problems)
    assert any("AB has 4 atoms, the fragments together 3" in p for p in cd.check_compatible(ab, {"A": a, "B": {
        **b, "symbols": b["symbols"][:1], "frac": b["frac"][:1]}}))


def test_noncollinear_layout_streaming_and_small_chunks(tmp_path):
    rng = np.random.default_rng(1)
    grid = rng.random((5, 4, 7))                                       # nx*ny not a multiple of 5 values per line
    sym, fr = sc._structure(sc.ATOMS_A)
    sc.write_chgcar(tmp_path / "CHGCAR", sym, sc.CELL, fr, grid, magnetization_blocks=3)
    full, h = cd.read_chgcar(tmp_path / "CHGCAR")
    assert np.allclose(full, grid)
    with cd.ChgcarStream(tmp_path / "CHGCAR", chunk_bytes=37) as s:     # chunk boundaries inside numbers
        sl = np.stack(list(s.slices()), axis=-1).transpose(1, 0, 2)
    assert np.allclose(sl, grid)


def test_truncated_and_overflowed_files_fail_loudly(tmp_path):
    sym, fr = sc._structure(sc.ATOMS_A)
    sc.write_chgcar(tmp_path / "CHGCAR", sym, sc.CELL, fr, np.ones((4, 4, 6)), augmentation=False)
    lines = (tmp_path / "CHGCAR").read_text().splitlines()
    (tmp_path / "short").write_text("\n".join(lines[:-3]) + "\n")
    with pytest.raises(ValueError, match="truncated"):
        cd.read_chgcar(tmp_path / "short")
    lines[-2] = "*****************  " + lines[-2]
    (tmp_path / "over").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="overflowed"):
        cd.read_chgcar(tmp_path / "over")
    (tmp_path / "junk").write_text("not a chgcar\n")
    with pytest.raises(ValueError, match="not a CHGCAR"):
        cd.read_chgcar(tmp_path / "junk")


def test_cube_and_plot(example, tmp_path):
    root, files, t = example
    cube, png = tmp_path / "edd.cube", tmp_path / "edd.png"
    assert cd.main(files + ["--no-write", "--cube", str(cube), "--plot", str(png)]) == 0
    lines = cube.read_text().splitlines()
    assert lines[2].split()[0] == "4" and lines[3].split()[0] == "8"
    vals = np.array(" ".join(lines[2 + 4 + 4:]).split(), dtype=float)
    assert vals.size == 8 * 8 * 80
    assert np.allclose(vals.reshape(t["shape"]), t["drho"] * cd.BOHR ** 3, atol=1e-8)
    assert png.stat().st_size > 10_000


def test_profile_along_another_axis(example):
    root, files, t = example
    res = cd.difference(files)
    prof = cd.profile(res, axis=0, origin=0.0)
    lam_ref = t["drho"].sum(axis=(1, 2)) / t["drho"][0].size * 500.0 / 5.0     # plane mean * V / L
    assert np.allclose(prof["lam"], lam_ref, atol=1e-12)


def test_check_only_and_bad_arguments(example, capsys):
    root, files, t = example
    assert cd.main(files + ["--check-only"]) == 0
    assert cd.main(files[:2]) == 1
    assert cd.main(files + ["--no-write", "--between", "1", "9"]) == 1
    assert "atom indices run from 1 to 4" in capsys.readouterr().err
