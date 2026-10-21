import csv
import itertools

import numpy as np
import pytest

from conftest import load

hb = load("Hydrogen_Bonds/hbonds.py")
import synthetic_hbonds  # noqa: E402  (examples/ is on sys.path through conftest)


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    folder = tmp_path_factory.mktemp("hbond_toy")
    return folder, synthetic_hbonds.make_hbond_example(folder)


def run(path, **kw):
    kw.setdefault("bridging_metal", ("Pb",))
    return hb.analyse(hb.read_structure(path), **kw)


@pytest.mark.parametrize("name", ["POSCAR", "POSCAR_triclinic"])
def test_known_counts(toy, name):
    folder, t = toy
    res = run(folder / name)
    s = res["summary"]
    assert s["n_groups"] == t["n_groups"] and s["n_donor_H"] == t["n_donor_H"] and s["n_other_H"] == t["n_other_H"]
    for c in t["cutoffs"]:
        assert s[f"n_HA_{hb.ck(c)}"] == t["n_HA"][c]
        assert s[f"n_hb_{hb.ck(c)}"] == t["n_hb"][c]
        for tag in ("free", "terminal", "bridging"):
            assert s[f"n_hb_{hb.ck(c)}_{tag}"] == t["n_hb_tag"][tag][c]
    assert s["n_rejected_angle"] == t["n_rejected_angle"] == 1
    assert s["soft"] == pytest.approx(t["soft"], abs=1e-6)
    assert s["d_HA_min"] == pytest.approx(2.30, abs=1e-6)


@pytest.mark.parametrize("name", ["POSCAR", "POSCAR_triclinic"])
def test_every_contact_distance_angle_and_tag(toy, name):
    folder, t = toy
    got = sorted((round(c["d_HA"], 4), round(c["angle"], 3), round(c["d_DA"], 4), c["tag"], c["acceptor_element"])
                 for c in run(folder / name)["contacts"])
    want = sorted((round(c["d_HA"], 4), round(c["angle"], 3), round(c["d_DA"], 4), c["tag"], c["acceptor"])
                  for c in t["contacts"])
    assert got == want


def test_per_group_aggregation(toy):
    folder, t = toy
    groups = run(folder / "POSCAR")["groups"]
    assert [g["label"] for g in groups] == ["NH3", "NH3", "OH2", "OH"]
    for g in groups:
        k = g["group"]
        assert g["n_hb_3.5"] == t["n_hb_group"][k]
        assert g["d_HA_min"] == pytest.approx(t["d_HA_min_group"][k], abs=1e-6)
        assert g["angle_at_min"] == pytest.approx(t["angle_at_min_group"][k], abs=1e-3)
    assert np.isnan(groups[3]["d_HA_min_hb"])                       # only contact fails the angle


def test_contact_across_the_boundary(toy):
    """Group 2's 3.10 Å contact only exists between atoms on opposite faces of the cell."""
    folder, t = toy
    st = hb.read_structure(folder / "POSCAR")
    res = hb.analyse(st)
    c = next(x for x in res["contacts"] if abs(x["d_HA"] - 3.10) < 1e-6)
    assert c["group"] == t["boundary_contact"]["group"]
    direct = np.linalg.norm(st.cart[c["acceptor_index"] - 1] - st.cart[c["H_index"] - 1])
    assert direct > 10.0                                            # without periodic images: no contact


def test_angle_filter_and_da_filter(toy):
    folder, _ = toy
    s = run(folder / "POSCAR", angle_min=90.0)["summary"]
    assert s["n_hb_3.0"] == 4 and s["n_rejected_angle"] == 0       # the 95° contact now counts
    s = run(folder / "POSCAR", angle_min=169.9)["summary"]
    assert s["n_hb_3.5"] == 2                                      # 170° and 175° only
    s = run(folder / "POSCAR", da_max=3.5)["summary"]
    assert s["n_hb_3.5"] == 1                                      # only the 2.30 Å O-H...Cl has D...A < 3.5


def test_c_h_is_not_a_donor(toy):
    """The iodide 2.60 Å in front of a methyl C-H is a contact only if C is declared a donor."""
    folder, t = toy
    assert all(abs(c["d_HA"] - t["trap_CH_distance"]) > 1e-3 for c in run(folder / "POSCAR")["contacts"])
    res = run(folder / "POSCAR", donors=("N", "O", "C"))
    assert any(abs(c["d_HA"] - t["trap_CH_distance"]) < 1e-6 for c in res["contacts"])


def test_min_h_selects_groups(toy):
    folder, _ = toy
    s = run(folder / "POSCAR", min_h=3)["summary"]
    assert s["n_groups"] == 2 and s["n_hb_3.5"] == 5               # the two NH3 groups only


def brute(cart_a, cart_b, cell, rcut, n=4):
    shifts = np.array(list(itertools.product(range(-n, n + 1), repeat=3))) @ cell
    out = []
    for i, a in enumerate(cart_a):
        for j, b in enumerate(cart_b):
            d = np.linalg.norm(b + shifts - a, axis=1)
            out += [(i, j, round(x, 6)) for x in d[d <= rcut]]
    return sorted(out)


def test_pairs_within_small_skewed_cell():
    """In a small, strongly skewed cell the cutoff exceeds half an interplanar spacing: the
    minimum image (rounded fractional difference) misses contacts, and one pair can touch
    through two images. pairs_within must return every image."""
    rng = np.random.default_rng(42)
    cell = np.array([[6.3, 0.0, 0.0], [5.2, 3.1, 0.0], [1.0, 1.5, 7.0]])
    assert hb.interplanar_spacings(cell).min() < 2 * 3.5
    a = rng.random((6, 3)) @ cell
    b = rng.random((9, 3)) @ cell
    i, j, V, D = hb.pairs_within(a, b, cell, 3.5)
    got = sorted((int(x), int(y), round(float(d), 6)) for x, y, d in zip(i, j, D))
    assert got == brute(a, b, cell, 3.5)
    assert np.allclose(np.linalg.norm(V, axis=1), D)
    # the naive minimum image finds fewer contacts
    df = (b[None] - a[:, None]) @ np.linalg.inv(cell)
    naive = np.linalg.norm((df - np.round(df)) @ cell, axis=2)
    assert (naive <= 3.5).sum() < len(got)


def test_triclinic_is_really_skewed(toy):
    folder, _ = toy
    st = hb.read_structure(folder / "POSCAR_triclinic")
    c = st.cell
    assert hb.angle_deg(c[0], c[1]) < 70 and hb.angle_deg(c[1], c[2]) < 85


def test_readers_cartesian_negative_scale_and_xdatcar(toy, tmp_path):
    folder, t = toy
    st = hb.read_structure(folder / "POSCAR")
    names = list(dict.fromkeys(st.symbols))
    counts = [st.symbols.count(n) for n in names]
    vol = abs(np.linalg.det(st.cell))
    head = ["cartesian copy", f"  {-vol:.10f}"] + ["  " + " ".join(f"{x:.10f}" for x in v) for v in st.cell]
    head += ["  " + " ".join(names), "  " + " ".join(map(str, counts))]
    (tmp_path / "POSCAR_cart").write_text("\n".join(head + ["Selective dynamics", "Cartesian"] + [
        " ".join(f"{x:.10f}" for x in p) + " T T T" for p in st.cart]) + "\n")
    frame = lambda k, fr: [f"Direct configuration=     {k}"] + [" ".join(f"{x:.10f}" for x in p) for p in fr]
    (tmp_path / "XDATCAR").write_text("\n".join(head[:1] + ["  1.0"] + head[2:]
                                                + frame(1, (st.frac + 0.01) % 1) + frame(2, st.frac)) + "\n")
    ref = run(folder / "POSCAR")["summary"]
    for f in ("POSCAR_cart", "XDATCAR"):
        s = run(tmp_path / f)["summary"]
        assert s["n_hb_3.5"] == ref["n_hb_3.5"] and s["soft"] == pytest.approx(ref["soft"], abs=1e-6)


def test_cli_batch_and_csv(toy, tmp_path, capsys):
    folder, t = toy
    out = tmp_path / "out"
    out.mkdir()
    rc = hb.main([str(folder / "POSCAR*"), "--bridging-metal", "Pb", "--csv", str(out / "contacts.csv"),
                  "--groups-csv", str(out / "groups.csv"), "--summary-csv", str(out / "summary.csv")])
    assert rc == 0
    text = capsys.readouterr().out
    assert "POSCAR_triclinic" in text
    rows = list(csv.DictReader(open(out / "summary.csv", encoding="utf-8")))
    assert len(rows) == 2 and all(int(r["n_hb_3.5"]) == t["n_hb"][3.5] for r in rows)
    contacts = list(csv.DictReader(open(out / "contacts.csv", encoding="utf-8")))
    assert len(contacts) == 2 * len(t["contacts"])
    assert sum(r["hbond"] == "False" for r in contacts) == 2
    groups = list(csv.DictReader(open(out / "groups.csv", encoding="utf-8")))
    assert len(groups) == 2 * t["n_groups"]


def test_cli_single_file_report_and_folder(toy, tmp_path, capsys):
    folder, _ = toy
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "CONTCAR").write_text((folder / "POSCAR").read_text())
    assert hb.main([str(run_dir), "--plot", str(tmp_path / "hb.png")]) == 0
    text = capsys.readouterr().out
    assert "4 donor group(s), 9 donor H" in text and "rejected by the angle: 1" in text
    assert (tmp_path / "hb.png").stat().st_size > 1000


def test_cli_fails_loudly(tmp_path, capsys):
    assert hb.main([str(tmp_path / "nothing*")]) == 1
    bad = tmp_path / "POSCAR"
    bad.write_text("vasp4\n1.0\n5 0 0\n0 5 0\n0 0 5\n1 1\nDirect\n0 0 0\n0.5 0.5 0.5\n")
    assert hb.main([str(bad)]) == 1
    assert "VASP 4" in capsys.readouterr().err
