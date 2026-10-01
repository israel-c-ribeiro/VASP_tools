"""Shared fixtures: every tool is a stand-alone script, so it is imported from its file path,
and every test runs on freshly generated synthetic VASP outputs (examples/synthetic_vasp.py)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import synthetic_vasp  # noqa: E402


def load(relpath: str):
    """Import a tool script (e.g. 'Band_Alignment_Vacuum/band_alignment.py') as a module."""
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def examples(tmp_path_factory):
    """All example folders, generated once per test session in a temporary directory."""
    out = tmp_path_factory.mktemp("examples")
    truth = {
        "synthetic_slab": synthetic_vasp.make_toy_slab_run(out / "synthetic_slab"),
        "synthetic_slab_typeII": synthetic_vasp.make_toy_slab_run(out / "synthetic_slab_typeII",
                                                                   homo_B=-3.60, efermi=-3.50),
    }
    synthetic_vasp.make_diagnostic_examples(out)
    (out / "layered_bulk").mkdir()
    synthetic_vasp.write_poscar(out / "layered_bulk" / "POSCAR", *synthetic_vasp.toy_slab(c=13.5, layer_z=4.3))
    return out, truth
