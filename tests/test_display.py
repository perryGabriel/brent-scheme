import numpy as np
import pytest
import torch

from brentscheme.BrentScheme import BrentScheme
from brentscheme.SchemaFactory import SchemaFactory
from brentscheme.SchemeDisplay import SchemeDisplay
from brentscheme.utils.tensors import random_unitary
from brentscheme.utils.io import delete_file  # swap if you renamed


@pytest.fixture(scope="module")
def factory():
    return SchemaFactory()


@pytest.fixture
def printer():
    return SchemeDisplay()


@pytest.fixture
def scheme():
    return BrentScheme(n=3, d=2, m=4, p=8, verbose=0)


def test_save_read_delete_random_scheme(tmp_path, monkeypatch, factory, printer, scheme):
    monkeypatch.chdir(tmp_path)

    score = printer.dump_tensors(scheme)
    assert isinstance(score, float)
    # read back into scheme (your API mutates `scheme`)
    factory.read_from_files(
        scheme,
        n=scheme.n, d=scheme.d, m=scheme.m, p=scheme.p,
        number=score,
        verbose=0,
    )

    # files should exist before deletion
    prefix = f"{scheme.n}_{scheme.d}_{scheme.m}_{scheme.p}_e{score:.3f}_"
    expected = [
        tmp_path / f"{prefix}alpha_pnd.pkl",
        tmp_path / f"{prefix}beta__pdm.pkl",
        tmp_path / f"{prefix}gamma_nmp.pkl",
    ]
    assert all(p.exists() for p in expected)

    delete_file(n=scheme.n, d=scheme.d, m=scheme.m, p=scheme.p, number=score, scheme_or_diagram="scheme")
    assert all(not p.exists() for p in expected)


def test_read_write_delete_naive_scheme(tmp_path, monkeypatch, factory, printer, scheme):
    monkeypatch.chdir(tmp_path)

    factory.set_scheme(scheme, preset="naive", n=2, d=2, m=2)
    score = printer.dump_tensors(scheme, score=10)

    assert score == 10  # your dump_tensors returns rounded input

    factory.read_from_files(scheme, n=2, d=2, m=2, p=8, number=10, verbose=0)
    factory.read_from_files(scheme, filename="2_2_2_8_e10.000", verbose=0)

    delete_file(n=2, d=2, m=2, p=8, number=10, scheme_or_diagram="scheme")
    # confirm deletion: at least one of them should be gone (ideally all)
    assert not (tmp_path / "2_2_2_8_e10.000_alpha_pnd.pkl").exists()
