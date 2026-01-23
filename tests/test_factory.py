import pytest
import numpy as np

from brentscheme.BrentScheme import BrentScheme
from brentscheme.SchemaFactory import SchemaFactory
from brentscheme.SchemeDisplay import SchemeDisplay
from brentscheme.utils.tensors import random_unitary
from brentscheme.utils.io import delete_file  # swap if you renamed


@pytest.mark.parametrize("preset", ["random", "complex"])
@pytest.mark.parametrize("n", range(1, 9))
def test_random_and_complex_schemes_square_have_finite_error(preset, n):
    # only testing n=d=m (like your if n==d==m prints)
    scheme = BrentScheme()
    SchemaFactory().set_scheme(scheme, preset=preset, n=n, d=n, m=n)
    val = SchemeDisplay().error(scheme)
    # should return a float (possibly -inf)
    assert isinstance(val, float), f"error returned non-float {val} for preset={preset}, n={n}. Type was {type(val)}"


@pytest.mark.parametrize("n", range(1, 7))
@pytest.mark.parametrize("d", range(1, 7))
@pytest.mark.parametrize("m", range(1, 7))
def test_naive_schemes_are_exact(n, d, m):
    scheme = BrentScheme()
    SchemaFactory().set_scheme(scheme, preset="naive", n=n, d=d, m=m)
    val = SchemeDisplay().error(scheme)
    # your script treats anything != -inf as failure
    assert val == -np.inf, f"naive failed for n,d,m={n},{d},{m} with error {val}"


@pytest.mark.parametrize("n", range(1, 5))
@pytest.mark.parametrize("d", range(1, 5))
@pytest.mark.parametrize("m", range(1, 5))
@pytest.mark.parametrize("level", range(0, 3))
def test_fourier_schemes_accuracy(n, d, m, level):
    scheme = BrentScheme()
    SchemaFactory().set_scheme(scheme, fourier=level, n=n, d=d, m=m)
    val = SchemeDisplay().error(scheme)
    # your script: fail if > -14
    assert val <= -14, f"fourier failed for n,d,m,level={n},{d},{m},{level} with error {val}"

@pytest.mark.parametrize(
    "preset, threshold",
    [
        ("strassen", -np.inf),  # exact
        ("winograd", -14),
        ("laderman", -14),
    ],
)
def test_named_presets(preset, threshold):
    scheme = BrentScheme()
    SchemaFactory().set_scheme(scheme, preset=preset)
    val = SchemeDisplay().error(scheme)
    if threshold == -np.inf:
        assert val == -np.inf
    else:
        assert val <= threshold

@pytest.mark.parametrize("inner_preset", ["strassen", "laderman"])
def test_scheme_composition(inner_preset):
    outer = BrentScheme()
    SchemaFactory().set_scheme(outer, "strassen", n=2)

    inner = BrentScheme()
    SchemaFactory().set_scheme(inner, inner_preset, n=2)

    result = SchemaFactory().compose_schemes(outer, inner)

    assert result.n == outer.n * inner.n
    assert result.p == outer.p * inner.p
    assert SchemeDisplay().error(result) <= -13