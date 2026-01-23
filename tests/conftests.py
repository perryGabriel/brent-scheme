import pytest

from brentscheme.BrentScheme import BrentScheme
from brentscheme.SchemaFactory import SchemaFactory
from brentscheme.SchemeDisplay import SchemeDisplay


@pytest.fixture(scope="session")
def factory():
    return SchemaFactory()


@pytest.fixture
def printer():
    return SchemeDisplay()


@pytest.fixture
def scheme():
    # fresh scheme per test to avoid state leakage
    return BrentScheme(n=3, d=3, m=3, p=27, verbose=0)
