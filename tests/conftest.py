import sys
from pathlib import Path

import pytest

# Ensure local src/ layout is importable without editable install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
    return BrentScheme(n=3, d=3, m=3, p=27, verbose=0)
