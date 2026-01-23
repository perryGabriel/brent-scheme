# tests/test_imports.py
import importlib

MODULES = [
    "brentscheme.utils.tensors",
    "brentscheme.utils.io",
    "brentscheme.BrentScheme",
    "brentscheme.SchemeDisplay",
    "brentscheme.SchemaFactory",
    "brentscheme.SchemeManipulator",
    "brentscheme.Stepper",
    "brentscheme.Trainer",
]

def test_all_modules_import():
    for m in MODULES:
        importlib.import_module(m)
