import torch

from brentscheme import BrentScheme, SchemeDisplay, Trainer


def test_training_reduces_random_scheme_error_quickly():
    torch.manual_seed(0)
    scheme = BrentScheme(n=2, d=2, m=2, p=7, preset="random")
    display = SchemeDisplay()

    before = display.metrics(scheme).log10_L1
    Trainer().train(scheme, epochs=80, batch_size=1, lr=1e-2, verbose=0)
    after = display.metrics(scheme).log10_L1

    # More negative means smaller error.
    assert after < before - 0.8, (before, after)
