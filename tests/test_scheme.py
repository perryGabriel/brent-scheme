# tests/test_scheme.py
from brentscheme.BrentScheme import BrentScheme
from brentscheme import SchemaFactory
import torch
import pytest


def test_forward_matches_triple_delta():
    for n,d,m in [(2,2,2), (3,3,3), (2,3,2), (4,4,2)]:
        scheme_small_exact = BrentScheme(n=n, d=d, m=m, preset="naive")
        out = scheme_small_exact.forward()
        target = scheme_small_exact.TRIPLE_DELTA_nmnddm
        assert torch.allclose(out, target, atol=0, rtol=0)

def test_clone_independent_storage():
    scheme_small_exact = BrentScheme(n=2, d=2, m=2, preset="naive")
    s2 = scheme_small_exact.clone()
    assert torch.allclose(s2.forward(), scheme_small_exact.forward())
    # mutate clone tensor and ensure original unchanged
    s2.alpha_pnd[0,0,0] += 1.0
    assert not torch.allclose(s2.forward(), scheme_small_exact.forward())

def test_measure_zero_error():
    scheme_small_exact = BrentScheme(n=2, d=2, m=2, preset="naive")
    error = scheme_small_exact.measure(scheme_small_exact.forward() - scheme_small_exact.TRIPLE_DELTA_nmnddm)
    assert isinstance(error, torch.Tensor)
    assert torch.isclose(error, torch.tensor(0.0, dtype=torch.float64), atol=0, rtol=0), f"Expected zero error, got {torch.exp(error)}, log-error {error}"

def test_measure_nonzero_error():
    scheme_small_exact = BrentScheme(n=2, d=2, m=2, preset="naive")
    error = scheme_small_exact.measure(scheme_small_exact.forward() - scheme_small_exact.TRIPLE_DELTA_nmnddm + 0.1)
    assert error > 0.0

@pytest.fixture
def gen():
    g = torch.Generator().manual_seed(0)
    return g

def test_naive_3x3_matches_matmul(gen):
    scheme = BrentScheme()
    SchemaFactory().set_scheme(scheme, "naive", n=3, d=3, m=3)

    for _ in range(10):
        A = torch.randn((3, 3), dtype=torch.float64, generator=gen)
        B = torch.randn((3, 3), dtype=torch.float64, generator=gen)
        err = scheme.measure(scheme(A, B) - (A @ B))
        assert torch.log10(err).item() <= -14



def test_scheme_device_cpu():
    scheme_small_exact = BrentScheme(n=2, d=2, m=2, preset="naive").to('cpu')
    assert scheme_small_exact.alpha_pnd.device.type == 'cpu'
    assert scheme_small_exact.beta__pdm.device.type == 'cpu'
    assert scheme_small_exact.gamma_nmp.device.type == 'cpu'

def test_scheme_device_cuda_if_available():
    scheme_small_exact = BrentScheme(n=2, d=2, m=2, preset="naive")
    if torch.cuda.is_available():
        scheme_cuda = scheme_small_exact.to('cuda')
        assert scheme_cuda.alpha_pnd.device.type == 'cuda'
        assert scheme_cuda.beta__pdm.device.type == 'cuda'
        assert scheme_cuda.gamma_nmp.device.type == 'cuda'
    else:
        print("CUDA not available; skipping CUDA device test.")
        assert True  # trivially pass if CUDA not available

def test_scheme_device_transfer():
    scheme_small_exact = BrentScheme(n=2, d=2, m=2, preset="naive")
    if torch.cuda.is_available():
        scheme_cuda = scheme_small_exact.to('cuda')
        scheme_cpu = scheme_cuda.to('cpu')
        assert scheme_cpu.alpha_pnd.device.type == 'cpu'
        assert scheme_cpu.beta__pdm.device.type == 'cpu'
        assert scheme_cpu.gamma_nmp.device.type == 'cpu'
    else:
        print("CUDA not available; skipping device transfer test.")
        assert True  # trivially pass if CUDA not available
        
def test_scheme_clone_device():
    scheme_small_exact = BrentScheme(n=2, d=2, m=2, preset="naive").to('cpu')
    scheme_clone = scheme_small_exact.clone()
    assert scheme_clone.alpha_pnd.device == scheme_small_exact.alpha_pnd.device
    assert scheme_clone.beta__pdm.device == scheme_small_exact.beta__pdm.device
    assert scheme_clone.gamma_nmp.device == scheme_small_exact.gamma_nmp.device

def test_scheme_to_device_chain():
    scheme_small_exact = BrentScheme(n=2, d=2, m=2, preset="naive")
    if torch.cuda.is_available():
        scheme_chain = scheme_small_exact.to('cuda').to('cpu').to('cuda')
        assert scheme_chain.alpha_pnd.device.type == 'cuda'
        assert scheme_chain.beta__pdm.device.type == 'cuda'
        assert scheme_chain.gamma_nmp.device.type == 'cuda'
    else:
        print("CUDA not available; skipping device chain test.")
        assert True  # trivially pass if CUDA not available

def test_scheme_measure_after_device_transfer():
    scheme_small_exact = BrentScheme(n=2, d=2, m=2, preset="naive")
    if torch.cuda.is_available():
        scheme_cuda = scheme_small_exact.to('cuda')
        error_cuda = scheme_cuda.measure(scheme_cuda.forward() - scheme_cuda.TRIPLE_DELTA_nmnddm)
        scheme_cpu = scheme_cuda.to('cpu')
        error_cpu = scheme_cpu.measure(scheme_cpu.forward() - scheme_cpu.TRIPLE_DELTA_nmnddm)
        assert torch.isclose(error_cuda, error_cpu, atol=1e-6, rtol=1e-6)
    else:
        print("CUDA not available; skipping measure after device transfer test.")
        assert True  # trivially pass if CUDA not available

def test_scheme_clone_after_device_transfer():
    scheme_small_exact = BrentScheme(n=2, d=2, m=2, preset="naive")
    if torch.cuda.is_available():
        scheme_cuda = scheme_small_exact.to('cuda')
        scheme_clone = scheme_cuda.clone()
        assert scheme_clone.alpha_pnd.device.type == 'cuda'
        assert scheme_clone.beta__pdm.device.type == 'cuda'
        assert scheme_clone.gamma_nmp.device.type == 'cuda'
    else:
        print("CUDA not available; skipping clone after device transfer test.")
        assert True  # trivially pass if CUDA not available

def test_scheme_forward_after_multiple_device_transfers():
    scheme_small_exact = BrentScheme(n=2, d=2, m=2, preset="naive")
    if torch.cuda.is_available():
        scheme_chain = scheme_small_exact.to('cuda').to('cpu').to('cuda')
        out = scheme_chain.forward()
        target = scheme_chain.TRIPLE_DELTA_nmnddm
        assert torch.allclose(out, target, atol=1e-6, rtol=1e-6)
    else:
        print("CUDA not available; skipping forward after multiple device transfers test.")
        assert True  # trivially pass if CUDA not available