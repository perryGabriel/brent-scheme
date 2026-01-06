import torch
import numpy as np

#@title Misc Functions
def permutation_matrix(indices):
  n = len(indices)
  matrix = torch.zeros((n, n), dtype=float)
  for i, idx in enumerate(indices):
      matrix[i, idx] = 1.0
  return matrix

def random_unitary(n):
  from scipy.linalg import qr
  H = np.random.randn(n, n)
  Q, R = qr(H)
  return torch.from_numpy(Q).type(torch.float64)

def rand_square(n):
  return torch.from_numpy(np.random.randn(n,n)).type(torch.float64)

def random_right_invertible(l, s=None, r=None):
  if r is None or r < l:
    r = l
  if s is None:
    s = torch.ones((l,))
  _S_ = torch.from_numpy(np.pad(np.diag(s), ((0, 0), (0,r-l)), 'constant', constant_values=((0, 0), (0,0)))).type(torch.float64)
  return random_unitary(l) @ _S_ @ random_unitary(r)

def delete_file(n, d, m, p, number, scheme_or_diagram):
  import os
  if scheme_or_diagram == 'scheme':
    os.remove(f"{n}_{d}_{m}_{p}_e{number:.3f}_alpha_pnd.pkl")
    os.remove(f"{n}_{d}_{m}_{p}_e{number:.3f}_beta__pdm.pkl")
    os.remove(f"{n}_{d}_{m}_{p}_e{number:.3f}_gamma_nmp.pkl")
  if scheme_or_diagram == 'diagram':
    os.remove(f'{n}_{d}_{m}_scheme_{p}_prod_{number:.3f}_best.png')