from brentscheme.misc import permutation_matrix
from brentscheme.BrentScheme import BrentScheme
from brentscheme.SchemaFactory import SchemaFactory
from brentscheme.SchemeDisplay import SchemeDisplay
from brentscheme.SchemeManipulator import SchemeManipulator
from brentscheme.Stepper import Stepper
from brentscheme.Trainer import Trainer

import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import pandas as pd
import time

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

# @title A Display object for printing and saving data about a scheme, including accuracy tests
class SchemeDisplay(object):

  def __init__(self):
    pass

  def print(self, scheme, verbose=0):
    print(f"Network trained for ({scheme.n} x {scheme.d}) @ ({scheme.d} x {scheme.m}) using {scheme.p} out of {scheme.n*scheme.d*scheme.m} multiplications; complexity is n^{scheme.complexity():.3f}\n\nUsing the L{scheme.L_norm} norm over the field {scheme.field}\n")

    if verbose > 0:
      self.test(scheme, verbose=2)
      print("")

    if verbose > 1:
      alphas = [[f'{scheme.alpha_pnd[pi,aj,ak]: .3f}*A[{aj+1},{ak+1}]' if scheme.alpha_pnd[pi,aj,ak] != 0.0 else "" for aj in range(scheme.n) for ak in range(scheme.d)] for pi in range(scheme.p)]
      alpha_sums = ["".join([f'{alphas[pi][elem]} + ' if alphas[pi][elem] != "" else "" for elem in range(scheme.n*scheme.d)])[:-3] for pi in range(scheme.p)]

      betas = [[f'{scheme.beta__pdm[pi,bj,bk]: .3f}*B[{bj+1},{bk+1}]' if scheme.beta__pdm[pi,bj,bk] != 0.0 else "" for bj in range(scheme.d) for bk in range(scheme.m)] for pi in range(scheme.p)]
      beta_sums  = ["".join([f'{betas[pi][elem]} + ' if betas[pi][elem] != "" else "" for elem in range(scheme.d*scheme.m)])[:-3] for pi in range(scheme.p)]

      products = [f'({alpha_sums[pi]}) * ({beta_sums[pi]})' for pi in range(scheme.p)]
      print("Products P_i = (alpha_ind * A_nd) * (beta_idm * B_dm)")
      for pi in range(scheme.p): print(f"P_{pi+1} =", products[pi])
      print("")

      gammas = [[[f'{scheme.gamma_nmp[gi,gj,gk]: .3f}*P_{gk+1}' if scheme.gamma_nmp[gi,gj,gk] != 0.0 else "" for gk in range(scheme.p)] for gj in range(scheme.m)] for gi in range(scheme.n)]
      gamma_sums = [["".join([f'{gammas[c1][c2][pi]} + ' if gammas[c1][c2][pi] != "" else "" for pi in range(scheme.p)])[:-3] for c2 in range(scheme.m)]for c1 in range(scheme.n)]
      print("Outputs AB_nm = gamma_nmi * P_i")
      for c1 in range(scheme.n):
        for c2 in range(scheme.m):
          print(f"AB[{c1+1},{c2+1}] =", gamma_sums[c1][c2])
      print("")

  def print_triple_deltas(self, scheme, output=None):
    if output is None:
      output = scheme.forward().reshape((scheme.n*scheme.d*scheme.m, scheme.n*scheme.d*scheme.m))
    target = scheme.TRIPLE_DELTA_nmnddm.reshape((scheme.n*scheme.d*scheme.m, scheme.n*scheme.d*scheme.m))
    error = output - target

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(target, cmap='seismic', interpolation='nearest')
    plt.title("Exact")

    plt.subplot(1,3,2)
    plt.imshow(output, cmap='seismic', interpolation='nearest')
    plt.title("Approximation")

    plt.subplot(1,3,3)
    plt.imshow(error, cmap='seismic', interpolation='nearest')
    plt.title(f"Max Error: {torch.max(torch.abs(error)):.5f}")
    plt.show()

  ### TEST ###

  def test(self, scheme, _range=10, num=200, verbose=0):
    error = (scheme.forward() - scheme.TRIPLE_DELTA_nmnddm)
    err_size = error.numel()

    if verbose == 0:
      return torch.log10(scheme.measure(error)).item()

    mags = torch.abs(error)
    L1 = torch.sum(mags) / err_size
    L2 = (torch.sum(mags**2) / err_size)**0.5
    Linf = torch.max(mags)

    if verbose == 1:
      return torch.log10(L1).item(), torch.log10(L2).item(), torch.log10(Linf).item()

    print(f'Avg L1 error: 10^{torch.log10(L1):.4f}, Avg L2 error: 10^{torch.log10(L2):.4f}, max error: 10^{torch.log10(Linf):.4f}')

    if verbose > 2:
      self.print_triple_deltas(scheme)



  ### FILE SAVE FUNCTIONS ###

  def dump_to_file(self, scheme, number=None):
    if number is None:
      number = round(self.test(scheme, verbose=0),3)
    else:
      number = round(number,3)
    import pickle
    file1 = open(f'{scheme.n}_{scheme.d}_{scheme.m}_{scheme.p}_e{number:.3f}_alpha_pnd.pkl', 'wb')
    pickle.dump(scheme.alpha_pnd, file1)
    file1.close()
    file1 = open(f'{scheme.n}_{scheme.d}_{scheme.m}_{scheme.p}_e{number:.3f}_beta__pdm.pkl', 'wb')
    pickle.dump(scheme.beta__pdm, file1)
    file1.close()
    file1 = open(f'{scheme.n}_{scheme.d}_{scheme.m}_{scheme.p}_e{number:.3f}_gamma_nmp.pkl', 'wb')
    pickle.dump(scheme.gamma_nmp, file1)
    file1.close()

    return number

printer = SchemeDisplay()