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

#@title A Factory for setting preset schema
class SchemaFactory(object):
  from brentscheme.SchemeManipulator import SchemeManipulator

  def __init__(self):
      pass

  def set_TRIPLE_DELTA(self, scheme):
    scheme.TRIPLE_DELTA_nmnddm = torch.einsum("ac,Ab,BC->cCaAbB", torch.eye(scheme.n), torch.eye(scheme.d), torch.eye(scheme.m)).type(torch.float64)

  def set_scheme(self, scheme, preset='random', **kwargs):
      if 'n' in kwargs: scheme.n = kwargs['n']
      elif scheme.n is None: scheme.n = 2
      if 'd' in kwargs: scheme.d = kwargs['d']
      elif scheme.d is None: scheme.d = scheme.n
      if 'm' in kwargs: scheme.m = kwargs['m']
      elif scheme.m is None: scheme.m = scheme.n
      if 'p' in kwargs: scheme.p = kwargs['p']
      elif scheme.p is None: scheme.p = scheme.n*scheme.d*scheme.m

      if 'fourier' in kwargs: preset = 'fourier'

      match preset:
        case 'random':   self.set_random(scheme)
        case 'complex':  self.set_random(scheme, norm=1, field='C')
        case 'naive':    self.set_naive(scheme)
        case 'fourier':
          self.set_naive(scheme, norm=1, field='C') # fourier level 0 makes a complex naive scheme
          self.Fourier(scheme, level=kwargs['fourier'])
        case 'strassen': self.set_Strassen(scheme)
        case 'winograd': self.set_Winograd(scheme)
        case 'laderman': self.set_Laderman(scheme)

  def set_random(self, scheme, Re_std_dev=1, Im_std_dev=1, norm=2, field='R'):
    scheme.alpha_pnd = torch.normal(0, Re_std_dev, size=(scheme.p, scheme.n, scheme.d)).type(torch.float64)
    scheme.beta__pdm = torch.normal(0, Re_std_dev, size=(scheme.p, scheme.d, scheme.m)).type(torch.float64)
    scheme.gamma_nmp = torch.normal(0, Re_std_dev, size=(scheme.n, scheme.m, scheme.p)).type(torch.float64)
    if field == 'C':
      scheme.alpha_pnd = scheme.alpha_pnd.type(torch.complex128) + torch.normal(0, Im_std_dev, size=(scheme.p, scheme.n, scheme.d)).type(torch.complex128)*1j
      scheme.beta__pdm = scheme.beta__pdm.type(torch.complex128) + torch.normal(0, Im_std_dev, size=(scheme.p, scheme.d, scheme.m)).type(torch.complex128)*1j
      scheme.gamma_nmp = scheme.gamma_nmp.type(torch.complex128) + torch.normal(0, Im_std_dev, size=(scheme.n, scheme.m, scheme.p)).type(torch.complex128)*1j
    from brentscheme.SchemeManipulator import SchemeManipulator
    SchemeManipulator().set_norm(scheme, norm=norm, field=field)
    self.set_TRIPLE_DELTA(scheme)

  def set_naive(self, scheme, norm=1, field='R'):
    scheme.p = scheme.n*scheme.d*scheme.m
    L = L_inv = torch.eye(scheme.n).type(torch.float64)
    M = M_inv = torch.eye(scheme.d).type(torch.float64)
    R = R_inv = torch.eye(scheme.m).type(torch.float64)
    b,c,a = torch.ones(scheme.n), torch.ones(scheme.d), torch.ones(scheme.m)
    scheme.alpha_pnd = torch.einsum("ia,i,   j,Ak,k->ijkaA", L_inv, b,        a, M,     c).reshape((scheme.p,scheme.n,scheme.d)).type(torch.float64)
    scheme.beta__pdm = torch.einsum(   "i,jB,j,kb,k->ijkbB",        b, R_inv, a, M_inv, c).reshape((scheme.p,scheme.d,scheme.m)).type(torch.float64)
    scheme.gamma_nmp = torch.einsum("ci,i,Cj,j,   k->cCijk", L,     b, R,     a,        c).reshape((scheme.n,scheme.m,scheme.p)).type(torch.float64)
    self.set_TRIPLE_DELTA(scheme)

  def Fourier(self, scheme, level=2, norm=1):
    if level <= 0: return # no Fourier
    import math

    scheme.alpha_pnd = scheme.alpha_pnd.type(torch.complex128)
    scheme.beta__pdm = scheme.beta__pdm.type(torch.complex128)
    scheme.gamma_nmp = scheme.gamma_nmp.type(torch.complex128)

    theta_n = 2 * 3.141592653589793238462643383279502884197j / scheme.n
    theta_d = 2 * 3.141592653589793238462643383279502884197j / scheme.d
    theta_m = 2 * 3.141592653589793238462643383279502884197j / scheme.m

    fourier_n = torch.pow(torch.full((scheme.n,), theta_n), torch.arange(scheme.n)).type(torch.complex128)
    fourier_d = torch.pow(torch.full((scheme.d,), theta_d), torch.arange(scheme.d)).type(torch.complex128)
    fourier_m = torch.pow(torch.full((scheme.m,), theta_m), torch.arange(scheme.m)).type(torch.complex128)

    vander_n = torch.vander(fourier_n, increasing=True).type(torch.complex128)
    vander_d = torch.vander(fourier_d, increasing=True).type(torch.complex128)
    vander_m = torch.vander(fourier_m, increasing=True).type(torch.complex128)

    # apply
    from brentscheme.SchemeManipulator import SchemeManipulator
    if level == 1: SchemeManipulator().change_basis(scheme, M=vander_d)
    elif level == 2: SchemeManipulator().change_basis(scheme, L=vander_n, R=vander_m, M=vander_d)
    # FIXME: apply a fourier transform along the product axis as well
    elif level >= 3: SchemeManipulator().change_basis(scheme, L=vander_n, R=vander_m, M=vander_d)
    self.set_TRIPLE_DELTA(scheme)

  def set_Strassen(self, scheme):
    scheme.n = scheme.d = scheme.m = 2
    scheme.p = 7
    scheme.alpha_pnd = torch.Tensor([[[ 1,0],[0, 1]],[[ 0,0],[1, 1]],[[ 1,0],[0, 0]],[[ 0,0],[0, 1]],[[ 1,1],[0, 0]],[[-1,0],[1, 0]],[[ 0,1],[0,-1]]]).type(torch.float64)
    scheme.beta__pdm  = torch.Tensor([[[ 1,0],[0, 1]],[[ 1,0],[0, 0]],[[ 0,1],[0,-1]],[[-1,0],[1, 0]],[[ 0,0],[0, 1]],[[ 1,1],[0, 0]],[[ 0,0],[1, 1]]]).type(torch.float64)
    scheme.gamma_nmp = torch.Tensor([[[1, 0,0,1,-1,0,1], [0, 0,1,0, 1,0,0]],[[0, 1,0,1, 0,0,0], [1,-1,1,0, 0,1,0]]]).type(torch.float64)
    from brentscheme.SchemeManipulator import SchemeManipulator
    SchemeManipulator().set_norm(scheme, norm=1, field='R')
    self.set_TRIPLE_DELTA(scheme)

  def set_Winograd(self, scheme):
    scheme.n = scheme.d = scheme.m = 2
    scheme.p = 7
    scheme.alpha_pnd = torch.Tensor([[[ 1,0],[ 0, 0]],[[ 0,1],[ 0, 0]],[[ 1,1],[-1,-1]],[[ 0,0],[ 0, 1]],[[-1,0],[ 1, 0]],[[ 0,0],[ 1, 1]],[[-1,0],[ 1, 1]]]).type(torch.float64)
    scheme.beta__pdm  = torch.Tensor([[[ 1, 0],[0, 0]],[[ 0, 0],[1, 0]],[[ 0, 0],[0, 1]],[[-1, 1],[1,-1]],[[ 0, 1],[0,-1]],[[-1, 1],[0, 0]],[[ 1,-1],[0, 1]]]).type(torch.float64)
    scheme.gamma_nmp = torch.Tensor([[[1,1,0,0,0,0,0], [1,0,1,0,0,1,1]],[[1,0,0,1,1,0,1], [1,0,0,0,1,1,1]]]).type(torch.float64)
    from brentscheme.SchemeManipulator import SchemeManipulator
    SchemeManipulator().set_norm(scheme, norm=1, field='R')
    self.set_TRIPLE_DELTA(scheme)

  def set_Laderman(self, scheme):
    scheme.n = scheme.d = scheme.m = 3
    scheme.p = 23
    scheme.alpha_pnd = torch.Tensor([[[ 1,1, 1],[-1,-1, 0],[ 0,-1,-1]],[[ 1,0, 0],[-1, 0, 0],[ 0, 0, 0]],[[ 0,0, 0],[ 0, 1, 0],[ 0, 0, 0]],[[-1,0, 0],[ 1, 1, 0],[ 0, 0, 0]],[[ 0,0, 0],[ 1, 1, 0],[ 0, 0, 0]], # 5
                               [[ 1,0, 0],[ 0, 0, 0],[ 0, 0, 0]],[[-1,0, 0],[ 0, 0, 0],[ 1, 1, 0]],[[-1,0, 0],[ 0, 0, 0],[ 1, 0, 0]],[[ 0,0, 0],[ 0, 0, 0],[ 1, 1, 0]],[[ 1,1, 1],[ 0,-1,-1],[-1,-1, 0]], # 10
                               [[ 0,0, 0],[ 0, 0, 0],[ 0, 1, 0]],[[ 0,0,-1],[ 0, 0, 0],[ 0, 1, 1]],[[ 0,0, 1],[ 0, 0, 0],[ 0, 0,-1]],[[ 0,0, 1],[ 0, 0, 0],[ 0, 0, 0]],[[ 0,0, 0],[ 0, 0, 0],[ 0, 1, 1]], # 15
                               [[ 0,0,-1],[ 0, 1, 1],[ 0, 0, 0]],[[ 0,0, 1],[ 0, 0,-1],[ 0, 0, 0]],[[ 0,0, 0],[ 0, 1, 1],[ 0, 0, 0]],[[ 0,1, 0],[ 0, 0, 0],[ 0, 0, 0]],[[ 0,0, 0],[ 0, 0, 1],[ 0, 0, 0]], # 20
                               [[ 0,0, 0],[ 1, 0, 0],[ 0, 0, 0]],[[ 0,0, 0],[ 0, 0, 0],[ 1, 0, 0]],[[ 0,0, 0],[ 0, 0, 0],[ 0, 0, 1]]]).type(torch.float64)
    scheme.beta__pdm  = torch.Tensor([[[ 0, 0, 0],[0, 1, 0],[ 0, 0, 0]],[[ 0,-1, 0],[0, 1, 0],[ 0, 0, 0]],[[-1, 1, 0],[1,-1,-1],[-1, 0, 1]],[[ 1,-1, 0],[0, 1, 0],[ 0, 0, 0]],[[-1, 1, 0],[0, 0, 0],[ 0, 0, 0]], # 5
                               [[ 1, 0, 0],[0, 0, 0],[ 0, 0, 0]],[[ 1, 0,-1],[0, 0, 1],[ 0, 0, 0]],[[ 0, 0, 1],[0, 0,-1],[ 0, 0, 0]],[[-1, 0, 1],[0, 0, 0],[ 0, 0, 0]],[[ 0, 0, 0],[0, 0, 1],[ 0, 0, 0]], # 10
                               [[-1, 0, 1],[1,-1,-1],[-1, 1, 0]],[[ 0, 0, 0],[0, 1, 0],[ 1,-1, 0]],[[ 0, 0, 0],[0, 1, 0],[ 0,-1, 0]],[[ 0, 0, 0],[0, 0, 0],[ 1, 0, 0]],[[ 0, 0, 0],[0, 0, 0],[-1, 1, 0]], # 15
                               [[ 0, 0, 0],[0, 0, 1],[ 1, 0,-1]],[[ 0, 0, 0],[0, 0, 1],[ 0, 0,-1]],[[ 0, 0, 0],[0, 0, 0],[-1, 0, 1]],[[ 0, 0, 0],[1, 0, 0],[ 0, 0, 0]],[[ 0, 0, 0],[0, 0, 0],[ 0, 1, 0]], # 20
                               [[ 0, 0, 1],[0, 0, 0],[ 0, 0, 0]],[[ 0, 1, 0],[0, 0, 0],[ 0, 0, 0]],[[ 0, 0, 0],[0, 0, 0],[ 0, 0, 1]]]).type(torch.float64)
    scheme.gamma_nmp = torch.Tensor([[[0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0], [1,0,0,1,1,1,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0], [0,0,0,0,0,1,1,0,1,1,0,0,0,1,0,1,0,1,0,0,0,0,0]],
                               [[0,1,1,1,0,1,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0], [0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,1,0,0]],
                               [[0,0,0,0,0,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0], [0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1]]]).type(torch.float64)
    from brentscheme.SchemeManipulator import SchemeManipulator
    SchemeManipulator().set_norm(scheme, norm=1, field='R')#15        20                  5        10        15        20                  5        10        15        20
    self.set_TRIPLE_DELTA(scheme)

  def read_from_files(self, scheme, n=None, d=None, m=None, p=None, number=None, filename=None, verbose=0):
    if filename is None:
      if n is None and p is None and number is not None:
        filename = f'{float(number):.3f}'
      elif n is not None and d is not None and m is not None and p is not None and number is not None:
        filename = f'{n}_{d}_{m}_{p}_e{float(number):.3f}'
      else:
        print('must provide a file header, number, or scheme sizes')
    import pickle
    file1 = open(f'{filename}_alpha_pnd.pkl', 'rb')
    alpha_pnd = pickle.load(file1)
    file1.close()
    file1 = open(f'{filename}_beta__pdm.pkl', 'rb')
    beta__pdm = pickle.load(file1)
    file1.close()
    file1 = open(f'{filename}_gamma_nmp.pkl', 'rb')
    gamma_nmp = pickle.load(file1)
    file1.close()

    from brentscheme.SchemeManipulator import SchemeManipulator
    SchemeManipulator().set(scheme, alpha_pnd, beta__pdm, gamma_nmp)
    self.set_TRIPLE_DELTA(scheme)
    if verbose > 0:
      from brentscheme.SchemeDisplay import SchemeDisplay
      SchemeDisplay().print(scheme, verbose=verbose)

  def compose_schemes(self, outer, inner): # cCaAbBi, zZxXyY
    from brentscheme.BrentScheme import BrentScheme
    result = BrentScheme()
    result.n = outer.n * inner.n
    result.d = outer.d * inner.d
    result.m = outer.m * inner.m
    result.p = outer.p * inner.p
    SchemaFactory().set_TRIPLE_DELTA(result)
    result.gamma_nmp = torch.einsum('cCi,zZj->czCZij', outer.gamma_nmp, inner.gamma_nmp).reshape((result.n, result.m, result.p))
    result.alpha_pnd = torch.einsum('iaA,jxX->ijaxAX', outer.alpha_pnd, inner.alpha_pnd).reshape((result.p, result.n, result.d))
    result.beta__pdm = torch.einsum('ibB,jyY->ijbyBY', outer.beta__pdm, inner.beta__pdm).reshape((result.p, result.d, result.m))
    return result
