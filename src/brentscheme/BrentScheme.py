from .BrentScheme import BrentScheme
from .misc import permutation_matrix
from .SchemaFactory import SchemaFactory
from .SchemeDisplay import SchemeDisplay
from .SchemeManipulator import SchemeManipulator
from .Stepper import Stepper
from .Trainer import Trainer

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

# @title A Scheme for multiplying matrices
class BrentScheme(nn.Module):
  def __init__(self, p=None, n=None, d=None, m=None, preset='random', verbose=0, **kwargs):
    """
    Args:
      n, d, m: perfoms a matrix multiplication on (n x d) @ (d x m) = (n x m)
      p: the number of base field products used in the multiplication scheme
      *args:
      **kwargs:
    """
    super(BrentScheme, self).__init__()
    self.__dict__.update(kwargs)
    self.n = 2 if n is None else n
    self.d = self.n if d is None else d
    self.m = self.n if m is None else m
    self.p = self.n*self.d*self.m if p is None else p
    if verbose > 0: print(f"A scheme for ({self.n} x {self.d}) @ ({self.d} x {self.m}) using {self.p} products: complexity is n^{self.complexity():.3f}")

    factory = SchemaFactory()
    factory.set_scheme(self, preset=preset)
    factory.set_TRIPLE_DELTA(self)

  def clone(self):
    test_scheme = BrentScheme()
    manipulator.set(test_scheme, self.alpha_pnd.clone(), self.beta__pdm.clone(), self.gamma_nmp.clone())
    return test_scheme

  def complexity(self):
    _size = self.n*self.d*self.m
    if _size == 1: return self.p
    return 3 * math.log(self.p, _size)

  def measure(self, x):
    if self.L_norm == torch.inf:
      return self.inv_norm(self.norm(x))
    else:
      return self.inv_norm(self.norm(x) / torch.numel(x))

  def forward(self, A_nd=None, B_dm=None):
    if A_nd is None or B_dm is None:
      return torch.einsum("cCi,iaA,ibB->cCaAbB", self.gamma_nmp, self.alpha_pnd, self.beta__pdm)
    else:
      return torch.einsum("cCi,iaA,ibB,aA,bB->cC", self.gamma_nmp, self.alpha_pnd, self.beta__pdm, A_nd, B_dm)

scheme = BrentScheme()