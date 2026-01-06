from brentscheme.misc import permutation_matrix
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

# @title A Multi-step trainer for schema
class Trainer(object):

  def __init__(self, n=2, p=7, scheme='random'):
    self.num_epochs = 0

  def train(self, scheme, epochs=200, batch_size=1, lr=1e-7, momentum=0.9, use_L2=False, penalty=0.0, verbose=0):
    try:
      stepper = Stepper()
      display = SchemeDisplay()
      y = [list(display.test(scheme, verbose=1))]
      import timeit

      start = timeit.timeit()
      for i in range(epochs):
        if use_L2:
          stepper.epoch_pseudoinverse(scheme, batch_size=batch_size, verbose=1)
        else:
          stepper.epoch(scheme, batch_size=batch_size, lr=lr, momentum=momentum, penalty=penalty)
        y.append(list(display.test(scheme, verbose=1)))
        # add the filter or other normalization here

      self.num_epochs = self.num_epochs + epochs*batch_size
    except:
      pass
    runtime = timeit.timeit() - start

    def fit_decay(x,y):
      from scipy.optimize import curve_fit
      # try:
      #   params, _ = curve_fit(lambda x,a,b,c:a * np.exp(-b * x) + c, x, y)
      # except:
      #   params = [0,0,0]
      # return params

    if verbose > 0:
      x = np.linspace(0,epochs,len(y))
      # params = fit_decay(x,y)

    if verbose > 1:
      # plt.axhline(y=params[2], color='black', linestyle='--')
      # plt.plot(x, params[0] * np.exp(-params[1] * x) + params[2], color='red', linestyle='--')

      plt.plot([i for i in range(epochs+1)], [j[0] for j in y], color='blue', label='L1')
      plt.title(f"Normalized Error For n={scheme.n}, p={scheme.p}: Ran in {np.abs(runtime):.4f} sec.")
      plt.xlabel(f"Number of Epochs: {self.num_epochs}")
      plt.ylabel(f"Average Error of Output Entries (Log 10)")
      plt.grid(axis='y')
      plt.legend()
      plt.show()

      plt.plot([i for i in range(epochs+1)], [j[1] for j in y], color='black', label='L2')
      plt.title(f"Normalized Error For n={scheme.n}, p={scheme.p}: Ran in {np.abs(runtime):.4f} sec.")
      plt.xlabel(f"Number of Epochs: {self.num_epochs}")
      plt.ylabel(f"Average Error of Output Entries (Log 10)")
      plt.grid(axis='y')
      plt.legend()
      plt.show()

      plt.plot([i for i in range(epochs+1)], [j[2] for j in y], color='red', label='Linf')
      plt.title(f"Normalized Error For n={scheme.n}, p={scheme.p}: Ran in {np.abs(runtime):.4f} sec.")
      plt.xlabel(f"Number of Epochs: {self.num_epochs}")
      plt.ylabel(f"Average Error of Output Entries (Log 10)")
      plt.grid(axis='y')
      plt.legend()
      plt.show()

    # if verbose > 0:
    #   return params

  # after optimizing in L2, optimize a change of basis to get Linf down. L1 will just not try hard enough.
  # unitary matrices are not Cholesky decomposable, nor are they constructable using this approach.
  # L2 error is invarient under change of basis - but others are not. Perhaps this suggests that these catchements are equivalent up to a basis change.
  def optimize_basis(self, scheme, batch_size=1000, lr=1e-6, loss_norm=np.inf, verbose=0):
    loss_fn = nn.L1Loss()
    pos = 0
    if loss_norm == np.inf:
      loss_fn = lambda x,y:torch.max(torch.abs(x-y))
      pos = 2

    score1 = printer.test(scheme, verbose=1)
    score2 = [np.inf] * 3
    while score2[pos] >= score1[pos]:
      L = torch.nn.Parameter(torch.eye(scheme.n) + 1e-14*random_unitary(scheme.n)).type(torch.float64).to(device)
      M = torch.nn.Parameter(torch.eye(scheme.d) + 1e-14*random_unitary(scheme.d)).type(torch.float64).to(device)
      R = torch.nn.Parameter(torch.eye(scheme.m) + 1e-14*random_unitary(scheme.m)).type(torch.float64).to(device)
      test_scheme = scheme.clone()
      manipulator.change_basis(test_scheme, L=L, M=M, R=R)
      score2 = printer.test(test_scheme, verbose=1)

    optimizer = optim.Adam([L, M, R], lr=lr)

    for i in range(batch_size):
      test_scheme = scheme.clone()
      optimizer.zero_grad()
      manipulator.change_basis(test_scheme, L=L, M=M, R=R)
      output = test_scheme.forward()
      target = test_scheme.TRIPLE_DELTA_nmnddm
      cost = loss_fn(output, target)
      cost.backward()
      optimizer.step()
      if verbose > 0 and i % (batch_size//10) == 0:
        display(cost.item())

    if cost < 10**score1[pos]:
      return L.cpu().detach().type(torch.float64), M.cpu().detach().type(torch.float64), R.cpu().detach().type(torch.float64)
    else:
      return torch.eye(scheme.n), torch.eye(scheme.d), torch.eye(scheme.m)

trainer = Trainer()