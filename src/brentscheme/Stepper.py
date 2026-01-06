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

# @title A Single-Step trainer for schema

class Stepper(object):

  def __init__(self):
    return

  # Construct the Jacobian: (output) cC aA bB   by   p1aA p2bB cCp3 (input = alpha,beta,gamma flatten())
  def Jacobian(self, scheme, test=False):
    Jacobian_alpha = torch.einsum("xa,XA,pbB,cCp->cCaAbBpxX", torch.eye(scheme.n), torch.eye(scheme.d), scheme.beta__pdm, scheme.gamma_nmp).reshape((scheme.n**2*scheme.d**2*scheme.m**2,scheme.p*scheme.n*scheme.d))
    Jacobian_beta = torch.einsum("xb,XB,paA,cCp->cCaAbBpxX", torch.eye(scheme.d), torch.eye(scheme.m), scheme.alpha_pnd, scheme.gamma_nmp).reshape((scheme.n**2*scheme.d**2*scheme.m**2,scheme.p*scheme.d*scheme.m))
    Jacobian_gamma = torch.einsum("xc,XC,paA,pbB->cCaAbBxXp", torch.eye(scheme.n), torch.eye(scheme.m), scheme.alpha_pnd, scheme.beta__pdm).reshape((scheme.n**2*scheme.d**2*scheme.m**2,scheme.p*scheme.n*scheme.m))
    Jacobian = torch.cat([Jacobian_alpha, Jacobian_beta, Jacobian_gamma], dim=1)
    if test:
      assert torch.allclose(3*scheme.forward().flatten(), Jacobian @ torch.cat([scheme.alpha_pnd.flatten(), scheme.beta__pdm.flatten(), scheme.gamma_nmp.flatten()]))
    return Jacobian

  # Computes gradient in all parameters simultaneously - grad in element-wise magnitude 1 according to scheme norm
  def get_abs_gradient_direction(self, scheme):
    params = - torch.linalg.pinv(self.Jacobian(scheme)) @ (scheme.forward().flatten() - scheme.TRIPLE_DELTA_nmnddm.flatten())
    params /= scheme.measure(params) # normalize
    delta_alpha = params[:scheme.p*scheme.n*scheme.d].reshape(scheme.p, scheme.n, scheme.d)
    delta_beta = params[scheme.p*scheme.n*scheme.d:scheme.p*scheme.n*scheme.d + scheme.p*scheme.d*scheme.m].reshape(scheme.p, scheme.d, scheme.m)
    delta_gamma = params[scheme.p*scheme.n*scheme.d + scheme.p*scheme.d*scheme.m:].reshape(scheme.n, scheme.m, scheme.p)
    return delta_alpha, delta_beta, delta_gamma

  def epoch_pseudoinverse(self, scheme, batch_size=10, verbose=0):
    for i in range(batch_size):
      AB_inv = torch.linalg.pinv(torch.einsum('iaA,ibB->iaAbB', scheme.alpha_pnd, scheme.beta__pdm).reshape((scheme.p, scheme.n*scheme.d**2*scheme.m))).reshape((scheme.n, scheme.d, scheme.d, scheme.m, scheme.p))
      scheme.gamma_nmp = torch.einsum('cCaAbB,aAbBi->cCi', scheme.TRIPLE_DELTA_nmnddm, AB_inv)

      AG_inv = torch.linalg.pinv(torch.einsum('iaA,cCi->iaAcC', scheme.alpha_pnd, scheme.gamma_nmp).reshape((scheme.p, scheme.n**2*scheme.d*scheme.m))).reshape((scheme.n, scheme.d, scheme.n, scheme.m, scheme.p))
      scheme.beta__pdm  = torch.einsum('cCaAbB,aAcCi->ibB', scheme.TRIPLE_DELTA_nmnddm, AG_inv)

      BG_inv = torch.linalg.pinv(torch.einsum('ibB,cCi->ibBcC', scheme.beta__pdm, scheme.gamma_nmp).reshape((scheme.p, scheme.n*scheme.d*scheme.m**2))).reshape((scheme.d, scheme.m, scheme.n, scheme.m, scheme.p))
      scheme.alpha_pnd = torch.einsum('cCaAbB,bBcCi->iaA', scheme.TRIPLE_DELTA_nmnddm, BG_inv)
    return

  def epoch(self, scheme, batch_size=10, lr=0.001, momentum=0.9, penalty=0.0):
    alpha = torch.nn.Parameter(scheme.alpha_pnd).type(torch.float64).to(device)
    beta  = torch.nn.Parameter(scheme.beta__pdm).type(torch.float64).to(device)
    gamma = torch.nn.Parameter(scheme.gamma_nmp).type(torch.float64).to(device)

    optimizer = optim.Adam([alpha, beta, gamma], lr=lr)

    loss_fn = nn.L1Loss()
    if scheme.L_norm > 1 and scheme.field == 'R':
      loss_fn = nn.MSELoss()
    if scheme.L_norm == torch.inf and scheme.field == 'R':
      loss_fn = lambda x,y:torch.max(x-y)

    for i in range(batch_size):
      optimizer.zero_grad()
      output = torch.einsum("cCi,iaA,ibB->cCaAbB", gamma, alpha, beta)
      target = scheme.TRIPLE_DELTA_nmnddm
      cost = loss_fn(output, target)
      if penalty > 0.0:
        cost = cost + penalty * (torch.sum(torch.abs(alpha)) + torch.sum(torch.abs(beta)) + torch.sum(torch.abs(gamma)))
      cost.backward()
      optimizer.step()

    scheme.alpha_pnd = alpha.cpu().detach().type(torch.float64)
    scheme.beta__pdm = beta.cpu().detach().type(torch.float64)
    scheme.gamma_nmp = gamma.cpu().detach().type(torch.float64)
    return
