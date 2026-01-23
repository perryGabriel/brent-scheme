

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

# @title A Manipulation object for an existing scheme
class SchemeManipulator(object):
  from brentscheme.utils.tensors import permutation_matrix
  
  def __init__(self):
    pass

  # set the member tensors manually
  def set(self, scheme, alpha_pnd, beta__pdm, gamma_nmp, norm=1):
    scheme.p = alpha_pnd.shape[0]
    scheme.n = alpha_pnd.shape[1]
    scheme.d = alpha_pnd.shape[2]
    scheme.m = gamma_nmp.shape[1]

    scheme.alpha_pnd = alpha_pnd
    scheme.beta__pdm = beta__pdm
    scheme.gamma_nmp = gamma_nmp

    self.set_norm(scheme, norm=norm)
    from brentscheme.SchemaFactory import SchemaFactory
    SchemaFactory().set_triple_delta(scheme)

  # change the norm or feild of the scheme (L1, L2, etc, R, C)
  def set_norm(self, scheme, norm=1, field=None):
    scheme.L_norm = norm

    if "field" not in scheme.__dict__:
      scheme.field = 'R'
      if scheme.alpha_pnd.dtype == torch.complex128 or scheme.beta__pdm == torch.complex128 or scheme.gamma_nmp == torch.complex128: field = 'C'
    else: scheme.field = field

    if scheme.field == 'R':
      scheme.alpha_pnd = torch.real(scheme.alpha_pnd).type(torch.float64)
      scheme.beta__pdm = torch.real(scheme.beta__pdm).type(torch.float64)
      scheme.gamma_nmp = torch.real(scheme.gamma_nmp).type(torch.float64)
    elif scheme.field == 'C':
      scheme.alpha_pnd = scheme.alpha_pnd.type(torch.complex128)
      scheme.beta__pdm = scheme.beta__pdm.type(torch.complex128)
      scheme.gamma_nmp = scheme.gamma_nmp.type(torch.complex128)

    if norm < 1:
      print(f"THE L{norm} NORM IS NOT SUPPORTED: TRY [1,inf]")
    elif norm == torch.inf:
      scheme.norm = lambda x : torch.max(torch.abs(x))
      scheme.inv_norm = lambda x : x
    elif norm == 1: # specialize for speedup
      scheme.norm = lambda x : torch.sum(torch.abs(x))
      scheme.inv_norm = lambda x : x # a norm should be F -> R, so this is ok
    else:
      scheme.norm = lambda x : torch.sum(torch.abs(x)**norm)
      scheme.inv_norm = lambda x : x**(1.0/norm) # a norm should be F -> R, so this is ok

  ### CLEAN UP AN EXISTING SCHEME ###

  # enforce that alpha and beta have normalized components along p, using current or passed norm
  def normalize(self, scheme, verbose=0):
    if verbose > 1: 
      from brentscheme.SchemeDisplay import SchemeDisplay
      SchemeDisplay().print(scheme)
    self.clean(scheme) # get rid of zero-norm products

    # make the matrix basis have balanced axis norms

    # L basis
    if scheme.L_norm == np.inf:
      alpha_mags_n = torch.from_numpy(np.array([torch.max(torch.abs(scheme.alpha_pnd[:,ni,:])) for ni in range(scheme.n)]))
      gamma_mags_n = torch.from_numpy(np.array([torch.max(torch.abs(scheme.gamma_nmp[ni,:,:])) for ni in range(scheme.n)]))
    else:
      alpha_mags_n = torch.einsum("pnd->n", torch.abs(scheme.alpha_pnd)**scheme.L_norm)**(1/scheme.L_norm)
      gamma_mags_n = torch.einsum("nmp->n", torch.abs(scheme.gamma_nmp)**scheme.L_norm)**(1/scheme.L_norm)

    avg_mags_n = (alpha_mags_n * gamma_mags_n)**(1/2)
    avg_mags_n, permutation_n = torch.sort(avg_mags_n, descending=True)
    scheme.alpha_pnd = scheme.alpha_pnd[:,permutation_n,:] / alpha_mags_n[permutation_n].reshape(1,-1,1) * avg_mags_n.reshape(1,-1,1)
    scheme.gamma_nmp = scheme.gamma_nmp[permutation_n,:,:] / gamma_mags_n[permutation_n].reshape(-1,1,1) * avg_mags_n.reshape(-1,1,1)

    if verbose > 0: print("new n magnitudes: ", avg_mags_n)

    # M basis
    if scheme.L_norm == np.inf:
      alpha_mags_d = torch.from_numpy(np.array([torch.max(torch.abs(scheme.alpha_pnd[:,:,di])) for di in range(scheme.d)]))
      beta__mags_d = torch.from_numpy(np.array([torch.max(torch.abs(scheme.beta__pdm[:,di,:])) for di in range(scheme.d)]))
    else:
      alpha_mags_d = torch.einsum("pnd->d", torch.abs(scheme.alpha_pnd)**scheme.L_norm)**(1/scheme.L_norm)
      beta__mags_d = torch.einsum("pdm->d", torch.abs(scheme.beta__pdm)**scheme.L_norm)**(1/scheme.L_norm)

    avg_mags_d = (alpha_mags_d * beta__mags_d)**(1/2)
    avg_mags_d, permutation_d = torch.sort(avg_mags_d, descending=True)
    scheme.alpha_pnd = scheme.alpha_pnd[:,:,permutation_d] / alpha_mags_d[permutation_d].reshape(1,1,-1) * avg_mags_d.reshape(1,1,-1)
    scheme.beta__pdm = scheme.beta__pdm[:,permutation_d,:] / beta__mags_d[permutation_d].reshape(1,-1,1) * avg_mags_d.reshape(1,-1,1)

    if verbose > 0: print("new d magnitudes: ", avg_mags_d)

    # R basis
    if scheme.L_norm == np.inf:
      beta__mags_m = torch.from_numpy(np.array([torch.max(torch.abs(scheme.beta__pdm[:,:,mi])) for mi in range(scheme.m)]))
      gamma_mags_m = torch.from_numpy(np.array([torch.max(torch.abs(scheme.gamma_nmp[:,mi,:])) for mi in range(scheme.m)]))
    else:
      beta__mags_m = torch.einsum("pdm->m", torch.abs(scheme.beta__pdm)**scheme.L_norm)**(1/scheme.L_norm)
      gamma_mags_m = torch.einsum("nmp->m", torch.abs(scheme.gamma_nmp)**scheme.L_norm)**(1/scheme.L_norm)

    avg_mags_m = (beta__mags_m * gamma_mags_m)**(1/2)
    avg_mags_m, permutation_m = torch.sort(avg_mags_m, descending=True)
    scheme.beta__pdm = scheme.beta__pdm[:,:,permutation_m] / beta__mags_m[permutation_m].reshape(1,1,-1) * avg_mags_m.reshape(1,1,-1)
    scheme.gamma_nmp = scheme.gamma_nmp[:,permutation_m,:] / gamma_mags_m[permutation_m].reshape(1,-1,1) * avg_mags_m.reshape(1,-1,1)

    if verbose > 0: print("new m magnitudes: ", avg_mags_m)

    # make the products of equal norms

    if scheme.L_norm == np.inf:
      alpha_mags_p = torch.from_numpy(np.array([torch.max(torch.abs(scheme.alpha_pnd[pi,:,:])) for pi in range(scheme.p)]))
      beta__mags_p = torch.from_numpy(np.array([torch.max(torch.abs(scheme.beta__pdm[pi,:,:])) for pi in range(scheme.p)]))
      gamma_mags_p = torch.from_numpy(np.array([torch.max(torch.abs(scheme.gamma_nmp[:,:,pi])) for pi in range(scheme.p)]))
    else:
      alpha_mags_p = torch.einsum("pnd->p", torch.abs(scheme.alpha_pnd)**scheme.L_norm)**(1/scheme.L_norm)
      beta__mags_p = torch.einsum("pdm->p", torch.abs(scheme.beta__pdm)**scheme.L_norm)**(1/scheme.L_norm)
      gamma_mags_p = torch.einsum("nmp->p", torch.abs(scheme.gamma_nmp)**scheme.L_norm)**(1/scheme.L_norm)

    avg_mags_p = (alpha_mags_p * beta__mags_p * gamma_mags_p)**(1/3)
    avg_mags_p, permutation_p = torch.sort(avg_mags_p, descending=True)
    scheme.alpha_pnd = scheme.alpha_pnd[permutation_p,:,:] / alpha_mags_p[permutation_p].reshape(-1,1,1) * avg_mags_p.reshape(-1,1,1)
    scheme.beta__pdm = scheme.beta__pdm[permutation_p,:,:] / beta__mags_p[permutation_p].reshape(-1,1,1) * avg_mags_p.reshape(-1,1,1)
    scheme.gamma_nmp = scheme.gamma_nmp[:,:,permutation_p] / gamma_mags_p[permutation_p].reshape(1,1,-1) * avg_mags_p.reshape(1,1,-1)

    if verbose > 0: print("new product magnitudes: ", avg_mags_p)

  # if the tensors have a zero magnitude in a p-axis cross section, drop that p index
  def clean(self, scheme):
    for prod in range(scheme.p-1, -1, -1):
      if scheme.norm(scheme.alpha_pnd[prod,:,:]) == 0 or scheme.norm(scheme.beta__pdm[prod,:,:]) == 0 or scheme.norm(scheme.gamma_nmp[:,:,prod]) == 0:
         self.drop_product(scheme, prod=prod+1)

  ### MANUALLY MUTATE AN EXISTING SCHEME ###

  # rounds each coefficient to a certian significant place
  def round(self, scheme, sig_figs=15):
    scheme.alpha_pnd = scheme.alpha_pnd.round(decimals=sig_figs)
    scheme.beta__pdm  = scheme.beta__pdm.round(decimals=sig_figs)
    scheme.gamma_nmp = scheme.gamma_nmp.round(decimals=sig_figs)

  # adds random noise to the scheme's coefficients
  def add_noise(self, scheme, epsilon=10**-15):
    scheme.alpha_pnd += torch.normal(0, epsilon, size=(scheme.p, scheme.n, scheme.d)).type(torch.float64)
    scheme.beta__pdm += torch.normal(0, epsilon, size=(scheme.p, scheme.d, scheme.m)).type(torch.float64)
    scheme.gamma_nmp += torch.normal(0, epsilon, size=(scheme.n, scheme.m, scheme.p)).type(torch.float64)

  # drops the least significant product from the central sum over p
  def chop(self, scheme, num=0, verbose=0):
    for i in range(num):
      gamma_scores_p = torch.einsum("nmp->p", torch.abs(scheme.gamma_nmp)) #FIXME: max along cross section at _p
      if verbose > 1: print(f"The gamma 2-tensors associated with each product have magnitudes: {gamma_scores_p}")
      _g = torch.argmin(gamma_scores_p)
      if verbose > 1: print(f"Dropping the product with magnitude: {gamma_scores_p[_g]}")
      self.drop_product(scheme, prod=_g+1, verbose=verbose)

  # scales the num_zeros_enforced smallest coefficients by decay_factor
  def enforce_zero_num(self, scheme, num_zeros_enforced=[0,0,0], decay_factor=0.7):
    for array_i, curr_zeros in enumerate(num_zeros_enforced):
      if curr_zeros == 0: continue
      array = [scheme.alpha_pnd, scheme.beta__pdm, scheme.gamma_nmp][array_i]
      elem_mag_to_set,_ = torch.sort(torch.abs(array).flatten())
      elem_mag_to_set = elem_mag_to_set[curr_zeros]
      array[torch.where(torch.abs(array)<=elem_mag_to_set)] = 0.0

  ### MAKE A SMALLER SCHEME ###

  # drop rows or columns from the scheme to reduce n,d,m dimentions
  def reduce_matrices(self, scheme, axes=[[],[],[]]):
    for i in range(3):
      if axes[i] is None:
        axes[i] = []
      elif isinstance(axes[i], int):
        axes[i] = [axes[i]]
      axes[i].sort(reverse=True)

    for _n in axes[0]:
      scheme.gamma_nmp = torch.concatenate((scheme.gamma_nmp[:_n,:,:], scheme.gamma_nmp[_n+1:,:,:]), axis=0)
      scheme.alpha_pnd = torch.concatenate((scheme.alpha_pnd[:,:_n,:], scheme.alpha_pnd[:,_n+1:,:]), axis=1)
    for _d in axes[1]:
      scheme.alpha_pnd = torch.concatenate((scheme.alpha_pnd[:,:,:_d], scheme.alpha_pnd[:,:,_d+1:]), axis=2)
      scheme.beta__pdm = torch.concatenate((scheme.beta__pdm[:,:_d,:], scheme.beta__pdm[:,_d+1:,:]), axis=1)
    for _m in axes[2]:
      scheme.beta__pdm = torch.concatenate((scheme.beta__pdm[:,:,:_m], scheme.beta__pdm[:,:,_m+1:]), axis=2)
      scheme.gamma_nmp = torch.concatenate((scheme.gamma_nmp[:,:_m,:], scheme.gamma_nmp[:,_m+1:,:]), axis=1)

    scheme.n -= len(axes[0])
    scheme.d -= len(axes[1])
    scheme.m -= len(axes[2])
    from brentscheme.SchemaFactory import SchemaFactory
    SchemaFactory().set_triple_delta(scheme)

  # drops the indicated product P_prod (int)
  def drop_product(self, scheme, prod=None, verbose=0):
    if prod is None: return
    prod -= 1 # UI is 1-indexed, tensor is 0-indexed
    scheme.alpha_pnd = torch.concatenate((scheme.alpha_pnd[:prod,:,:], scheme.alpha_pnd[prod+1:,:,:]), axis=0)
    scheme.beta__pdm = torch.concatenate((scheme.beta__pdm[:prod,:,:], scheme.beta__pdm[prod+1:,:,:]), axis=0)
    scheme.gamma_nmp = torch.concatenate((scheme.gamma_nmp[:,:,:prod], scheme.gamma_nmp[:,:,prod+1:]), axis=2)

    scheme.p = scheme.p - 1
    if verbose > 0: print(f'p is now: {scheme.p}')

  ### CHANGING AXES BASES ###

  # change of basis on n,d,m axes - uses normal matrix multiplication bases
  def change_basis(self, scheme, L=None, M=None, R=None):
    # works best with well-conditioned matrices and vectors close to ones().
    # basis change is: (L)C(R) = (L)A(M)(M_inv)B(R)
    # where the new scheme uses C' = (L)C(R), A' = (L)A(M), and B' = (M_inv)B(R)

    # L is shape nN, M is shape Dd, R is shape Mm (capitals are new, small axis, lowercase are the old, larger size/invertable axis)
    # L is left invertable, R is right invertable, M is right invertable

    if L is None: L = torch.eye(scheme.n)
    if M is None: M = torch.eye(scheme.d)
    if R is None: R = torch.eye(scheme.m)

    scheme.field = 'R'
    new_type = torch.float64
    if scheme.alpha_pnd.dtype == torch.complex128 or scheme.beta__pdm == torch.complex128 or scheme.gamma_nmp == torch.complex128 or \
    L.dtype == torch.complex128 or M.dtype == torch.complex128 or R.dtype == torch.complex128:
      scheme.field = 'C'
      new_type = torch.complex128

    scheme.alpha_pnd = scheme.alpha_pnd.type(new_type)
    scheme.beta__pdm = scheme.beta__pdm.type(new_type)
    scheme.gamma_nmp = scheme.gamma_nmp.type(new_type)

    L = L.type(new_type)
    R = R.type(new_type)
    M = M.type(new_type)

    L_inv = torch.linalg.pinv(L)
    R_inv = torch.linalg.pinv(R)
    M_inv = torch.linalg.pinv(M)

    scheme.alpha_pnd = torch.einsum("nN,Dd,pnd->pND", L,            M    , scheme.alpha_pnd).type(new_type)
    scheme.beta__pdm = torch.einsum("Mm,dD,pdm->pDM",        R,     M_inv, scheme.beta__pdm).type(new_type)
    scheme.gamma_nmp = torch.einsum("Nn,mM,nmp->NMp", L_inv, R_inv,        scheme.gamma_nmp).type(new_type)

    scheme.n = scheme.alpha_pnd.size(dim=1)
    scheme.d = scheme.alpha_pnd.size(dim=2)
    scheme.m = scheme.gamma_nmp.size(dim=1)
    scheme.p = scheme.alpha_pnd.size(dim=0)
    from brentscheme.SchemaFactory import SchemaFactory
    factory = SchemaFactory()
    factory.set_triple_delta(scheme)

    self.set_norm(scheme, norm=scheme.L_norm, field=scheme.field)

  # insert a ones vector along the p-axis sum and decompose it arbirarily
  def scale_products(self, scheme, a, b):
    a = torch.Tensor(a)
    b = torch.Tensor(b)
    scheme.alpha_pnd = torch.einsum("pnd,p->pnd", scheme.alpha_pnd, a)
    scheme.beta__pdm = torch.einsum("pdm,p->pdm", scheme.beta__pdm, b)
    scheme.gamma_nmp = torch.einsum("nmp,p,p->nmp", scheme.gamma_nmp, 1/a, 1/b)

  def permute_products(self, scheme, permutation=None, verbose=0):
    if permutation is None: return # clustering not implemented in PyTorch colab
    #   gamma, products = scheme.flatten()
    #   u,_,__ = torch.linalg.svd(products)
    #   permutation = find_row_clustering(u, verbose=verbose)

    from brentscheme.utils.tensors import permutation_matrix
    permutation = permutation_matrix(permutation)
    scheme.alpha_pnd = torch.einsum('ij,jaA->iaA', permutation, scheme.alpha_pnd)
    scheme.beta__pdm = torch.einsum('ij,jbB->ibB', permutation, scheme.beta__pdm)
    scheme.gamma_nmp = torch.einsum('cCj,ij->cCi', scheme.gamma_nmp, permutation)
