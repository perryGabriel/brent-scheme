import torch

#@title Misc Functions
def permutation_matrix(indices):
  n = len(indices)
  matrix = torch.zeros((n, n), dtype=float)
  for i, idx in enumerate(indices):
      matrix[i, idx] = 1.0
  return matrix