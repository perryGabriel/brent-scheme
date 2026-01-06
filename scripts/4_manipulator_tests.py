from brentscheme.BrentScheme import BrentScheme
from brentscheme.misc import random_unitary, rand_square, random_right_invertible
from brentscheme.SchemaFactory import SchemaFactory
from brentscheme.SchemeDisplay import SchemeDisplay
from brentscheme.SchemeManipulator import SchemeManipulator
import numpy as np
import torch

scheme = BrentScheme(n=3, d=2, m=4, p=8, verbose=0)
factory = SchemaFactory()
printer = SchemeDisplay()
manipulator = SchemeManipulator()

print("="*40)
print("Setting different norms and fields, printing scheme, norm, and log error for each...")
print("Errors thrown if norm and measure functions fail the tests...")

manipulator.set_norm(scheme, norm=1, field='R')
printer.print(scheme)
assert scheme.norm(torch.Tensor([1,2,3])).item() == 6.0
assert scheme.measure(torch.Tensor([1,2,3])).item() == 2.0

manipulator.set_norm(scheme, norm=2, field='R')
printer.print(scheme)
assert scheme.norm(torch.Tensor([1,2,3])).item() == 14.0
assert np.abs(scheme.measure(torch.Tensor([1,2,3])).item() - 2.1602468490600586) < 1e-6

manipulator.set_norm(scheme, norm=3, field='R')
printer.print(scheme)
assert scheme.norm(torch.Tensor([1,2,3])).item() == 36.0
assert np.abs(scheme.measure(torch.Tensor([1,2,3])).item() - 2.289428472518921) < 1e-6

manipulator.set_norm(scheme, norm=torch.inf, field='R')
printer.print(scheme)
assert scheme.norm(torch.Tensor([1,2,3])).item() == 3.0
assert scheme.measure(torch.Tensor([1,2,3])).item() == 3.0

manipulator.set_norm(scheme, norm=1, field='C')
printer.print(scheme)
assert scheme.norm(torch.Tensor([1,2,3])).item() == 6.0
assert scheme.measure(torch.Tensor([1,2,3])).item() == 2.0

manipulator.set_norm(scheme, norm=torch.inf, field='C')
printer.print(scheme)
assert scheme.norm(torch.Tensor([1,2,3])).item() == 3.0
assert scheme.measure(torch.Tensor([1,2,3])).item() == 3.0

print("="*40)
print("Testing random unitary generator and basis change function...")
for n in range(1,7):
  A = random_unitary(2)
  if torch.sum(torch.abs(A @ A.T - torch.eye(2))) > 10**-14:
    print("Unitary matrix generator failed for n=", n)

for n in range(1,6):
  for d in range(1,6):
    for m in range(1,6):
      factory.set_scheme(scheme, preset='naive', n=n, d=d, m=m)
      manipulator.change_basis(scheme, L=random_unitary(n), M=random_unitary(d), R=random_unitary(m))
      if printer.test(scheme) > -14:
        print("failed on test: ", n,d,m)

print("="*40)
print("Testing random square matrices to check pseudoinverse stability...")
for n in range(1,7):
  for d in range(1,7):
    for m in range(1,7):
      factory.set_scheme(scheme, preset='naive', n=n, d=d, m=m)
      manipulator.change_basis(scheme, L=rand_square(n), M=rand_square(d), R=rand_square(m))
      if printer.test(scheme) > -11: # bad conditioning introduces some error
          print("failed on accuracy: test", n,d,m,printer.test(scheme))

print("="*40)
print("Testing random right invertible generator and basis change function...")
for n in range(2,4):
  for m in range(n-1, n+3):
    A = random_right_invertible(l=n,r=m)
    A_inv = torch.linalg.pinv(A)
    if torch.sum(torch.abs(A @ A_inv - torch.eye(n))) > 10**-14:
      print("Right Invertable matrix generator failed for n=", n)

for n in range(2,4):
  for d in range(2,4):
    for m in range(2,4):
      for d1 in range(d-1, d+1):
        factory.set_scheme(scheme, preset='naive', n=n, d=d, m=m)
        # d1 <= d
        manipulator.change_basis(scheme, M=random_right_invertible(l=d1,r=d))
        if printer.test(scheme) > -14:
          if d1 < d:
            print("failed on accuracy: test", n,d,d1,m,printer.test(scheme))
          else:
            print("failed on square basis!")
        if scheme.p != n*d*m or scheme.alpha_pnd.size(0) != scheme.p: # number of products stays the same as the larger scheme (inefficient)
          print("failed on size: test", n,d,d1,m,scheme.p)
        if scheme.alpha_pnd.size(2) != d1 or scheme.beta__pdm.size(1) != d1: # number of products stays the same as the larger scheme (inefficient)
          print("failed on axes: test", n,d,d1,m,scheme.p)

for n in range(3,6):
  for d in range(3,6):
    for m in range(3,6):
      for n1 in range(n-2, n+1):
        for d1 in range(d-2, d+1):
          for m1 in range(m-2, m+1):
            factory.set_scheme(scheme, preset='naive', n=n, d=d, m=m)
            # k1 <= k for all k
            manipulator.change_basis(scheme, L=random_right_invertible(l=n1,r=n).T, M=random_right_invertible(l=d1,r=d), R=random_right_invertible(l=m1,r=m))
            if printer.test(scheme) > -14:
              if n < n1 or m < m1 or d < d1:
                print("failed on accuracy: test", n,n1,d,d1,m,m1,printer.test(scheme))
              else:
                print("failed on square basis!")
            if scheme.p != n*d*m or scheme.gamma_nmp.size(2) != scheme.p: # number of products stays the same as the larger scheme (inefficient)
              print("failed on size: test", n,n1,d,d1,m,m1,scheme.p)
            if scheme.gamma_nmp.size(0) != n1 or scheme.beta__pdm.size(2) != m1: # number of products stays the same as the larger scheme (inefficient)
              print("failed on axes: test", n,d,d1,m,scheme.p)

print("="*40)
print("Testing product permutation function...")
# see if the products are in groups of four according to tpye of product (test sorting products)
factory.set_scheme(scheme, 'strassen')
printer.print(scheme, verbose=2)
# then mess them up
permute = np.arange(scheme.p)
np.random.shuffle(permute)
print(permute)
manipulator.permute_products(scheme, permutation=permute)
printer.print(scheme, verbose=2)

print("="*40)
print("Testing noise addition, chopping, and rounding functions...")
factory.set_scheme(scheme, 'naive', n=2)
manipulator.add_noise(scheme, epsilon=10**-3)
print("-3: ", printer.test(scheme))
manipulator.round(scheme, sig_figs=2)
if printer.test(scheme) != -np.inf:
  print("rounding didn't work")
manipulator.add_noise(scheme, epsilon=10**-3)
manipulator.chop(scheme, num=1, verbose=2)
manipulator.round(scheme, sig_figs=2)
printer.print(scheme)

print("="*40)
print("Testing zero number enforcement function...")
factory.set_scheme(scheme, 'naive', n=2, d=2, m=2)
manipulator.change_basis(scheme, L=random_unitary(2), M=random_unitary(2), R=random_unitary(2))

printer.print(scheme, verbose=2)
manipulator.enforce_zero_num(scheme, num_zeros_enforced=[2,2,2], decay_factor=0.0)
printer.print(scheme, verbose=2)

print("="*40)
print("Testing matrix reduction and cleaning functions...")
factory.set_scheme(scheme, 'laderman', n=3)
manipulator.reduce_matrices(scheme, axes=[[1], [0], [2]])
manipulator.clean(scheme)
printer.print(scheme, verbose=2)

print("="*40)
print("Testing normalization function...")
manipulator.normalize(scheme)
printer.print(scheme, verbose=2)