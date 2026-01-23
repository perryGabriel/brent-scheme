from brentscheme.BrentScheme import BrentScheme
from brentscheme.utils.tensors import random_unitary
from brentscheme.utils.io import delete_file
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
print("Testing scheme saving, reading, and deleting a random scheme...")
# Save the scheme to files
score = printer.dump_tensors(scheme)
print(f"Score: {score}")
# Read into locals() a scheme from files. Remember to specify which scheme you want!
factory.read_from_files(scheme, n=scheme.n, d=scheme.d, m=scheme.m, p=scheme.p, number=score, verbose=2)
# Delete a scheme from files. Remember to specify which scheme you want!
delete_file(n=scheme.n, d=scheme.d, m=scheme.m, p=scheme.p, number=score, scheme_or_diagram='scheme')

print("="*40)
print("Testing reading, writing, and deleting a naive scheme...")
factory.set_scheme(scheme, preset='naive', n=2, d=2, m=2)
printer.dump_tensors(scheme, score=10)
factory.read_from_files(scheme, n=2, d=2, m=2, p=8, number=10, verbose=2)
factory.read_from_files(scheme, filename="2_2_2_8_e10.000", verbose=2)
delete_file(n=2, d=2, m=2, p=8, number=10, scheme_or_diagram='scheme')

print("="*40)
print("Testing all random and complex schemes for n,d,m in [1,8], printing log error floats...")
for n in range(1,9):
  for d in range(1,9):
    for m in range(1,9):
      factory.set_scheme(scheme, preset='random', n=n, d=d, m=m)
      if n == d == m: print(printer.error(scheme))
for n in range(1,9):
  for d in range(1,9):
    for m in range(1,9):
      factory.set_scheme(scheme, preset='complex', n=n, d=d, m=m)
      if n == d == m: print(printer.error(scheme))

print("="*40)
print("Testing all naive schemes for n,d,m in [1,6], printing 'failed on test: n' if test fails...")
for n in range(1,7):
  for d in range(1,7):
    for m in range(1,7):
      factory.set_scheme(scheme, preset='naive', n=n, d=d, m=m)
      if printer.error(scheme) != -np.inf:
        print("failed on test: ",n)

print("="*40)
print("Testing all fourier schemes for n,d,m in [1,5] and levels in [0,2], printing 'failed on test: n,d,m,level' if test fails...")
max_size = 5
for n in range(1,max_size):
  for d in range(1,max_size):
    for m in range(1,max_size):
      for level in range(0,3):
        factory.set_scheme(scheme, fourier=level, n=n, d=d, m=m)
        if printer.error(scheme) > -14:
            print("failed on test: ",n,d,m,level, printer.error(scheme))

print("="*40)
print("Testing all preset schemes (strassen, winograd, laderman), printing 'failed on test: preset' if test fails...")
factory.set_scheme(scheme, preset='strassen')
if printer.error(scheme) != -np.inf:
  print("failed on test: strassen")
factory.set_scheme(scheme, preset='winograd')
if printer.error(scheme) > -14:
  print("failed on test: winograd")
factory.set_scheme(scheme, preset='laderman')
if printer.error(scheme) > -14:
  print("failed on test: laderman")

print("="*40)
print("Testing scheme composition, printing failures...")
outer = BrentScheme()
factory.set_scheme(outer, 'strassen', n=2)
inner = BrentScheme()
factory.set_scheme(inner, 'strassen', n=2)

result = factory.compose_schemes(outer, inner)
if result.n != outer.n * inner.n:
  print(f"Matrix Sizes Failed for test 1")
if result.p != outer.p * inner.p:
  print(f"Product Size Failed for test 1")
if printer.error(result) > -13:
  print(f"Accuracy Failed for test 1")


outer = BrentScheme()
factory.set_scheme(outer, 'strassen', n=2)
inner = BrentScheme()
factory.set_scheme(inner, 'laderman', n=2)

result = factory.compose_schemes(outer, inner)
if result.n != outer.n * inner.n:
  print(f"Matrix Sizes Failed for test 2")
if result.p != outer.p * inner.p:
  print(f"Product Size Failed for test 2")
if printer.error(result) > -13:
  print(f"Accuracy Failed for test 2")

print("="*40)
print("Final test: naive (3x3) @ (3x3) scheme, 10 random multiplications, log error should be < -14 each time, else throws error...")
factory.set_scheme(scheme, 'naive', n=3, d=3, m=3)
for i in range (10):
  A = random_unitary(3)
  B = random_unitary(3)
  if torch.log10(scheme.measure(scheme(A, B) - (A @ B))) > -14:
    print("failure")

print("="*40)
print("Testing degenerate schemes, printing failures...")

# inner product, solve alpha
factory.set_scheme(scheme, 'random', n=1, d=4, m=1, p=4)
factory.degenerate_scheme(scheme, alpha_pnd=None, beta__pdm=scheme.beta__pdm, gamma_nmp=scheme.gamma_nmp)
assert scheme.n == 1 and scheme.d == 4 and scheme.m == 1 and scheme.p == 4
assert printer.error(scheme) < -12

# inner product, solve beta
factory.set_scheme(scheme, 'random', n=1, d=4, m=1, p=4)
factory.degenerate_scheme(scheme, alpha_pnd=scheme.alpha_pnd, beta__pdm=None, gamma_nmp=scheme.gamma_nmp)
assert scheme.n == 1 and scheme.d == 4 and scheme.m == 1 and scheme.p == 4
assert printer.error(scheme) < -12 

# outer product, solve gamma
factory.set_scheme(scheme, 'random', n=2, d=1, m=2, p=4)
factory.degenerate_scheme(scheme, alpha_pnd=scheme.alpha_pnd, beta__pdm=scheme.beta__pdm, gamma_nmp=None)
assert scheme.n == 2 and scheme.d == 1 and scheme.m == 2 and scheme.p == 4
assert printer.error(scheme) < -12

# vector-matrix product, solve beta
factory.set_scheme(scheme, 'random', n=1, d=2, m=3, p=6)
factory.degenerate_scheme(scheme, alpha_pnd=scheme.alpha_pnd, beta__pdm=None, gamma_nmp=scheme.gamma_nmp)
assert scheme.n == 1 and scheme.d == 2 and scheme.m == 3 and scheme.p == 6
assert printer.error(scheme) < -12

# matrix-vector product, solve alpha
factory.set_scheme(scheme, 'random', n=3, d=2, m=1, p=6)
factory.degenerate_scheme(scheme, alpha_pnd=None, beta__pdm=scheme.beta__pdm, gamma_nmp=scheme.gamma_nmp)
assert scheme.n == 3 and scheme.d == 2 and scheme.m == 1 and scheme.p == 6
assert printer.error(scheme) < -12

