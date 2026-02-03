from brentscheme.BrentScheme import BrentScheme
from brentscheme.SchemaFactory import SchemaFactory
from brentscheme.SchemeManipulator import SchemeManipulator
from brentscheme.SchemeDisplay import SchemeDisplay
from brentscheme.Stepper import Stepper
import numpy as np
import torch

scheme = BrentScheme(n=3, d=2, m=4, p=8, verbose=0)
factory = SchemaFactory()
printer = SchemeDisplay()
stepper = Stepper()


print("="*40)
print("TEST 1: PSEUDOINVERSE STEP")
factory.set_scheme(scheme, 'random', n=3, p=22)
epochs = 50

for i in range(epochs):
  stepper.epoch_pseudoinverse(scheme)
if printer.error(scheme) > -1.0:
  print("Run pseudoinverse test again, may be faulty: ", printer.error(scheme))

print("="*40)
print("TEST 2: TORCH STEP")
factory.set_scheme(scheme, 'random', n=2, d=2, m=2, p=8)
epochs = 300

for i in range(epochs):
  stepper.epoch(scheme, momentum=0.9)
if printer.error(scheme) > -1:
  print("Run torch test again, may be faulty: ", printer.error(scheme))

print("="*40)
print("TEST 3: LINEAR PROGRAM using INF NORM")
np.random.seed(1)
scheme = BrentScheme(n=2,d=3,m=4,p=20)
printer.report(scheme, verbose=1)
printer = SchemeDisplay()
manipulator = SchemeManipulator()
stepper = Stepper()

L2_score = lambda x: torch.sum(torch.square(x.forward() - x.TRIPLE_DELTA_nmnddm))
Linf_score = lambda x: torch.max(torch.abs(x.forward() - x.TRIPLE_DELTA_nmnddm))
printer.plot_triple_deltas(scheme)

manipulator.set_norm(scheme, 1)

from tqdm import trange
for i in trange(100):
  # stepper.epoch_pseudoinverse(scheme, batch_size=10)
  if L2_score(scheme) < 1e-6:
    break
  stepper.optimize(scheme, batch_size=2, method=stepper.optimize_infinity_norm)
  if Linf_score(scheme) < 1e-6:
    break

  manipulator.normalize(scheme)

printer.report(scheme, verbose=1)
printer.plot_triple_deltas(scheme)