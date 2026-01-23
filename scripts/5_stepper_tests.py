from brentscheme.BrentScheme import BrentScheme
from brentscheme.SchemaFactory import SchemaFactory
from brentscheme.SchemeDisplay import SchemeDisplay
from brentscheme.Stepper import Stepper

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