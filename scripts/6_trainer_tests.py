from brentscheme.BrentScheme import BrentScheme
from brentscheme.SchemaFactory import SchemaFactory
from brentscheme.SchemeDisplay import SchemeDisplay
from brentscheme.SchemeManipulator import SchemeManipulator
from brentscheme.Stepper import Stepper
from brentscheme.Trainer import Trainer
import numpy as np
import torch

scheme = BrentScheme(n=3, d=2, m=4, p=8, verbose=0)
factory = SchemaFactory()
printer = SchemeDisplay()
manipulator = SchemeManipulator()
stepper = Stepper()
trainer = Trainer()

print("="*40)
print("TEST 1: Basic Trainer Test, no output")
factory.set_scheme(scheme, 'random', n=2, d=2, m=2, p=8)
trainer.train(scheme, epochs=100, batch_size=1, lr=1e-2, momentum=0.9, verbose=0)
#FIXME: Implement a tempurature related to accuracy

print("="*40)
print("TEST 2: L1 Norm Trainer Test, should be bad")
factory.set_scheme(scheme, 'random', n=3, d=3, m=3, p=22)
manipulator.set_norm(scheme, norm=1, field='R')
trainer.train(scheme, epochs=500, batch_size=10, use_L2=False, verbose=1)
printer.report(scheme, verbose=1)
printer.plot_triple_deltas(scheme)

print("="*40)
print("TEST 3: L2 Norm Trainer Test, should be bad")
factory.set_scheme(scheme, 'random', n=3, d=3, m=3, p=22)
trainer.train(scheme, epochs=500, batch_size=10, use_L2=False, verbose=1)
printer.report(scheme, verbose=1)
printer.plot_triple_deltas(scheme)

print("="*40)
print("TEST 2: L1000 Norm Trainer Test, should be bad")
factory.set_scheme(scheme, 'random', n=3, d=3, m=3, p=22)
manipulator.set_norm(scheme, norm=1000, field='R')
trainer.train(scheme, epochs=500, batch_size=10, use_L2=False, verbose=1)
printer.report(scheme, verbose=1)
printer.plot_triple_deltas(scheme)

print("="*40)
print("TEST 1: Train Using Pseudoinverse Projection, should be good")
factory.set_scheme(scheme, 'random', n=3, d=3, m=3, p=22)
trainer.train(scheme, epochs=500, batch_size=1, use_L2=True, verbose=1)
printer.plot_triple_deltas(scheme)

print("="*40)
print("TEST 2: Train Using Infinity Norm (LP), should be mid")
manipulator.set_norm(scheme, norm=torch.inf, field='R')
printer.report(scheme, verbose=1)
trainer.train(scheme, epochs=200, batch_size=10, lr=1e-5, penalty=1e-7, use_L2=False, verbose=1)
printer.plot_triple_deltas(scheme)

print("="*40)
print("TEST 3: Train Using Infinity Norm (LP), should be good")
factory.set_scheme(scheme, 'random', n=3, d=3, m=3, p=27)
manipulator.set_norm(scheme, norm=torch.inf, field='R')
printer.report(scheme, verbose=1)
trainer.train(scheme, epochs=200, batch_size=10, lr=1e-2, penalty=1e-5, use_L2=False, verbose=1)
printer.plot_triple_deltas(scheme)

print("="*40)
print("TEST 4: Test Plot of Errors During Training")
factory.set_scheme(scheme, 'random', n=3, d=3, m=3, p=22)
trainer.train(scheme, epochs=500, batch_size=1, lr=1e-2, use_L2=False, penalty=1e-5, verbose=2)
if printer.error(scheme) < -1.0:
    printer.report(scheme, verbose=4)