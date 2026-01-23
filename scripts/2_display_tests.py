from brentscheme.BrentScheme import BrentScheme
from brentscheme.utils.io import delete_file
from brentscheme.SchemeDisplay import SchemeDisplay
import numpy as np


scheme = BrentScheme(n=3, d=2, m=4, p=8, verbose=0)
printer = SchemeDisplay()

print("="*40)
print("Displaying a test scheme triple delta...")
printer.plot_triple_deltas(scheme)

print("="*40)
print("Displaying a test scheme in three verbosity levels...")
printer.report(scheme)

print("="*40)
printer.report(scheme, verbose=1)

print("="*40)
printer.report(scheme, verbose=2)

print("="*40)
print("Printing just the log10(L2 error)...")
print(printer.error(scheme))

print("="*40)
print("Printing all errors plus same triple delta fig...")
printer.metrics(scheme)

print("="*40)
print("Saving and deleting a test scheme...")
# Save the scheme to files
score = printer.dump_tensors(scheme)
print(f"Score: {score}")
# Delete a scheme from files. Remember to specify which scheme you want!
delete_file(n=scheme.n, d=scheme.d, m=scheme.m, p=scheme.p, number=np.round(score,3), scheme_or_diagram='scheme')
