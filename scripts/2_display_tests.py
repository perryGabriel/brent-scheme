from brentscheme.BrentScheme import BrentScheme
from brentscheme.misc import delete_file
from brentscheme.SchemaFactory import SchemaFactory
from brentscheme.SchemeDisplay import SchemeDisplay
import numpy as np


scheme = BrentScheme(n=3, d=2, m=4, p=8, verbose=0)
printer = SchemeDisplay()

print("="*40)
print("Displaying a test scheme triple delta...")
printer.print_triple_deltas(scheme)

print("="*40)
print("Displaying a test scheme in three verbosity levels...")
printer.print(scheme)

print("="*40)
printer.print(scheme, verbose=1)

print("="*40)
printer.print(scheme, verbose=2)

print("="*40)
print("Printing just the log10(L2 error)...")
print(printer.test(scheme, verbose=0))

print("="*40)
print("Printing all errors plus same triple delta fig...")
printer.test(scheme, verbose=4)

print("="*40)
print("Saving and deleting a test scheme...")
# Save the scheme to files
score = printer.dump_to_file(scheme)
print(f"Score: {score}")
# Delete a scheme from files. Remember to specify which scheme you want!
delete_file(n=scheme.n, d=scheme.d, m=scheme.m, p=scheme.p, number=np.round(score,3), scheme_or_diagram='scheme')
