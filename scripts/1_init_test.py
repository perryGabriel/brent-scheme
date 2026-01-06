from brentscheme.BrentScheme import BrentScheme

print("="*40)
print("Initializing a test scheme...")
print("Output should be: 'A scheme for (3 x 2) @ (2 x 4) using 8 products: complexity is n^1.963'")
scheme = BrentScheme(n=3, d=2, m=4, p=8, verbose=1)