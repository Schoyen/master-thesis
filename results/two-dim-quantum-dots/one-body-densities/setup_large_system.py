import sys
import numpy as np
from tdho_system import cache_large_system


if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <l>")
    sys.exit()

l = int(sys.argv[1])
shells = list((np.cumsum(np.arange(20)) * 2).astype(int))

if l not in shells:
    print(f"l = {l} not in {shells}")
    sys.exit()

print(f"Starting caching for l = {l}")
cache_large_system(n=2, l=l, radius_length=10, num_grid_points=201)
