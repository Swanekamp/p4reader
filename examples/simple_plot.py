"""
Minimal example showing how to plot a field from a p4 file.
"""

import matplotlib.pyplot as plt
from p4reader import P4Reader

FLDS = P4Reader("flds0001.p4")

plt.pcolormesh(FLDS.r, FLDS.z, FLDS.Ez)
plt.xlabel("r")
plt.ylabel("z")
plt.title(f"Ez at time = {FLDS.time}")
plt.colorbar(label="Ez")

plt.show()
