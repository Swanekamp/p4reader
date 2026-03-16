# p4reader

`p4reader` is a lightweight Python library for reading and analyzing **Chicago / LSP-style `.p4` simulation output files**.

The library is designed to make it easy to load field, particle, and structure data from pulsed-power and plasma simulations and use them in Python for analysis and visualization.

It is primarily used together with the **simview** or other visualization tools but can also be used independently.

---

# Features

- Read Chicago / LSP `.p4` output files
- Access field and current density arrays
- Read simulation structure files
- Support for time-history diagnostics
- Simple Python interface for analysis and plotting

Typical uses include:

- inspecting electromagnetic fields
- analyzing current density distributions
- computing derived quantities (current enclosed, vector magnitudes, etc.)
- post-processing simulation diagnostics

---

# Requirements

p4reader has minimal dependencies.

Required:

- Python 3.9+
- numpy

# Installation

Clone the repository:

```bash
git clone https://github.com/swanekamp/p4reader.git
cd p4reader

# Example

```python
from p4reader import P4Fields

fld = P4Fields("flds0001.p4")

print("time =", fld.time)
print("Ez shape =", fld.Ez.shape)
print("Jz shape =", fld.Jz.shape)