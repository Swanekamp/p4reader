# p4reader

`p4reader` is a lightweight Python library for reading and analyzing **Chicago / LSP-style `.p4` simulation output files**.

The library is designed to make it easy to load field, particle, and structure data from pulsed-power and plasma simulations and use them in Python for analysis and visualization.

It is commonly used together with the **[simview](https://github.com/swanekamp/simview)** visualization toolkit
but can also be used independently for data inspection and analysis.

---

# Features

- Read Chicago / LSP `.p4` output files
- Access field and current density arrays
- Read simulation structure files
- Support for time-history diagnostics
- Simple Python interface for analysis and plotting

Typical use cases include:

- inspecting electromagnetic fields
- analyzing current density distributions
- computing derived quantities (current enclosed, vector magnitudes, etc.)
- post-processing simulation diagnostics

---
## Typical Workflow
```
Chicago / LSP simulation
        │
        ▼
     .p4 files
  (simulation data)
        │
        ▼
     p4reader
  (data reader)
        │
        ▼
   NumPy arrays
        │
        ▼
     simview
 (visualization)
        │
        ▼
plots / animations / diagnostics
```
Visualization toolkit: 
- simview → https://github.com/swanekamp/simview
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
```
Install the package and make it editable:
```bash
pip install -e .
```

# Example
Example script:

```python
# examples/read_fields.py

from p4reader import P4Fields

FLDS = P4Fields("flds0001.p4")
SCLR = P4Fields("sclr0001.p4")
print("time:", FLDS.time)
print("grid:", FLDS.r.shape, FLDS.z.shape)
print("Jz shape:", FLDS.Jz.shape)
print("Ez max:", FLDS.Ez.max())
print("rho shape:", SCLR.rho.shape)
```
## Other Examples

- `examples/read_fields.py` – read field and scalar data  
- `examples/inspect_p4.py` – inspect files for available fields and scalars  
- `examples/read_structure.py` – read structure files and create piecewise line representations of conductors  
- `examples/simple_plot.py` – create a simple plot

## Project Ecosystem

These libraries are designed to work together for analyzing and visualizing
pulsed-power and plasma simulations. They are commonly used with
the Chicago / LSP particle-in-cell codes but can be adapted to other
simulation frameworks.

| Project | Description |
|-------|-------------|
| **p4reader** | Read Chicago / LSP `.p4` simulation output into NumPy arrays |
| **simview** | Visualization utilities for simulation data |