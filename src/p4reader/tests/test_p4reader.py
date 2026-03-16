#!/usr/bin/env python3
"""
Test script for p4reader classes

Tests:
    P4Reader      (fldsXXXX.p4)
    P4Particles   (partXXXX.p4)
    P4Structure   (struct.p4)
"""

import os
from pathlib import Path
import numpy as np

from p4reader import P4Reader, P4Particles, P4Structure

os.chdir('/mnt/c/Users/sswan/OneDrive/Documents/Runs/Chicago/Fisica/run38')
RUN_DIR = Path(".")


# ------------------------------------------------------------
# Test field reader
# ------------------------------------------------------------

def test_fields():

    files = sorted(RUN_DIR.glob("flds*.p4"))

    if not files:
        print("No field files found")
        return

    fname = files[0]

    print("\n==============================")
    print("Testing P4Reader")
    print("==============================")

    FLDS = P4Reader(str(fname))

    print("file:", fname)
    print("title:", FLDS.title)
    print("time:", FLDS.time, "ns")
    print("geometry:", FLDS.geom)

    print("grid shape:")
    print("   r:", FLDS.r.shape)
    print("   z:", FLDS.z.shape)

    for name in FLDS.names:
        data = getattr(FLDS, name)
        print(f"{name:6s} min={np.min(data):.3g} max={np.max(data):.3g}")

    print("Field reader OK")


# ------------------------------------------------------------
# Test particle reader
# ------------------------------------------------------------

def test_particles():

    files = sorted(RUN_DIR.glob("part*.p4"))

    if not files:
        print("\nNo particle files found")
        return

    fname = files[0]

    print("\n==============================")
    print("Testing P4Particles")
    print("==============================")

    PART = P4Particles(str(fname))

    print("file:", fname)
    print("title:", PART.title)
    print("time:", PART.time)

    print("particle count:", len(PART.q))

    for name in PART.names:
        data = getattr(PART, name)
        print(f"{name:6s} min={np.min(data):.3g} max={np.max(data):.3g}")

    print("Particle reader OK")


# ------------------------------------------------------------
# Test structure reader
# ------------------------------------------------------------

def test_structure():

    fname = RUN_DIR / "struct.p4"

    if not fname.exists():
        print("\nNo struct.p4 found")
        return

    print("\n==============================")
    print("Testing P4Structure")
    print("==============================")

    STRUCT = P4Structure(str(fname))

    print("file:", fname)
    print("title:", STRUCT.title)

    print("Attributes in STRUCT:")
    print(dir(STRUCT))

    print("material ids:", np.unique(STRUCT.mid))
    print("types:", np.unique(STRUCT.mty))

    # ---- test body detection ----

    bodies = STRUCT.find_connected_bodies()

    print("connected bodies:", len(bodies))

    for i, body in enumerate(bodies):
        print(f"body {i} segments:", len(body))

    print("Structure reader OK")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":

    print("\nRunning p4reader tests\n")

    test_fields()
    test_particles()
    test_structure()

    print("\nAll tests completed\n")