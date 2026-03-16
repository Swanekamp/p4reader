"""
Example: read_structure.py

Load and inspect a Chicago struct.p4 file.
"""

from p4reader import P4Structure

# Load the structure file
STRUCT = P4Structure("struct.p4")

print("Title:", STRUCT.title)
print("Geometry type:", STRUCT.geom)
print("Dimension:", STRUCT.dimen)

# Number of boundary segments
print("Total boundary segments:", len(STRUCT.xa))

# Example: list first few segments
print("\nFirst 5 segments:")
for i in range(5):
    print(
        f"({STRUCT.xa[i]:.3f}, {STRUCT.za[i]:.3f}) -> "
        f"({STRUCT.xb[i]:.3f}, {STRUCT.zb[i]:.3f})"
    )

# Extract R-Z slice segments
lines = STRUCT.get_rz_segments()

print("\nSegments in R-Z plane:", len(lines))

# Detect connected conductor bodies
bodies = STRUCT.find_connected_bodies()

print("\nNumber of connected conductor bodies:", len(bodies))

for i, body in enumerate(bodies[:5]):
    print(f"Body {i}: {len(body)} segments")
