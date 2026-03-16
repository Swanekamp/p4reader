"""
Example: inspect_run.py

Inspect a Chicago run directory and list available vector and scalar
quantities in the flds*.p4 and sclr*.p4 files.
"""

from p4reader.inspect import inspect_p4_files, print_available_p4_list

run_dir = "."

# Quick printed summary
print_available_p4_list(run_dir)

# Programmatic access
fields = inspect_p4_files(run_dir)

print("\nVector quantities:")
for name in fields["vectors"]:
    print("  ", name)

print("\nScalar quantities:")
for name in fields["scalars"]:
    print("  ", name)
