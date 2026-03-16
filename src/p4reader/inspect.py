from pathlib import Path
from .p4reader import P4Reader

def inspect_p4_files(run_dir):
    """
    Inspect the first flds*.p4 and sclr*.p4 files in a run directory
    and report which vector components and scalar fields are available.

    Returns
    -------
    dict
        {
            "vectors": [list of vector components],
            "scalars": [list of scalar fields]
        }
    """

    run_dir = Path(run_dir)

    flds_file = sorted(run_dir.glob("flds*.p4"))
    sclr_file = sorted(run_dir.glob("sclr*.p4"))

    vectors = []
    scalars = []

    # --- inspect vector file ---
    if flds_file:
        flds = P4Reader(flds_file[0])

        for name, value in vars(flds).items():

            if name.startswith("_"):
                continue

            if name in ("r", "z", "x", "time", "time_ns"):
                continue

            if hasattr(value, "shape"):
                vectors.append(name)

    # --- inspect scalar file ---
    if sclr_file:
        sclr = P4Reader(sclr_file[0])

        for name, value in vars(sclr).items():

            if name.startswith("_"):
                continue

            if name in ("r", "z", "x", "time", "time_ns"):
                continue

            if hasattr(value, "shape"):
                scalars.append(name)

    return {
        "vectors": sorted(vectors),
        "scalars": sorted(scalars),
    }

def print_available_p4_list(run_dir):
    run_dir = Path(run_dir)

    fields = inspect_p4_files(run_dir)

    print("\nVector components in flds*.p4:")
    for v in fields["vectors"]:
        print("   ", v)

    print("\nScalar fields in sclr*.p4:")
    for s in fields["scalars"]:
        print("   ", s)

