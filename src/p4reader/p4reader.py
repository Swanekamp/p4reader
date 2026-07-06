from cProfile import label
import struct
import numpy as np
import struct
import re
from pathlib import Path

class P4History:
    """
    Reader for Chicago ASCII history/probe dumps (e.g. history.p4).

    Expected format:
      #Chicago simulation: ...
      #revision
      #File type: probes
      #Number of data items: N
      #0: time: ns
      #1: Probe name ... : amperes
      ...
      i  val0 val1 ... valN-1

    Notes
    -----
    - First numeric column is treated as a record/sample index.
    - Remaining columns correspond to the N declared history items.
    - Both raw labels and sanitized attribute names are retained.
    """

    def __init__(self, fname):
        self.fname = str(fname)
        self.path = Path(fname)

        self.header_lines = []
        self.file_type = None
        self.nitems = None
        self.item_defs = []      # list of dicts with idx, label, unit, attr
        self.labels = []         # raw labels in file order
        self.units = []          # units in file order
        self.names = []          # sanitized names in file order
        self.short_labels = []   # short labels for plotting (derived from raw labels)
        self.short_name_map = {} # mapping from raw label to short label

        self._read()

    def _read(self):
        lines = self.path.read_text(errors="replace").splitlines()

        data_start = None
        for i, line in enumerate(lines):
            if not line:
                continue
            if line.startswith("#"):
                self.header_lines.append(line)
            else:
                data_start = i
                break

        if data_start is None:
            raise ValueError(f"No numeric data found in {self.fname}")

        self._parse_header(self.header_lines)

        # Load numeric table
        arr = np.loadtxt(lines[data_start:])
        if arr.ndim == 1:
            arr = arr[None, :]

        if arr.shape[1] < 2:
            raise ValueError("History file must contain index + at least one data column")

        self.index = arr[:, 0].astype(int)
        self.data = arr[:, 1:]

        if self.nitems is not None and self.data.shape[1] != self.nitems:
            raise ValueError(
                f"Header says {self.nitems} history items but table has {self.data.shape[1]} data columns"
            )

        self.nrows = self.data.shape[0]
        self.ncols = self.data.shape[1]

        # Expose each trace as an attribute
        for j, name in enumerate(self.names):
            setattr(self, name, self.data[:, j])

        # Common convenience alias
        if "time" in self.names:
            self.time = getattr(self, "time")

    def _parse_header(self, header_lines):
        item_pattern = re.compile(r"^#\s*(\d+):\s*(.*)$")

        used_names = set()

        self.item_defs = []
        self.labels = []
        self.short_labels = []
        self.units = []
        self.names = []

        for line in header_lines:
            if line.startswith("#File type:"):
                self.file_type = line.split(":", 1)[1].strip()
                continue

            if line.startswith("#Number of data items:"):
                self.nitems = int(line.split(":", 1)[1].strip())
                continue

            m = item_pattern.match(line)
            if not m:
                continue

            idx = int(m.group(1))
            rest = m.group(2).strip()

            # Split at the LAST colon so labels like "... potential 2: amperes"
            # become label="... potential 2", unit="amperes"
            if ":" in rest:
                label, unit = rest.rsplit(":", 1)
                label = label.strip()
                unit = unit.strip()
            else:
                label = rest
                unit = ""

            short_label = self._shorten_chicago_label(label)
            attr = self._sanitize_name(label)

            # Ensure uniqueness of Python attribute names
            base = attr
            k = 2
            while attr in used_names:
                attr = f"{base}_{k}"
                k += 1
            used_names.add(attr)

            self.item_defs.append({
                "idx": idx,
                "label": label,
                "short_label": short_label,
                "unit": unit,
                "attr": attr,
            })

        # Sort in case header lines were out of order
        self.item_defs.sort(key=lambda d: d["idx"])

        self.labels = [d["label"] for d in self.item_defs]
        self.short_labels = [d["short_label"] for d in self.item_defs]
        self.units = [d["unit"] for d in self.item_defs]
        self.names = [d["attr"] for d in self.item_defs]

    @staticmethod
    def _sanitize_name(label):
        name = label.strip().lower()

        # Friendly replacements
        name = name.replace("%", "pct")
        name = name.replace("/", "_per_")
        name = name.replace("-", "_")
        name = name.replace("(", "_")
        name = name.replace(")", "_")

        # Collapse everything else non-alnum to underscore
        name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")

        if not name:
            name = "trace"

        if name[0].isdigit():
            name = "v_" + name

        return name

    def __len__(self):
        return self.nrows

    def __getitem__(self, key):
        j = self._col_index(key)
        return self.data[:, j]
    
    def keys(self):
        return list(self.names)

    def raw_labels(self):
        return list(self.labels)

    def get_unit(self, key):
        j = self._col_index(key)
        return self.units[j]

    def get_label(self, key):
        j = self._col_index(key)
        return self.labels[j]

    def find(self, text):
        text = text.lower()
        return [
            (j, self.names[j], self.short_labels[j], self.labels[j], self.units[j])
            for j in range(self.ncols)
            if text in self.names[j].lower()
            or text in self.labels[j].lower()
            or text in self.short_labels[j].lower()
        ]

    def summary(self):
        lines = [
            f"P4History('{self.fname}')",
            f"  file_type = {self.file_type}",
            f"  nrows     = {self.nrows}",
            f"  ncols     = {self.ncols}",
            "  traces:"
        ]
        for j, (name, label, unit) in enumerate(zip(self.names, self.labels, self.units)):
            lines.append(f"    [{j:02d}] {name}    ({unit})")
            lines.append(f"         {label}")
        return "\n".join(lines)
    
    @staticmethod
    def _shorten_chicago_label(label):
        """
        Convert a full Chicago history label into a shorter display/lookup label.

        Examples
        --------
        "ITarget_Top_OUTERRING (1.48e+01 0.00e+00 4.90e+00)-(1.52e+01 6.28e+00 4.90e+00), potential 2"
            -> "itarget_top_outerring"

        "EB_IN at z = 0.00e+00-2.00e-01, species 1"
            -> "eb_in"

        "global number, species 1"
            -> "global_number"
        """
        short = label.strip()

        # Drop everything starting with a coordinate block
        if "(" in short:
            short = short.split("(", 1)[0].strip()

        # Drop " at z = ..." style suffixes
        short = re.split(r"\s+at\s+", short, maxsplit=1, flags=re.IGNORECASE)[0].strip()

        # Drop trailing comma metadata like ", potential 2" or ", species 1"
        short = short.split(",", 1)[0].strip()

        # Normalize into a Python-friendly short key
        short = short.lower()
        short = short.replace("%", "pct")
        short = short.replace("/", "_per_")
        short = short.replace("-", "_")
        short = short.replace("(", "_").replace(")", "_")
        short = re.sub(r"[^0-9a-zA-Z_]+", "_", short)
        short = re.sub(r"_+", "_", short).strip("_")

        return short if short else label.strip().lower()

    def _col_index(self, key):
        if isinstance(key, int):
            return key
        if key in self.names:
            return self.names.index(key)
        if key in self.labels:
            return self.labels.index(key)
        if key in self.short_labels:
            matches = [j for j, s in enumerate(self.short_labels) if s == key]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise KeyError(f"Short label '{key}' is ambiguous; matches columns {matches}")
        raise KeyError(key)

    def get_short_label(self, key):
        j = self._col_index(key)
        return self.short_labels[j]    
  
class P4Reader:

    def __init__(self, fname):

        self.fname = fname

        with open(fname, "rb") as f:

            # ----------------------------
            # HEADER
            # ----------------------------
            self.type  = self._read_int(f)
            self.dver  = self._read_int(f)
            self.title = self._read_counted_string(f)
            self.revision = self._read_counted_string(f)
            self.time  = self._read_float(f)
            self.geom  = self._read_int(f)
            self.nprocs = self._read_int(f)
            self.nquant = self._read_int(f)

            self.names = [
                self._read_counted_string(f)
                for _ in range(self.nquant)
            ]

            self.units = [
                self._read_counted_string(f)
                for _ in range(self.nquant)
            ]

            # Decide mode
            if self.type == 2:
                self.mode = "vector"
            else:
                self.mode = "scalar"

            # ----------------------------
            # BLOCKS
            # ----------------------------
            self.blocks = []
            for _ in range(self.nprocs):
                block = self._read_block(f)
                self.blocks.append(block)

        # ----------------------------
        # Assemble global
        # ----------------------------
        self._assemble_global()

    # ==========================================================
    # Low-level XDR readers
    # ==========================================================
    def _read_int(self, f):
        return struct.unpack(">i", f.read(4))[0]

    def _read_float(self, f):
        return struct.unpack(">f", f.read(4))[0]

    def _read_floats(self, f, n):
        return struct.unpack(f">{n}f", f.read(4*n))

    def _read_counted_string(self, f):
        maxlen = self._read_int(f)
        length = self._read_int(f)

        if length <= 0:
            return ""

        raw = f.read(length)
        pad = (4 - (length % 4)) % 4
        if pad:
            f.read(pad)

        return raw.decode(errors="ignore")

    # ==========================================================
    # Read one MPI block
    # ==========================================================
    def _read_block(self, f):

        block = {}

        block["iR"] = self._read_int(f)
        block["jR"] = self._read_int(f)
        block["kR"] = self._read_int(f)

        # ---- nI + x grid ----
        nI = self._read_int(f)
        x = np.array(self._read_floats(f, nI))

        # ---- detect nJ ----
        pos = f.tell()
        maybe = self._read_int(f)

        if maybe < 10:
            nJ = maybe
            y = np.array(self._read_floats(f, nJ))
            nK = self._read_int(f)
            z = np.array(self._read_floats(f, nK))
        else:
            f.seek(pos)
            nJ = 1
            y = np.array([0.0])
            nK = self._read_int(f)
            z = np.array(self._read_floats(f, nK))

        block["nI"] = nI
        block["nJ"] = nJ
        block["nK"] = nK
        block["x"] = x
        block["z"] = z

        nnodes = nI * nJ * nK

        # ---- read quantities ----
        for name in self.names:

            if self.mode == "vector":
                data = np.array(self._read_floats(f, nnodes * 3))
                block[name] = data
            else:
                data = np.array(self._read_floats(f, nnodes))
                block[name] = data

        return block

    # ==========================================================
    # Assemble global arrays
    # ==========================================================
    def _assemble_global(self):
        """
        Assemble MPI/domain blocks into global arrays.

        This version uses the physical x/z coordinates to place blocks,
        rather than trusting iR/kR offsets. That is important for
        multi-region Chicago runs, where kR may restart in each z-region.
        """

        # Build global coordinate arrays from all blocks
        r_round = 12
        z_round = 12

        all_r = np.concatenate([np.round(b["x"], r_round) for b in self.blocks])
        all_z = np.concatenate([np.round(b["z"], z_round) for b in self.blocks])

        self.r = np.unique(all_r)
        self.z = np.unique(all_z)

        self.nr = len(self.r)
        self.nz = len(self.z)

        if self.mode == "vector":
            global_data = {
                name: np.zeros((self.nz, self.nr, 3))
                for name in self.names
            }
        else:
            global_data = {
                name: np.zeros((self.nz, self.nr))
                for name in self.names
            }

        # Assemble by coordinate matching
        for b in self.blocks:

            bx = np.round(b["x"], r_round)
            bz = np.round(b["z"], z_round)

            ir = np.searchsorted(self.r, bx)
            iz = np.searchsorted(self.z, bz)

            for name in self.names:

                if self.mode == "vector":
                    local = b[name].reshape((b["nK"], b["nI"], 3))
                    global_data[name][np.ix_(iz, ir)] = local
                else:
                    local = b[name].reshape((b["nK"], b["nI"]))
                    global_data[name][np.ix_(iz, ir)] = local

        # Attach to object
        for name in self.names:
            full = global_data[name]
            setattr(self, name, full)

            if self.mode == "vector":
                setattr(self, name + "r", full[:, :, 0])
                setattr(self, name + "y", full[:, :, 1])
                setattr(self, name + "z", full[:, :, 2])
                
    def get_unit(obj, quantity):
        if quantity in obj.names:
            return obj.units[obj.names.index(quantity)]

        if quantity.endswith(("r", "y", "z")):
            base = quantity[:-1]
            if base in obj.names:
                return obj.units[obj.names.index(base)]

        raise KeyError(f"No unit found for {quantity}")

class P4Particles:
    """
    Reader for Chicago particle-list dumps (e.g., part*.p4).
    Format (based on part10763.p4):
      int type (=1)
      int dver (=2)
      counted_string title  (XDR length written twice)
      counted_string revision
      float time
      int geom
      int nprocs
      int nqty (=0)
      int a
      int b
      int nparticles
      int nvar (=7)
      nvar counted_string units
      then nparticles records of:
        int species
        nvar floats
    """

    def __init__(self, fname: str):
        self.fname = fname
        self.mode = "particle"

        with open(fname, "rb") as f:
            self.type = self._read_i32(f)
            self.dver = self._read_i32(f)

            self.title = self._read_counted_string(f)
            self.revision = self._read_counted_string(f)

            self.time = self._read_f32(f)
            self.geom = self._read_i32(f)
            self.nprocs = self._read_i32(f)
            self.nqty = self._read_i32(f)

            # Particle header
            self.a = self._read_i32(f)
            self.b = self._read_i32(f)
            self.np = self._read_i32(f)
            self.nvar = self._read_i32(f)

            float_units = [self._read_counted_string(f) for _ in range(self.nvar)]
            self.units = [None] + float_units
            # Reasonable default names based on your units pattern
            # (None, microcoulombs, cm, cm, cm, beta-gamma, beta-gamma, beta-gamma)
            float_names = ["q", "x", "y", "z", "bgx", "bgy", "bgz"][:self.nvar]
            self.names = ["species"] + float_names # species is int, rest are float

            # Read particle records efficiently
            rec_dtype = np.dtype([
                ("species", ">i4"),
                ("vals", (">f4", self.nvar)),
            ])
            arr = np.fromfile(f, dtype=rec_dtype, count=self.np)

        # Promote each column to an attribute: self.q, self.x, ...
        self.species = arr["species"].astype(np.int32) 
        vals = arr["vals"].astype(np.float32)
        for i, name in enumerate(float_names):
            setattr(self, name, vals[:, i])

        # Keep the full matrix too
        self.data = vals

    @staticmethod
    def _read_i32(f) -> int:
        b = f.read(4)
        if len(b) != 4:
            raise EOFError("Unexpected EOF while reading int32")
        return struct.unpack(">i", b)[0]

    @staticmethod
    def _read_f32(f) -> float:
        b = f.read(4)
        if len(b) != 4:
            raise EOFError("Unexpected EOF while reading float32")
        return struct.unpack(">f", b)[0]

    @staticmethod
    def _read_counted_string(f, maxlen=100_000) -> str:
        # Chicago writes length twice (counted_string calls xdr_string)
        l1 = struct.unpack(">I", f.read(4))[0]
        l2 = struct.unpack(">I", f.read(4))[0]
        if l2 > maxlen:
            raise ValueError(f"Unreasonable string length {l2}")
        s = f.read(l2)
        pad = (4 - (l2 % 4)) % 4
        if pad:
            f.read(pad)
        return s.decode("utf-8", errors="replace")

    def __getitem__(self, key: str):
            if hasattr(self, key):
                return getattr(self, key)
            raise KeyError(key)
    
class P4Structure:
    """
    Reader for Chicago struct.p4 structural boundary dumps.

    Parameters
    ----------
    fname : str or Path
        Path to the struct.p4 file.
    dim : {None, 1, 2}, optional
        Force the file to be interpreted as a 1D or 2D structure dump.
        ``None`` (default) trusts the ``dimen`` field in the file header,
        falling back to 1D if no segments are present. Pass ``dim=1`` for
        1D Cartesian runs (no r-z boundary segments) or ``dim=2`` for
        axisymmetric / 2D Cartesian runs.
    """

    def __init__(self, fname, dim=None):
        if dim not in (None, 1, 2):
            raise ValueError(f"dim must be None, 1, or 2 (got {dim!r})")
        self.fname = fname
        self._force_dim = dim
        self._read()

    # --------------------------
    # XDR helpers
    # --------------------------

    def _read_int(self, f):
        b = f.read(4)
        if len(b) != 4:
            raise EOFError
        return struct.unpack(">i", b)[0]

    def _read_float(self, f):
        b = f.read(4)
        if len(b) != 4:
            raise EOFError
        return struct.unpack(">f", b)[0]

    def _read_counted_string(self, f):
        maxlen = self._read_int(f)
        length = self._read_int(f)

        if length <= 0:
            return ""

        s = f.read(length)
        pad = (-length) % 4
        if pad:
            f.read(pad)

        return s.decode(errors="replace")

    # --------------------------
    # File reader
    # --------------------------

    def _read(self):
        segments = []

        with open(self.fname, "rb") as f:
            self.title = self._read_counted_string(f)
            self.geom = self._read_int(f)
            self.dimen = self._read_int(f)

            while True:
                try:
                    nty = self._read_int(f)
                    mty = self._read_int(f)
                    nid = self._read_int(f)
                    mid = self._read_int(f)

                    xa = self._read_float(f)
                    ya = self._read_float(f)
                    za = self._read_float(f)

                    xb = self._read_float(f)
                    yb = self._read_float(f)
                    zb = self._read_float(f)

                except EOFError:
                    break

                segments.append(
                    (nty, mty, nid, mid, xa, ya, za, xb, yb, zb)
                )

        # Resolve dimensionality: explicit override wins, else the file
        # header, else infer from whether any segments were read.
        if self._force_dim is not None:
            self.dim = self._force_dim
        elif self.dimen in (1, 2):
            self.dim = self.dimen
        else:
            self.dim = 2 if segments else 1

        if self.dim == 1:
            # 1D Cartesian runs have no r-z boundary segments.
            empty_i = np.array([], dtype=int)
            empty_f = np.array([], dtype=float)
            self.nty = empty_i
            self.mty = empty_i
            self.nid = empty_i
            self.mid = empty_i
            self.xa = empty_f
            self.ya = empty_f
            self.za = empty_f
            self.xb = empty_f
            self.yb = empty_f
            self.zb = empty_f
            return

        # 2D path — reshape guards against the (0, 10) edge case.
        segments = np.array(segments, dtype=float).reshape(-1, 10)

        self.nty = segments[:, 0].astype(int)
        self.mty = segments[:, 1].astype(int)
        self.nid = segments[:, 2].astype(int)
        self.mid = segments[:, 3].astype(int)

        self.xa = segments[:, 4]
        self.ya = segments[:, 5]
        self.za = segments[:, 6]

        self.xb = segments[:, 7]
        self.yb = segments[:, 8]
        self.zb = segments[:, 9]

    # --------------------------
    # Convenience filters
    # --------------------------

    def conductor_mask(self):
        """
        Returns mask of boundary segments where either side is conductor.
        """
        return (self.nty == 1) | (self.mty == 1)

    def get_rz_segments(self, theta=0.0, tol=1e-6):
        """
        Return (N,2,2) array of line segments in (r,z) plane
        at specified theta slice.
        """
        in_plane = (
            np.abs(self.ya - theta) < tol
        ) & (
            np.abs(self.yb - theta) < tol
        )

        mask = self.conductor_mask() & in_plane

        lines = np.stack(
            [
                np.stack([self.xa[mask], self.za[mask]], axis=1),
                np.stack([self.xb[mask], self.zb[mask]], axis=1),
            ],
            axis=1,
        )

        return lines
    
    def find_connected_bodies(self, tol=1e-6):
        """
        Returns list of lists of segment indices.
        Each list corresponds to one connected structure.
        """

        # Only conductor boundaries
        mask = (self.mty == 1) | (self.nty == 1)

        indices = np.where(mask)[0]

        # Build endpoint mapping
        points = {}

        def key(x, z):
            return (round(x / tol), round(z / tol))

        for idx in indices:
            p1 = key(self.xa[idx], self.za[idx])
            p2 = key(self.xb[idx], self.zb[idx])

            points.setdefault(p1, []).append(idx)
            points.setdefault(p2, []).append(idx)

        visited = set()
        bodies = []

        for idx in indices:
            if idx in visited:
                continue

            stack = [idx]
            body = []

            while stack:
                s = stack.pop()
                if s in visited:
                    continue

                visited.add(s)
                body.append(s)

                p1 = key(self.xa[s], self.za[s])
                p2 = key(self.xb[s], self.zb[s])

                neighbors = points[p1] + points[p2]

                for n in neighbors:
                    if n not in visited:
                        stack.append(n)

            bodies.append(body)

        return bodies
    
class P4Target:
    """
    Reader for Chicago target.p4 dump files.
    Format (based on target*.p4 files):"""
    def __init__(self, fname):
        self.fname = fname
        self._read()

import struct
import re
from pathlib import Path
import numpy as np


import struct
import re
from pathlib import Path
import numpy as np


import struct
import re
from pathlib import Path
import numpy as np


class P4ParticleDiagnostic:
    """
    Reader for Chicago particle diagnostic dumps (diag*.p4).

    Inferred layout from diag53814.p4:
      int   type          (= 4)
      int   dver
      str   title         (Chicago counted string)
      str   revision      (Chicago counted string)
      float time
      int   geom

      Then 4 descriptor records, each containing 5 counted strings:
          tag
          coord_name
          coord_unit
          label
          unit

      Then one data block per descriptor:
          int   npts
          float coord[npts]
          float values[npts]

    Notes
    -----
    - This version is based on the current sample file structure.
    - The descriptor count is currently hard-wired to 4.
    """

    def __init__(self, fname):
        self.fname = str(fname)
        self.path = Path(fname)
        self.mode = "particle_diagnostic"

        self.type = None
        self.dver = None
        self.title = ""
        self.revision = ""
        self.time = None
        self.geom = None

        self.labels = []
        self.short_labels = []
        self.names = []
        self.units = []

        self.coord = None
        self.coord_name = None
        self.coord_unit = None

        self.data = {}       # sanitized_name -> 1D array
        self.raw_data = {}   # raw label -> 1D array
        self.meta = []       # per-trace metadata dicts

        self._read()

    # ==========================================================
    # Low-level readers
    # ==========================================================

    @staticmethod
    def _read_i32(f):
        b = f.read(4)
        if len(b) != 4:
            raise EOFError("Unexpected EOF while reading int32")
        return struct.unpack(">i", b)[0]

    @staticmethod
    def _read_f32(f):
        b = f.read(4)
        if len(b) != 4:
            raise EOFError("Unexpected EOF while reading float32")
        return struct.unpack(">f", b)[0]

    @staticmethod
    def _read_f32_array(f, n):
        if n < 0:
            raise ValueError(f"Negative array length: {n}")
        b = f.read(4 * n)
        if len(b) != 4 * n:
            raise EOFError(f"Unexpected EOF while reading {n} float32 values")
        return np.asarray(struct.unpack(f">{n}f", b), dtype=np.float32)

    @staticmethod
    def _read_counted_string(f, maxlen=100_000):
        """
        Chicago/XDR counted string format:
            int maxlen
            int length
            bytes
            pad to 4-byte boundary
        """
        max_declared = P4ParticleDiagnostic._read_i32(f)
        length = P4ParticleDiagnostic._read_i32(f)

        if length < 0 or length > maxlen:
            raise ValueError(f"Unreasonable counted string length {length}")

        raw = f.read(length)
        if len(raw) != length:
            raise EOFError("Unexpected EOF while reading counted string")

        pad = (4 - (length % 4)) % 4
        if pad:
            f.read(pad)

        return raw.decode("utf-8", errors="replace")

    # ==========================================================
    # Label helpers
    # ==========================================================

    @staticmethod
    def _sanitize_name(label):
        name = label.strip().lower()
        name = name.replace("%", "pct")
        name = name.replace("/", "_per_")
        name = name.replace("-", "_")
        name = name.replace("(", "_")
        name = name.replace(")", "_")
        name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")

        if not name:
            name = "trace"

        if name[0].isdigit():
            name = "v_" + name

        return name

    @staticmethod
    def _shorten_chicago_label(label):
        short = label.strip()

        if "(" in short:
            short = short.split("(", 1)[0].strip()

        short = re.split(r"\s+at\s+", short, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        short = short.split(",", 1)[0].strip()

        short = short.lower()
        short = short.replace("%", "pct")
        short = short.replace("/", "_per_")
        short = short.replace("-", "_")
        short = short.replace("(", "_").replace(")", "_")
        short = re.sub(r"[^0-9a-zA-Z_]+", "_", short)
        short = re.sub(r"_+", "_", short).strip("_")

        return short if short else label.strip().lower()

    def _make_unique_names(self, labels):
        used = set()
        names = []

        for label in labels:
            attr = self._sanitize_name(label)
            base = attr
            k = 2
            while attr in used:
                attr = f"{base}_{k}"
                k += 1
            used.add(attr)
            names.append(attr)

        return names

    # ==========================================================
    # Main read
    # ==========================================================

    def _read(self):
        with open(self.fname, "rb") as f:
            # ----------------------------
            # Common header
            # ----------------------------
            self.type = self._read_i32(f)
            self.dver = self._read_i32(f)
            self.title = self._read_counted_string(f)
            self.revision = self._read_counted_string(f)
            self.time = self._read_f32(f)
            self.geom = self._read_i32(f)

            if self.type != 4:
                raise ValueError(
                    f"{self.fname} is not a particle diagnostic file "
                    f"(type={self.type}, expected 4)"
                )

            # ----------------------------
            # Descriptor section
            # ----------------------------
            descriptors = []

            for j in range(4):
                pos = f.tell()
                try:
                    tag = self._read_counted_string(f)
                    coord_name = self._read_counted_string(f)
                    coord_unit = self._read_counted_string(f)
                    label = self._read_counted_string(f)
                    unit = self._read_counted_string(f)
                except Exception as e:
                    raise ValueError(
                        f"Failed while parsing descriptor {j} of 4 in {self.fname} "
                        f"at byte offset {pos}"
                    ) from e

                descriptors.append(
                    {
                        "tag": tag,
                        "coord_name": coord_name,
                        "coord_unit": coord_unit,
                        "label": label,
                        "unit": unit,
                    }
                )

            if not descriptors:
                raise ValueError(f"No diagnostic descriptors found in {self.fname}")

            self.meta = descriptors

            # ----------------------------
            # Data section
            # ----------------------------
            self.labels = [d["label"] for d in descriptors]
            self.units = [d["unit"] for d in descriptors]
            self.short_labels = [self._shorten_chicago_label(lbl) for lbl in self.labels]
            self.names = self._make_unique_names(self.labels)

            common_coord = None
            self.coord_name = descriptors[0]["coord_name"]
            self.coord_unit = descriptors[0]["coord_unit"]

            for j, d in enumerate(descriptors):
                npts = self._read_i32(f)
                x = self._read_f32_array(f, npts)
                y = self._read_f32_array(f, npts)

                if common_coord is None:
                    common_coord = x
                else:
                    if len(x) != len(common_coord) or not np.allclose(x, common_coord):
                        d["coord_mismatch"] = True

                raw_label = d["label"]
                clean_name = self.names[j]

                self.raw_data[raw_label] = y
                self.data[clean_name] = y

                setattr(self, clean_name, y)

            self.coord = common_coord

            if self.coord_name:
                coord_attr = self._sanitize_name(self.coord_name)
                setattr(self, coord_attr, self.coord)

    # ==========================================================
    # Convenience API
    # ==========================================================

    def __len__(self):
        return 0 if self.coord is None else len(self.coord)

    def keys(self):
        return list(self.names)

    def raw_labels(self):
        return list(self.labels)

    def _col_index(self, key):
        if isinstance(key, int):
            return key
        if key in self.names:
            return self.names.index(key)
        if key in self.labels:
            return self.labels.index(key)
        if key in self.short_labels:
            matches = [j for j, s in enumerate(self.short_labels) if s == key]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise KeyError(f"Short label '{key}' is ambiguous; matches columns {matches}")
        raise KeyError(key)

    def __getitem__(self, key):
        j = self._col_index(key)
        return self.data[self.names[j]]

    def get_unit(self, key):
        j = self._col_index(key)
        return self.units[j]

    def get_label(self, key):
        j = self._col_index(key)
        return self.labels[j]

    def get_short_label(self, key):
        j = self._col_index(key)
        return self.short_labels[j]

    def find(self, text):
        text = text.lower()
        return [
            (j, self.names[j], self.short_labels[j], self.labels[j], self.units[j])
            for j in range(len(self.names))
            if text in self.names[j].lower()
            or text in self.labels[j].lower()
            or text in self.short_labels[j].lower()
        ]

    def summary(self):
        lines = [
            f"P4ParticleDiagnostic('{self.fname}')",
            f"  type        = {self.type}",
            f"  dver        = {self.dver}",
            f"  time        = {self.time}",
            f"  geom        = {self.geom}",
            f"  coord       = {self.coord_name} ({self.coord_unit})",
            f"  ntraces     = {len(self.names)}",
            f"  npoints     = {len(self)}",
            "  traces:"
        ]
        for j, (name, label, unit) in enumerate(zip(self.names, self.labels, self.units)):
            lines.append(f"    [{j:02d}] {name}    ({unit})")
            lines.append(f"         {label}")
        return "\n".join(lines)
    
import struct
import re
from pathlib import Path
import numpy as np


class P4TargetRecord:
    def __init__(self):
        self.fields = {}
        self.units = {}
        self.names = []

    def keys(self):
        return list(self.names)

    def __getitem__(self, key):
        if key in self.fields:
            return self.fields[key]
        raise KeyError(key)


class P4ParticleTarget:
    """
    Reader for Chicago particle target dumps (targ*.p4).

    Supports files containing multiple target records.
    """

    def __init__(self, fname):
        self.fname = str(fname)
        self.path = Path(fname)
        self.mode = "particle_target"

        self.fields = {}
        self.units = {}
        self.targets = []

        self._read()

    # ----------------------------
    # Low-level readers
    # ----------------------------

    @staticmethod
    def _read_i32(f):
        b = f.read(4)
        if len(b) != 4:
            raise EOFError("Unexpected EOF while reading int32")
        return struct.unpack(">i", b)[0]

    @staticmethod
    def _read_f32(f):
        b = f.read(4)
        if len(b) != 4:
            raise EOFError("Unexpected EOF while reading float32")
        return struct.unpack(">f", b)[0]

    @staticmethod
    def _read_f32_array(f, n):
        if n < 0:
            raise ValueError(f"Negative array length: {n}")
        b = f.read(4 * n)
        if len(b) != 4 * n:
            raise EOFError(f"Unexpected EOF while reading {n} float32 values")
        return np.asarray(struct.unpack(f">{n}f", b), dtype=np.float32)

    @staticmethod
    def _read_string(f, maxlen=100_000):
        max_declared = P4ParticleTarget._read_i32(f)
        length = P4ParticleTarget._read_i32(f)

        if length < 0 or length > maxlen:
            raise ValueError(f"Unreasonable counted string length {length}")

        raw = f.read(length)
        if len(raw) != length:
            raise EOFError("Unexpected EOF while reading counted string")

        pad = (4 - (length % 4)) % 4
        if pad:
            f.read(pad)

        return raw.decode("utf-8", errors="replace")

    @staticmethod
    def _sanitize_name(label):
        name = label.strip().lower()
        name = name.replace("%", "pct")
        name = name.replace("/", "_per_")
        name = name.replace("-", "_")
        name = name.replace("(", "_").replace(")", "_")
        name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")

        if not name:
            name = "field"

        if name[0].isdigit():
            name = "v_" + name

        return name

    # ----------------------------
    # Main reader
    # ----------------------------

    def _read(self):
        with open(self.fname, "rb") as f:
            self.type = self._read_i32(f)
            self.dver = self._read_i32(f)
            self.title = self._read_string(f)
            self.revision = self._read_string(f)
            self.time = self._read_f32(f)
            self.geom = self._read_i32(f)

            if self.type != 5:
                raise ValueError(
                    f"{self.fname} is not a particle target file "
                    f"(type={self.type}, expected 5)"
                )

            self.nquant = self._read_i32(f)

            self.raw_names = [
                self._read_string(f) for _ in range(self.nquant)
            ]

            self.raw_units = [
                self._read_string(f) for _ in range(self.nquant)
            ]

            self.names = [
                self._sanitize_name(name) for name in self.raw_names
            ]

            # ----------------------------
            # Read all target records
            # ----------------------------

            self.targets = []

            while True:
                pos = f.tell()
                f.seek(0, 2)
                end = f.tell()
                f.seek(pos)

                if pos >= end:
                    break

                rec = P4TargetRecord()

                rec.target_id = self._read_i32(f)

                rec.x_name = self._read_string(f)
                rec.x_unit = self._read_string(f)

                rec.y_name = self._read_string(f)
                rec.y_unit = self._read_string(f)

                rec.nx = self._read_i32(f)
                rec.x = self._read_f32_array(f, rec.nx)

                rec.ny = self._read_i32(f)
                rec.y = self._read_f32_array(f, rec.ny)

                rec.raw_names = list(self.raw_names)
                rec.names = list(self.names)
                rec.raw_units = list(self.raw_units)

                for raw_name, name, unit in zip(
                    self.raw_names, self.names, self.raw_units
                ):
                    arr = self._read_f32_array(f, rec.nx * rec.ny)
                    arr = arr.reshape((rec.ny, rec.nx))

                    rec.fields[name] = arr
                    rec.units[name] = unit
                    setattr(rec, name, arr)

                if rec.x_name.strip().lower() in ("r", "radius"):
                    rec.r = rec.x

                if rec.y_name.strip().lower() in ("theta", "th"):
                    rec.theta = rec.y

                self.targets.append(rec)

            self.ntargets = len(self.targets)

            if self.ntargets == 0:
                raise ValueError(f"No targets found in {self.fname}")

            # ----------------------------
            # Backward-compatible aliases:
            # expose first target at top level
            # ----------------------------

            first = self.targets[0]

            self.target_id = first.target_id
            self.x_name = first.x_name
            self.x_unit = first.x_unit
            self.y_name = first.y_name
            self.y_unit = first.y_unit

            self.nx = first.nx
            self.ny = first.ny
            self.x = first.x
            self.y = first.y

            self.fields = first.fields
            self.units = first.units

            for name in self.names:
                setattr(self, name, first.fields[name])

            if hasattr(first, "r"):
                self.r = first.r

            if hasattr(first, "theta"):
                self.theta = first.theta

    # ----------------------------
    # Convenience API
    # ----------------------------

    def keys(self):
        return list(self.names)

    def raw_labels(self):
        return list(self.raw_names)

    def __getitem__(self, key):
        if key in self.fields:
            return self.fields[key]

        if key in self.raw_names:
            idx = self.raw_names.index(key)
            return self.fields[self.names[idx]]

        raise KeyError(key)

    def get_unit(self, key):
        if key in self.units:
            return self.units[key]

        if key in self.raw_names:
            idx = self.raw_names.index(key)
            return self.raw_units[idx]

        raise KeyError(key)

    def get_target(self, target_id):
        for rec in self.targets:
            if rec.target_id == target_id:
                return rec
        raise KeyError(f"No target with target_id={target_id}")

    def summary(self):
        lines = [
            f"P4ParticleTarget('{self.fname}')",
            f"  type       = {self.type}",
            f"  dver       = {self.dver}",
            f"  time       = {self.time}",
            f"  geom       = {self.geom}",
            f"  ntargets   = {self.ntargets}",
            f"  nquant     = {self.nquant}",
            "  fields:",
        ]

        for raw, name, unit in zip(self.raw_names, self.names, self.raw_units):
            lines.append(f"    {name:12s} ({unit})   raw='{raw}'")

        lines.append("  targets:")
        for i, rec in enumerate(self.targets):
            # Optional: include a quick diagnostic value
            try:
                emax = np.max(rec.energy)
                e_str = f", maxE={emax:.3g} J/cm^2"
            except Exception:
                e_str = ""

            lines.append(
                f"    [{i}] target_id={rec.target_id}: "
                f"{rec.x_name} ({rec.x_unit}) nx={rec.nx}, "
                f"{rec.y_name} ({rec.y_unit}) ny={rec.ny}"
                f"{e_str}"
            )

        return "\n".join(lines)