import struct
import numpy as np
import struct

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

        # Determine global size
        max_r = 0
        max_z = 0

        for b in self.blocks:
            max_r = max(max_r, b["iR"] + b["nI"])
            max_z = max(max_z, b["kR"] + b["nK"])

        self.nr = max_r
        self.nz = max_z

        self.r = np.zeros(self.nr)
        self.z = np.zeros(self.nz)

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

        for b in self.blocks:

            i0 = b["iR"]
            i1 = i0 + b["nI"]

            k0 = b["kR"]
            k1 = k0 + b["nK"]

            self.r[i0:i1] = b["x"]
            self.z[k0:k1] = b["z"]

            for name in self.names:

                if self.mode == "vector":
                    local = b[name].reshape((b["nK"], b["nI"], 3))
                    global_data[name][k0:k1, i0:i1, :] = local
                else:
                    local = b[name].reshape((b["nK"], b["nI"]))
                    global_data[name][k0:k1, i0:i1] = local

        # Attach to object
        for name in self.names:
            full = global_data[name]
            setattr(self, name, full)

            if self.mode == "vector":
                setattr(self, name + "r", full[:, :, 0])
                setattr(self, name + "y", full[:, :, 1])
                setattr(self, name + "z", full[:, :, 2])


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
    """

    def __init__(self, fname):
        self.fname = fname
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

        segments = np.array(segments)

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