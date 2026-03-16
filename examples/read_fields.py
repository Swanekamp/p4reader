from p4reader import P4Fields

FLDS = P4Fields("flds0001.p4")
SCLR = P4Fields("sclr0001.p4")
print("time:", FLDS.time)
print("grid:", FLDS.r.shape, FLDS.z.shape)
print("Jz shape:", FLDS.Jz.shape)
print("Ez max:", FLDS.Ez.max())
print("rho shape:", SCLR.rho.shape)
