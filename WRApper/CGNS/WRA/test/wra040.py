import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll41.hdf',Mll.MODE_READ)

n=a.gravity_read(1)
print n

a.close()
