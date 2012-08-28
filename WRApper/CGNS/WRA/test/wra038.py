import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll39.hdf',Mll.MODE_READ)

n=a.biter_read(1)
print n

a.close()
