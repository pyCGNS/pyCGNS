import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll45.hdf',Mll.MODE_READ)

n=a.bc_area_read(1,1,1)
print n

a.close()
