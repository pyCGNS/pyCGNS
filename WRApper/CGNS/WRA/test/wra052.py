import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll51.hdf',Mll.MODE_READ)

d=a.cell_dim(1)
print d

t=a.index_dim(1,1)
print t

a.close()

