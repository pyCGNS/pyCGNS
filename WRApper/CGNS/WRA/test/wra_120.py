import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll23.hdf',Mll.MODE_READ)

a.gopath("/Base")
p=a.ndescriptors()
print p
a.close()
