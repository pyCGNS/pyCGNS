import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll30.hdf',Mll.MODE_READ)

a.gopath('/Base/Zone 01')
p=a.ordinal_read()
print p
