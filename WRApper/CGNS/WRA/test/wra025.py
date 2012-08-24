import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll28.hdf',Mll.MODE_READ)

a.gopath('/Base/Zone 01/GridCoordinates/CoordinateX')
p=a.conversion_info()
print p
t=a.conversion_read()
print t

