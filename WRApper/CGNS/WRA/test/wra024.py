import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll27.hdf',Mll.MODE_READ)

a.gopath('/Base/Zone 01/GridCoordinates/CoordinateX')
p=a.nexponents()
print p
t=a.exponents_info()
print t
o=a.expfull_read()
print o

a.gopath('/Base/Zone 01/GridCoordinates/CoordinateY')
p=a.nexponents()
print p
t=a.exponents_info()
print t
o=a.expfull_read()
print o

a.gopath('/Base/Zone 01/GridCoordinates/CoordinateZ')
p=a.nexponents()
print p
