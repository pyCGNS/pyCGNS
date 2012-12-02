import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll52.hdf',Mll.MODE_READ)

a.gopath('/Base/Zone 01/GridCoordinates')
n=a.is_link()
print n

a.close()

