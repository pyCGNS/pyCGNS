import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll29.hdf',Mll.MODE_READ)

a.gopath('/Base/Zone 01/Initialize')
p=a.gridlocation_read()
print p
