import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll30.hdf',Mll.MODE_READ)

a.gopath('/Base/Zone 01/ZoneBC/BC')
p=a.ptset_info()
print p

