import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll31.hdf',Mll.MODE_READ)
a.gopath('/Base/Zone 01')
b=a.famname_read()
print b
a.close()
