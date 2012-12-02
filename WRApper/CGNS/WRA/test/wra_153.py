import CGNS.WRA.mll as Mll
import numpy as N

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll53.hdf',Mll.MODE_READ)

a.gopath('/Base/family/family BC')
d=a.bcdataset_info()
print d

t=a.bcdataset_read(1)
print t

a.close()
