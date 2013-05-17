#  -------------------------------------------------------------------------
#  pyCGNS - Python package for CFD General Notation Sys
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
import CGNS.WRA.mll as Mll
import numpy as N

print 'CGNS.WRA.mll','#153 - bcdataset_info/bcdataset_read'

# ----------------------------------------------------------------------
a=Mll.pyCGNS('tmp/testmll53.hdf',Mll.MODE_READ)
a.gopath('/Base/family/family BC')
d=a.bcdataset_info()
t=a.bcdataset_read(1)
a.close()
